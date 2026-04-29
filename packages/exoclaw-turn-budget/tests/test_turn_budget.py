"""Tests for budget trackers, BudgetWrapper, and TurnBudgetPolicy."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from exoclaw.providers.types import LLMResponse, ResponseFormat
from exoclaw_turn_budget import (
    BudgetWrapper,
    DailyBudgetConfig,
    DailyBudgetTracker,
    Enforcement,
    TurnBudgetConfig,
    TurnBudgetPolicy,
    TurnBudgetTracker,
)


@dataclass
class FakeProvider:
    """Records every chat() call so tests can assert on injected messages
    and the model parameter (which the wrapper may rewrite for fallback).
    """

    usage: dict[str, int] = field(default_factory=lambda: {"total_tokens": 100})
    seen_messages: list[list[dict[str, object]]] = field(default_factory=list)
    seen_models: list[str | None] = field(default_factory=list)
    content: str = "ok"

    def get_default_model(self) -> str:
        return "fake-model"

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        self.seen_messages.append(list(messages))
        self.seen_models.append(model)
        return LLMResponse(content=self.content, usage=dict(self.usage))


# ── TurnBudgetTracker ────────────────────────────────────────────────────


class TestTurnBudgetTracker:
    def test_record_increments_tokens_and_iterations(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig())
        tracker.record({"total_tokens": 1234})
        tracker.record({"total_tokens": 500})
        assert tracker.total_tokens == 1734
        assert tracker.iterations_seen == 2

    def test_record_falls_back_to_prompt_plus_completion(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig())
        tracker.record({"prompt_tokens": 100, "completion_tokens": 50})
        assert tracker.total_tokens == 150

    def test_record_handles_missing_usage(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig())
        tracker.record(None)
        tracker.record({})
        assert tracker.total_tokens == 0
        assert tracker.iterations_seen == 2

    def test_reset_clears_state(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig(warning_thresholds=(0.5,)))
        tracker.record({"total_tokens": 1_000_000})
        _ = tracker.consume_threshold_warning()
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.iterations_seen == 0
        # Threshold warning rearmed after reset.
        tracker.record({"total_tokens": 1_000_000})
        assert tracker.consume_threshold_warning() is not None

    def test_exhausted_on_iteration_budget(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig(iteration_budget=3, token_budget=None))
        for _ in range(3):
            tracker.record({"total_tokens": 1})
        assert tracker.is_at_limit() is True

    def test_exhausted_on_token_budget(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig(iteration_budget=None, token_budget=1000))
        tracker.record({"total_tokens": 1000})
        assert tracker.is_at_limit() is True

    def test_threshold_fires_once_then_silent(self) -> None:
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(iteration_budget=10, warning_thresholds=(0.5,))
        )
        for _ in range(5):
            tracker.record({"total_tokens": 0})
        first = tracker.consume_threshold_warning()
        second = tracker.consume_threshold_warning()
        assert first is not None and "50%" in first
        assert second is None

    def test_subsequent_thresholds_fire_in_order(self) -> None:
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(iteration_budget=10, warning_thresholds=(0.5, 0.8, 0.9))
        )
        for _ in range(5):
            tracker.record({"total_tokens": 0})
        assert "50%" in (tracker.consume_threshold_warning() or "")
        for _ in range(3):
            tracker.record({"total_tokens": 0})
        assert "80%" in (tracker.consume_threshold_warning() or "")
        tracker.record({"total_tokens": 0})
        assert "90%" in (tracker.consume_threshold_warning() or "")

    def test_token_axis_drives_warning_when_higher(self) -> None:
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(
                iteration_budget=100,
                token_budget=1000,
                warning_thresholds=(0.5,),
            )
        )
        tracker.record({"total_tokens": 800})  # tokens at 80%, iters at 1%
        msg = tracker.consume_threshold_warning()
        assert msg is not None
        assert "tokens" in msg


# ── DailyBudgetTracker ───────────────────────────────────────────────────


class TestDailyBudgetTracker:
    def test_record_only_counts_primary_models(self) -> None:
        tracker = DailyBudgetTracker(
            DailyBudgetConfig(daily_budget=1000, primary_models=("glm-5.1",))
        )
        tracker.record({"total_tokens": 500}, model="glm-5.1")
        tracker.record({"total_tokens": 9999}, model="minimax/m2")
        tracker.record({"total_tokens": 200}, model=None)
        assert tracker.total_tokens == 500

    def test_record_counts_all_models_when_primary_unset(self) -> None:
        tracker = DailyBudgetTracker(DailyBudgetConfig(daily_budget=1000, primary_models=()))
        tracker.record({"total_tokens": 100}, model="a")
        tracker.record({"total_tokens": 200}, model="b")
        assert tracker.total_tokens == 300

    def test_auto_resets_at_day_boundary(self) -> None:
        clock_value = [1_700_000_000.0]  # arbitrary epoch — middle of a day

        def fake_clock() -> float:
            return clock_value[0]

        tracker = DailyBudgetTracker(
            DailyBudgetConfig(daily_budget=1_000_000),
            clock=fake_clock,
        )
        tracker.record({"total_tokens": 500_000})
        tracker.maybe_auto_reset()
        assert tracker.total_tokens == 500_000

        # Advance 25 hours — definitely a new day-key.
        clock_value[0] += 25 * 3600
        tracker.maybe_auto_reset()
        assert tracker.total_tokens == 0
        # Threshold warnings rearmed too.
        tracker.record({"total_tokens": 999_999})
        assert tracker.is_at_limit() is False  # 999,999 < 1,000,000

    def test_does_not_reset_within_same_day(self) -> None:
        clock_value = [1_700_000_000.0]

        def fake_clock() -> float:
            return clock_value[0]

        tracker = DailyBudgetTracker(
            DailyBudgetConfig(daily_budget=1_000_000),
            clock=fake_clock,
        )
        tracker.record({"total_tokens": 500_000})
        clock_value[0] += 1 * 3600  # +1 hour
        tracker.maybe_auto_reset()
        assert tracker.total_tokens == 500_000

    def test_at_limit_uses_fallback_template_for_fallback_enforcement(self) -> None:
        tracker = DailyBudgetTracker(
            DailyBudgetConfig(
                daily_budget=100,
                enforcement=Enforcement.FALLBACK,
                fallback_model="cheap",
            )
        )
        tracker.record({"total_tokens": 100}, model=None)
        msg = tracker.at_limit_message()
        assert "fallback" in msg.lower()


# ── BudgetWrapper ────────────────────────────────────────────────────────


class TestBudgetWrapperBasic:
    async def test_records_usage_after_chat(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig())
        inner = FakeProvider(usage={"total_tokens": 250})
        wrapper = BudgetWrapper(inner, tracker)

        await wrapper.chat(messages=[{"role": "user", "content": "hi"}])

        assert tracker.iterations_seen == 1
        assert tracker.total_tokens == 250

    async def test_does_not_inject_warning_under_threshold(self) -> None:
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(iteration_budget=100, warning_thresholds=(0.5,))
        )
        inner = FakeProvider(usage={"total_tokens": 1})
        wrapper = BudgetWrapper(inner, tracker)

        await wrapper.chat(messages=[{"role": "user", "content": "hi"}])
        assert len(inner.seen_messages[0]) == 1

    async def test_injects_warning_when_threshold_crossed(self) -> None:
        cfg = TurnBudgetConfig(
            iteration_budget=2,
            warning_thresholds=(0.5,),
            enforcement=Enforcement.OBSERVE,
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 1})
        wrapper = BudgetWrapper(inner, tracker)

        await wrapper.chat(messages=[{"role": "user", "content": "first"}])
        await wrapper.chat(messages=[{"role": "user", "content": "second"}])

        second_call = inner.seen_messages[1]
        assert len(second_call) == 2
        assert "50%" in str(second_call[1]["content"])

    def test_get_default_model_delegates(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig())
        inner = FakeProvider()
        wrapper = BudgetWrapper(inner, tracker)
        assert wrapper.get_default_model() == "fake-model"


class TestBudgetWrapperEnforcement:
    async def test_cutoff_returns_synthetic_response(self) -> None:
        cfg = TurnBudgetConfig(iteration_budget=1, enforcement=Enforcement.CUTOFF)
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker)

        # First call: tracker empty, normal response.
        await wrapper.chat(messages=[{"role": "user", "content": "go"}])
        # Second call: at limit. Wrapper should NOT call inner.
        response = await wrapper.chat(messages=[{"role": "user", "content": "more"}])

        assert len(inner.seen_messages) == 1  # second call short-circuited
        assert "Budget exhausted" in (response.content or "")
        assert response.has_tool_calls is False

    async def test_fallback_rewrites_model(self) -> None:
        cfg = TurnBudgetConfig(
            iteration_budget=1,
            enforcement=Enforcement.FALLBACK,
            fallback_model="cheap-model",
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker, Enforcement.FALLBACK, "cheap-model")

        await wrapper.chat(messages=[{"role": "user", "content": "go"}], model="primary")
        await wrapper.chat(messages=[{"role": "user", "content": "more"}], model="primary")

        assert inner.seen_models == ["primary", "cheap-model"]

    async def test_warn_at_limit_injects_once(self) -> None:
        cfg = TurnBudgetConfig(
            iteration_budget=1,
            enforcement=Enforcement.WARN,
            warning_thresholds=(),  # isolate the at-limit warning
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker)

        await wrapper.chat(messages=[{"role": "user", "content": "go"}])
        await wrapper.chat(messages=[{"role": "user", "content": "more"}])
        await wrapper.chat(messages=[{"role": "user", "content": "again"}])

        # Second call: at-limit notice injected. Third call: silent.
        assert len(inner.seen_messages[1]) == 2
        assert "Budget exhausted" in str(inner.seen_messages[1][1]["content"])
        assert len(inner.seen_messages[2]) == 1

    async def test_observe_takes_no_action_at_limit(self) -> None:
        cfg = TurnBudgetConfig(
            iteration_budget=1,
            enforcement=Enforcement.OBSERVE,
            warning_thresholds=(),  # isolate at-limit behavior from threshold warnings
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker)

        await wrapper.chat(messages=[{"role": "user", "content": "go"}])
        await wrapper.chat(messages=[{"role": "user", "content": "more"}])

        assert len(inner.seen_messages) == 2
        assert all(len(m) == 1 for m in inner.seen_messages)

    def test_fallback_without_model_raises(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig())
        with pytest.raises(ValueError, match="fallback_model"):
            BudgetWrapper(FakeProvider(), tracker, Enforcement.FALLBACK, None)


class TestDailyBudgetWrapper:
    async def test_fallback_tokens_dont_deplete_budget(self) -> None:
        cfg = DailyBudgetConfig(
            daily_budget=100,
            primary_models=("glm-5.1",),
            enforcement=Enforcement.FALLBACK,
            fallback_model="cheap",
        )
        tracker = DailyBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 50})
        wrapper = BudgetWrapper(inner, tracker, Enforcement.FALLBACK, "cheap")

        # Two primary calls = 100 tokens — at the limit.
        await wrapper.chat(messages=[{"role": "user", "content": "a"}], model="glm-5.1")
        await wrapper.chat(messages=[{"role": "user", "content": "b"}], model="glm-5.1")
        assert tracker.is_at_limit() is True

        # Subsequent calls route to fallback. Fallback tokens should NOT
        # add to the daily budget — primary_models filtering takes care of it.
        for _ in range(5):
            await wrapper.chat(messages=[{"role": "user", "content": "c"}], model="glm-5.1")
        # Still 100, no overflow from fallback usage.
        assert tracker.total_tokens == 100
        # All fallback calls saw the rewritten model.
        assert inner.seen_models[2:] == ["cheap"] * 5


# ── TurnBudgetPolicy ─────────────────────────────────────────────────────


class TestTurnBudgetPolicy:
    async def test_should_continue_initial_call_resets_tracker(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig(iteration_budget=10))
        tracker.record({"total_tokens": 999_999})
        tracker.iterations_seen = 5

        policy = TurnBudgetPolicy(tracker)
        assert await policy.should_continue(0, []) is True
        assert tracker.iterations_seen == 0
        assert tracker.total_tokens == 0

    async def test_cutoff_blocks_when_exhausted(self) -> None:
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(iteration_budget=2, enforcement=Enforcement.CUTOFF)
        )
        policy = TurnBudgetPolicy(tracker)
        await policy.should_continue(0, [])
        tracker.record({"total_tokens": 1})
        tracker.record({"total_tokens": 1})
        assert await policy.should_continue(2, []) is False

    async def test_observe_never_blocks(self) -> None:
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(iteration_budget=1, enforcement=Enforcement.OBSERVE)
        )
        policy = TurnBudgetPolicy(tracker)
        await policy.should_continue(0, [])
        tracker.record({"total_tokens": 1})
        # Even though tracker.is_at_limit() is True, OBSERVE means don't block.
        assert tracker.is_at_limit() is True
        assert await policy.should_continue(1, []) is True

    async def test_on_limit_reached_returns_message(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig(iteration_budget=2))
        policy = TurnBudgetPolicy(tracker)
        await policy.should_continue(0, [])
        tracker.record({"total_tokens": 42})
        tracker.record({"total_tokens": 42})
        msg = await policy.on_limit_reached(2, [])
        assert "Budget exhausted" in msg


# ── End-to-end (wrapper + policy + tracker) ──────────────────────────────


class TestEndToEnd:
    async def test_two_turns_get_independent_budgets(self) -> None:
        tracker = TurnBudgetTracker(TurnBudgetConfig(iteration_budget=2, token_budget=None))
        inner = FakeProvider(usage={"total_tokens": 1})
        wrapper = BudgetWrapper(inner, tracker)
        policy = TurnBudgetPolicy(tracker)

        # Turn 1: exhaust.
        await policy.should_continue(0, [])
        await wrapper.chat(messages=[{"role": "user", "content": "t1.1"}])
        await wrapper.chat(messages=[{"role": "user", "content": "t1.2"}])
        assert await policy.should_continue(2, []) is False

        # Turn 2: should_continue(0, ...) resets the tracker.
        assert await policy.should_continue(0, []) is True
        assert tracker.iterations_seen == 0
        await wrapper.chat(messages=[{"role": "user", "content": "t2.1"}])
        assert await policy.should_continue(1, []) is True
