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
    FileBudgetStore,
    InMemoryBudgetStore,
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
    seen_tools: list[list[dict[str, object]] | None] = field(default_factory=list)
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
        self.seen_tools.append(list(tools) if tools is not None else None)
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

    def test_jump_past_multiple_thresholds_fires_highest(self) -> None:
        """When utilization jumps from below 50% to 100% in a single
        call (e.g. ``iteration_budget=1``), we want the *highest* crossed
        threshold — not "50%" warning when we're actually at 100%."""
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(
                iteration_budget=1,
                token_budget=None,
                warning_thresholds=(0.5, 0.8, 0.9),
            )
        )
        # Below at_limit so the suppression doesn't kick in: first record
        # would push to 1/1 = at_limit. Use a no-record path: directly
        # bump iteration count via record() and then reset is_at_limit
        # by upping the budget. Easiest is to just test with a 2-iter
        # budget and jump halfway in one go.
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(
                iteration_budget=10,
                token_budget=1000,
                warning_thresholds=(0.5, 0.8, 0.9),
            )
        )
        # Tokens jump straight to 90% on first iteration.
        tracker.record({"total_tokens": 900})
        msg = tracker.consume_threshold_warning()
        assert msg is not None
        assert "90%" in msg
        # And subsequent calls don't fire 50% or 80% retroactively.
        assert tracker.consume_threshold_warning() is None

    def test_threshold_warnings_suppressed_when_at_limit(self) -> None:
        """Once the budget is exhausted, the at-limit message takes over
        — threshold warnings shouldn't surface a stale "50%" when we're
        actually at 100%."""
        tracker = TurnBudgetTracker(
            TurnBudgetConfig(
                iteration_budget=1,
                warning_thresholds=(0.5,),
            )
        )
        tracker.record({"total_tokens": 0})  # at_limit immediately
        assert tracker.is_at_limit() is True
        assert tracker.consume_threshold_warning() is None

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

    async def test_cutoff_surfaces_stranded_threshold_warning(self) -> None:
        """If the same call that crosses 100% also crosses 80%/90% from
        below, the threshold notice would otherwise be stranded forever
        — the next call's cutoff would jump straight from "you crossed
        50%" to "exhausted". CUTOFF prepends the highest unfired threshold
        warning to its synthetic content so the agent sees the climb."""
        cfg = TurnBudgetConfig(
            iteration_budget=10,
            token_budget=1000,
            warning_thresholds=(0.5, 0.8, 0.9),
            enforcement=Enforcement.CUTOFF,
        )
        tracker = TurnBudgetTracker(cfg)
        # Burn 50% of tokens — fires the 50% warning on the next call.
        inner = FakeProvider(usage={"total_tokens": 500})
        wrapper = BudgetWrapper(inner, tracker)
        await wrapper.chat(messages=[{"role": "user", "content": "first"}])
        # Now jump straight from 50% to 110% in a single call.
        inner.usage = {"total_tokens": 600}
        await wrapper.chat(messages=[{"role": "user", "content": "huge"}])
        assert tracker.is_at_limit() is True
        # Next call CUTOFFs. Synthetic content should contain BOTH the
        # 90% threshold warning (highest unfired, crossed by the same
        # call that exhausted) and the cutoff message.
        response = await wrapper.chat(messages=[{"role": "user", "content": "more"}])
        content = response.content or ""
        assert "90%" in content
        assert "Budget exhausted" in content

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

    async def test_close_delegates_to_inner(self) -> None:
        """Inner providers (e.g. ``OpenAIStreamingProvider``) own HTTP
        clients that need to be released at shutdown — the wrapper has
        to forward ``close()`` so those resources don't leak."""

        @dataclass
        class ClosableFake(FakeProvider):
            close_calls: int = 0

            async def close(self) -> None:
                self.close_calls += 1

        inner = ClosableFake()
        wrapper = BudgetWrapper(inner, TurnBudgetTracker(TurnBudgetConfig()))
        await wrapper.close()
        assert inner.close_calls == 1

    async def test_close_no_op_when_inner_lacks_close(self) -> None:
        """``FakeProvider`` has no ``close`` — wrapper should silently
        no-op rather than raising AttributeError."""
        wrapper = BudgetWrapper(FakeProvider(), TurnBudgetTracker(TurnBudgetConfig()))
        await wrapper.close()  # must not raise

    async def test_tracking_falls_back_to_inner_default_model(self) -> None:
        """When the caller omits ``model`` (common — agent loop relies
        on the inner provider's defaulting), the daily tracker would
        ignore the call (``model=None`` is treated as "not a primary").
        Wrapper resolves to ``inner.get_default_model()`` for tracking
        so the daily budget actually counts those tokens."""
        cfg = DailyBudgetConfig(
            daily_budget=1000,
            primary_models=("fake-model",),
            enforcement=Enforcement.OBSERVE,  # focus on tracking, not fallback
        )
        tracker = DailyBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 250})  # default_model = "fake-model"
        wrapper = BudgetWrapper(inner, tracker)

        # Caller omits model — without the fix the tracker would see
        # ``None`` and skip the count.
        await wrapper.chat(messages=[{"role": "user", "content": "hi"}])
        assert tracker.total_tokens == 250


class TestBudgetWrapperToolStrip:
    """Optional escalation between warning_thresholds and the hard cutoff
    — once utilization passes ``tool_strip_threshold`` the wrapper stops
    forwarding (some or all) tools so the model has to text-respond
    instead of issuing more tool_calls until the budget is exhausted.
    """

    @staticmethod
    def _exec_tool() -> dict[str, object]:
        # OpenAI-style nested ``function.name`` shape — exercises the
        # _tool_name extraction path most providers use.
        return {"type": "function", "function": {"name": "exec"}}

    @staticmethod
    def _send_tool() -> dict[str, object]:
        # Anthropic/flat shape — verifies _tool_name handles both.
        return {"name": "send_message"}

    async def test_disabled_by_default(self) -> None:
        """Default config has ``tool_strip_threshold=None`` — opt-in.
        Tools always pass through, even past 90%."""
        cfg = TurnBudgetConfig(
            iteration_budget=10,
            warning_thresholds=(0.5,),
            enforcement=Enforcement.OBSERVE,
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker)
        tools = [self._exec_tool()]
        for _ in range(9):
            await wrapper.chat(messages=[{"role": "user", "content": "x"}], tools=tools)
        # All calls saw tools intact.
        assert all(t == tools for t in inner.seen_tools)

    async def test_strips_all_tools_when_disallow_empty(self) -> None:
        cfg = TurnBudgetConfig(
            iteration_budget=10,
            warning_thresholds=(),  # isolate from threshold-warning injection
            enforcement=Enforcement.OBSERVE,
            tool_strip_threshold=0.5,
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker)
        tools = [self._exec_tool(), self._send_tool()]
        # Burn 5/10 iterations to reach the strip threshold.
        for _ in range(5):
            await wrapper.chat(messages=[{"role": "user", "content": "x"}], tools=tools)
        # Next call is past 50% — tools should be dropped entirely.
        await wrapper.chat(messages=[{"role": "user", "content": "x"}], tools=tools)
        # Calls 1-5 had tools; call 6 had None.
        assert inner.seen_tools[:5] == [tools] * 5
        assert inner.seen_tools[5] is None
        # And the strip notice was injected.
        last_msgs = inner.seen_messages[5]
        assert any("Budget critical" in str(m.get("content", "")) for m in last_msgs)
        assert any("Tools are disabled" in str(m.get("content", "")) for m in last_msgs)

    async def test_filters_only_disallowed_tools(self) -> None:
        cfg = TurnBudgetConfig(
            iteration_budget=10,
            warning_thresholds=(),
            enforcement=Enforcement.OBSERVE,
            tool_strip_threshold=0.5,
            tool_strip_disallow=("exec",),
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker)
        tools = [self._exec_tool(), self._send_tool()]
        for _ in range(5):
            await wrapper.chat(messages=[{"role": "user", "content": "x"}], tools=tools)
        await wrapper.chat(messages=[{"role": "user", "content": "x"}], tools=tools)
        # Last call: exec dropped, send_message kept.
        assert inner.seen_tools[5] == [self._send_tool()]
        # Strip notice uses singular grammar for the one-tool case.
        last_msgs = inner.seen_messages[5]
        assert any("Tool exec is disabled" in str(m.get("content", "")) for m in last_msgs)

    async def test_filter_to_empty_passes_none(self) -> None:
        """If the disallow list covers every available tool, fall back to
        passing ``tools=None`` rather than an empty list — providers
        treat ``[]`` and ``None`` differently and an empty list can be
        a validation error on some backends."""
        cfg = TurnBudgetConfig(
            iteration_budget=10,
            warning_thresholds=(),
            enforcement=Enforcement.OBSERVE,
            tool_strip_threshold=0.5,
            tool_strip_disallow=("exec", "send_message"),
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker)
        tools = [self._exec_tool(), self._send_tool()]
        for _ in range(5):
            await wrapper.chat(messages=[{"role": "user", "content": "x"}], tools=tools)
        await wrapper.chat(messages=[{"role": "user", "content": "x"}], tools=tools)
        assert inner.seen_tools[5] is None

    def test_disallow_clause_grammar(self) -> None:
        """Strip-message phrasing should match the disallow-list size:
        singular for one tool, ``X and Y`` for two, Oxford-comma for 3+.
        Lock the wording so Copilot's phrasing nit doesn't regress."""
        from exoclaw_turn_budget.tracker import _disallow_clause

        assert _disallow_clause(()) == "Tools are disabled"
        assert _disallow_clause(("exec",)) == "Tool exec is disabled"
        assert _disallow_clause(("exec", "web")) == "Tools exec and web are disabled"
        assert (
            _disallow_clause(("exec", "web", "shell")) == "Tools exec, web, and shell are disabled"
        )

    async def test_strip_no_op_when_caller_passes_no_tools(self) -> None:
        """Caller-side ``tools=None`` should pass through untouched even
        past the strip threshold — there's nothing to strip and we
        shouldn't synthesize a strip notice for a tool-less call."""
        cfg = TurnBudgetConfig(
            iteration_budget=10,
            warning_thresholds=(),
            enforcement=Enforcement.OBSERVE,
            tool_strip_threshold=0.5,
        )
        tracker = TurnBudgetTracker(cfg)
        inner = FakeProvider(usage={"total_tokens": 0})
        wrapper = BudgetWrapper(inner, tracker)
        for _ in range(5):
            await wrapper.chat(messages=[{"role": "user", "content": "x"}])
        await wrapper.chat(messages=[{"role": "user", "content": "x"}])
        assert inner.seen_tools == [None] * 6
        # No strip notice injected — would just confuse the agent.
        assert all(len(m) == 1 for m in inner.seen_messages)


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


# ── BudgetStateStore — InMemory + File implementations ───────────────────


class TestInMemoryBudgetStore:
    def test_load_returns_none_initially(self) -> None:
        store = InMemoryBudgetStore()
        assert store.load() is None

    def test_save_then_load_roundtrips(self) -> None:
        store = InMemoryBudgetStore()
        store.save({"day_key": 42, "total_tokens": 100})
        assert store.load() == {"day_key": 42, "total_tokens": 100}

    def test_clear_drops_state(self) -> None:
        store = InMemoryBudgetStore()
        store.save({"day_key": 42, "total_tokens": 100})
        store.clear()
        assert store.load() is None


class TestFileBudgetStore:
    def test_load_returns_none_when_file_absent(self, tmp_path) -> None:
        store = FileBudgetStore(tmp_path / "missing.json")
        assert store.load() is None

    def test_save_creates_file_atomically(self, tmp_path) -> None:
        target = tmp_path / "subdir" / "state.json"
        store = FileBudgetStore(target)
        store.save({"day_key": 42, "total_tokens": 1234})

        # Real file replaces — no .tmp left over.
        assert target.exists()
        assert not (target.parent / (target.name + ".tmp")).exists()
        loaded = store.load()
        assert loaded is not None
        assert loaded["day_key"] == 42
        assert loaded["total_tokens"] == 1234

    def test_load_handles_corrupt_file(self, tmp_path) -> None:
        """A torn write or hand-edit shouldn't crash the agent loop —
        load returns None and the caller starts fresh."""
        target = tmp_path / "state.json"
        target.write_text("not json {{{", encoding="utf-8")
        assert FileBudgetStore(target).load() is None

    def test_load_rejects_non_object_root(self, tmp_path) -> None:
        target = tmp_path / "state.json"
        target.write_text("[1, 2, 3]", encoding="utf-8")
        assert FileBudgetStore(target).load() is None

    def test_clear_removes_file(self, tmp_path) -> None:
        target = tmp_path / "state.json"
        store = FileBudgetStore(target)
        store.save({"day_key": 1})
        assert target.exists()
        store.clear()
        assert not target.exists()

    def test_clear_no_op_when_file_absent(self, tmp_path) -> None:
        FileBudgetStore(tmp_path / "missing.json").clear()  # must not raise


# ── DailyBudgetTracker — durability via store ────────────────────────────


class TestDailyTrackerPersistence:
    def test_record_persists_to_store(self) -> None:
        store = InMemoryBudgetStore()
        tracker = DailyBudgetTracker(
            DailyBudgetConfig(daily_budget=1000),
            store=store,
        )
        tracker.record({"total_tokens": 250})

        loaded = store.load()
        assert loaded is not None
        assert loaded["total_tokens"] == 250
        assert "day_key" in loaded

    def test_restart_recovers_token_count(self) -> None:
        """The whole point — a container restart at 14:30 UTC after
        spending 7M shouldn't reset the counter to zero."""
        clock = [1_700_000_000.0]
        store = InMemoryBudgetStore()
        tracker = DailyBudgetTracker(
            DailyBudgetConfig(daily_budget=10_000_000),
            clock=lambda: clock[0],
            store=store,
        )
        tracker.record({"total_tokens": 7_000_000})
        assert tracker.total_tokens == 7_000_000

        # Simulate restart — fresh tracker, same store, same wall-clock day.
        recovered = DailyBudgetTracker(
            DailyBudgetConfig(daily_budget=10_000_000),
            clock=lambda: clock[0],
            store=store,
        )
        assert recovered.total_tokens == 7_000_000

    def test_restart_after_day_rollover_starts_fresh(self) -> None:
        clock = [1_700_000_000.0]
        store = InMemoryBudgetStore()
        tracker = DailyBudgetTracker(
            DailyBudgetConfig(daily_budget=10_000_000),
            clock=lambda: clock[0],
            store=store,
        )
        tracker.record({"total_tokens": 5_000_000})

        # Advance 25 hours — the day rolled, persisted state is stale.
        clock[0] += 25 * 3600
        recovered = DailyBudgetTracker(
            DailyBudgetConfig(daily_budget=10_000_000),
            clock=lambda: clock[0],
            store=store,
        )
        assert recovered.total_tokens == 0

    def test_threshold_warning_state_persists(self) -> None:
        """Warning fired before restart shouldn't refire after."""
        clock = [1_700_000_000.0]
        store = InMemoryBudgetStore()
        cfg = DailyBudgetConfig(daily_budget=1000, warning_thresholds=(0.5,))
        tracker = DailyBudgetTracker(cfg, clock=lambda: clock[0], store=store)
        tracker.record({"total_tokens": 500})
        first = tracker.consume_threshold_warning()
        assert first is not None and "50%" in first

        # Restart — recovered tracker should know the 50% warning fired.
        recovered = DailyBudgetTracker(cfg, clock=lambda: clock[0], store=store)
        assert recovered.consume_threshold_warning() is None

    def test_at_limit_warning_state_persists(self) -> None:
        clock = [1_700_000_000.0]
        store = InMemoryBudgetStore()
        cfg = DailyBudgetConfig(
            daily_budget=100,
            enforcement=Enforcement.WARN,
        )
        tracker = DailyBudgetTracker(cfg, clock=lambda: clock[0], store=store)
        tracker.record({"total_tokens": 100})
        first = tracker.consume_at_limit_warning()
        assert first is not None  # at-limit notice fires once

        recovered = DailyBudgetTracker(cfg, clock=lambda: clock[0], store=store)
        assert recovered.is_at_limit() is True
        assert recovered.consume_at_limit_warning() is None  # already fired

    def test_corrupt_persisted_state_falls_back_to_fresh(self) -> None:
        store = InMemoryBudgetStore()
        # Hand-craft bogus state — wrong day key, malformed warned list.
        store.save(
            {
                "day_key": "not-an-int",
                "total_tokens": "lots",
                "warned_thresholds": "nope",
            }
        )
        tracker = DailyBudgetTracker(DailyBudgetConfig(daily_budget=1000), store=store)
        # Garbled values should not crash; tracker starts at zero.
        assert tracker.total_tokens == 0

    def test_default_store_is_inmemory(self) -> None:
        """Backwards-compatible — passing no store gives the original
        non-durable behavior."""
        tracker = DailyBudgetTracker(DailyBudgetConfig(daily_budget=1000))
        tracker.record({"total_tokens": 500})
        assert tracker.total_tokens == 500
        # No store keyword set — nothing persists.
        recovered = DailyBudgetTracker(DailyBudgetConfig(daily_budget=1000))
        assert recovered.total_tokens == 0


class TestDailyTrackerWithFileStore:
    def test_end_to_end_disk_persistence(self, tmp_path) -> None:
        path = tmp_path / "budget.json"
        cfg = DailyBudgetConfig(daily_budget=1_000_000)
        clock = [1_700_000_000.0]

        t1 = DailyBudgetTracker(cfg, clock=lambda: clock[0], store=FileBudgetStore(path))
        t1.record({"total_tokens": 600_000})
        assert path.exists()

        # Fresh tracker, fresh store object — only the on-disk file
        # bridges the two.
        t2 = DailyBudgetTracker(cfg, clock=lambda: clock[0], store=FileBudgetStore(path))
        assert t2.total_tokens == 600_000
