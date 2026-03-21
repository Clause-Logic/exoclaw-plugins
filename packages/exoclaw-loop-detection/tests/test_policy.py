"""Tests for LoopDetectionPolicy."""

from __future__ import annotations

from exoclaw_loop_detection import LoopDetectionConfig, LoopDetectionPolicy


class TestShouldContinue:
    async def test_allows_diverse_tool_calls(self) -> None:
        """Many different tool calls should never be blocked."""
        policy = LoopDetectionPolicy(LoopDetectionConfig(global_circuit_breaker=1000))
        for i in range(100):
            policy.record(f"tool_{i}", {"arg": str(i)})
        assert await policy.should_continue(100, []) is True

    async def test_allows_below_threshold(self) -> None:
        """Repeats below critical_threshold are allowed."""
        cfg = LoopDetectionConfig(critical_threshold=5)
        policy = LoopDetectionPolicy(cfg)
        for _ in range(4):
            policy.record("search", {"q": "cats"})
        assert await policy.should_continue(4, []) is True

    async def test_blocks_repeat_at_threshold(self) -> None:
        """Same tool+args repeated critical_threshold times triggers stop."""
        cfg = LoopDetectionConfig(critical_threshold=5)
        policy = LoopDetectionPolicy(cfg)
        for _ in range(5):
            policy.record("search", {"q": "cats"})
        assert await policy.should_continue(5, []) is False

    async def test_repeat_resets_on_different_call(self) -> None:
        """A different call in between resets the streak."""
        cfg = LoopDetectionConfig(critical_threshold=5)
        policy = LoopDetectionPolicy(cfg)
        for _ in range(4):
            policy.record("search", {"q": "cats"})
        policy.record("read_file", {"path": "/tmp/x"})
        for _ in range(4):
            policy.record("search", {"q": "cats"})
        assert await policy.should_continue(9, []) is True

    async def test_blocks_ping_pong(self) -> None:
        """Alternating A-B-A-B pattern triggers stop."""
        cfg = LoopDetectionConfig(critical_threshold=6, detect_repeat=False)
        policy = LoopDetectionPolicy(cfg)
        for _ in range(4):
            policy.record("search", {"q": "cats"})
            policy.record("read_file", {"path": "/tmp/x"})
        assert await policy.should_continue(8, []) is False

    async def test_global_circuit_breaker(self) -> None:
        """Global circuit breaker fires regardless of pattern."""
        cfg = LoopDetectionConfig(global_circuit_breaker=10)
        policy = LoopDetectionPolicy(cfg)
        # All diverse calls — no pattern
        for i in range(10):
            policy.record(f"tool_{i}", {"i": i})
        assert await policy.should_continue(10, []) is False

    async def test_empty_history_allows(self) -> None:
        """No history recorded yet — should continue."""
        policy = LoopDetectionPolicy()
        assert await policy.should_continue(0, []) is True

    async def test_repeat_detection_disabled(self) -> None:
        """When detect_repeat=False, repeats are allowed."""
        cfg = LoopDetectionConfig(critical_threshold=3, detect_repeat=False)
        policy = LoopDetectionPolicy(cfg)
        for _ in range(5):
            policy.record("search", {"q": "cats"})
        assert await policy.should_continue(5, []) is True

    async def test_ping_pong_detection_disabled(self) -> None:
        """When detect_ping_pong=False, alternating patterns are allowed."""
        cfg = LoopDetectionConfig(critical_threshold=4, detect_ping_pong=False)
        policy = LoopDetectionPolicy(cfg)
        for _ in range(4):
            policy.record("search", {"q": "cats"})
            policy.record("read_file", {"path": "/tmp/x"})
        assert await policy.should_continue(8, []) is True


class TestOnLimitReached:
    async def test_circuit_breaker_message(self) -> None:
        """Circuit breaker message mentions iteration count."""
        cfg = LoopDetectionConfig(global_circuit_breaker=10)
        policy = LoopDetectionPolicy(cfg)
        msg = await policy.on_limit_reached(10, [])
        assert "10" in msg
        assert "runaway" in msg.lower()

    async def test_repeat_message(self) -> None:
        """Repeat detection message names the tool."""
        policy = LoopDetectionPolicy()
        for _ in range(5):
            policy.record("web_search", {"q": "cats"})
        msg = await policy.on_limit_reached(5, [])
        assert "web_search" in msg
        assert "loop" in msg.lower()

    async def test_empty_history_message(self) -> None:
        """Fallback message when no history."""
        policy = LoopDetectionPolicy()
        msg = await policy.on_limit_reached(0, [])
        assert "progress" in msg.lower()


class TestRecord:
    def test_history_trimmed_to_size(self) -> None:
        """History is trimmed to history_size."""
        cfg = LoopDetectionConfig(history_size=5)
        policy = LoopDetectionPolicy(cfg)
        for i in range(10):
            policy.record("tool", {"i": i})
        assert len(policy._history) == 5

    def test_reset_clears_history(self) -> None:
        """reset() clears all recorded history."""
        policy = LoopDetectionPolicy()
        policy.record("tool", {"x": 1})
        policy.reset()
        assert len(policy._history) == 0
