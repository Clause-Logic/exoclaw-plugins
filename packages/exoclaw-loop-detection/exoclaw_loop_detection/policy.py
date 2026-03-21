"""IterationPolicy implementation with pattern-based loop detection."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from exoclaw_loop_detection.config import LoopDetectionConfig


@dataclass
class _ToolCall:
    """Fingerprint of a single tool invocation."""

    name: str
    args_hash: str


class LoopDetectionPolicy:
    """Replaces the hard ``max_iterations`` cap with pattern-based detection.

    Instead of killing the loop after N iterations regardless of progress,
    this policy watches for degenerate patterns:

    * **Repeat detection** — the same tool with the same arguments called
      ``critical_threshold`` times in a row.
    * **Ping-pong detection** — two distinct tool calls alternating for
      ``critical_threshold`` cycles (A→B→A→B…).
    * **Global circuit breaker** — absolute safety net after
      ``global_circuit_breaker`` iterations, regardless of pattern.

    Productive runs that call many *different* tools are never interrupted.
    """

    def __init__(self, config: LoopDetectionConfig | None = None) -> None:
        self._config = config or LoopDetectionConfig()
        self._history: list[_ToolCall] = []

    @staticmethod
    def _fingerprint(name: str, args: dict[str, object] | list[object] | object) -> str:
        return json.dumps({"n": name, "a": args}, sort_keys=True, ensure_ascii=False)

    def record(self, name: str, args: dict[str, object] | list[object] | object) -> None:
        """Record a tool call. Call this from your executor or hook."""
        self._history.append(_ToolCall(name=name, args_hash=self._fingerprint(name, args)))
        if len(self._history) > self._config.history_size:
            self._history = self._history[-self._config.history_size :]

    def reset(self) -> None:
        """Clear history (e.g. between sessions)."""
        self._history.clear()

    # -- IterationPolicy protocol -----------------------------------------

    async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
        """Return False when a degenerate pattern is detected or circuit breaker fires."""
        cfg = self._config

        if iteration >= cfg.global_circuit_breaker:
            return False

        history = self._history[-cfg.history_size :]
        if not history:
            return True

        # --- Repeat detection: same tool+args N times in a row ---
        if cfg.detect_repeat and len(history) >= cfg.critical_threshold:
            last = history[-1]
            streak = 0
            for entry in reversed(history):
                if entry.name == last.name and entry.args_hash == last.args_hash:
                    streak += 1
                else:
                    break
            if streak >= cfg.critical_threshold:
                return False

        # --- Ping-pong detection: A B A B A B ... ---
        if cfg.detect_ping_pong and len(history) >= cfg.critical_threshold:
            if len(history) >= 4:
                a, b = history[-2], history[-1]
                if a.name != b.name or a.args_hash != b.args_hash:
                    cycles = 0
                    for i in range(len(history) - 1, 0, -2):
                        if (
                            history[i].name == b.name
                            and history[i].args_hash == b.args_hash
                            and history[i - 1].name == a.name
                            and history[i - 1].args_hash == a.args_hash
                        ):
                            cycles += 2
                        else:
                            break
                    if cycles >= cfg.critical_threshold:
                        return False

        return True

    async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str:
        """Return a diagnostic message explaining why the loop was stopped."""
        cfg = self._config
        history = self._history[-cfg.history_size :]

        if iteration >= cfg.global_circuit_breaker:
            return (
                f"I've made {iteration} tool calls without finishing. "
                "Stopping to avoid runaway behavior. "
                "Try breaking the task into smaller steps."
            )

        if history:
            counts = Counter((e.name, e.args_hash) for e in history)
            (top_name, _), top_count = counts.most_common(1)[0]
            return (
                f"I appear to be stuck in a loop — called `{top_name}` "
                f"with the same arguments {top_count} times. "
                "Try rephrasing or breaking the task into smaller steps."
            )

        return "I've been iterating without making progress. Please try a different approach."
