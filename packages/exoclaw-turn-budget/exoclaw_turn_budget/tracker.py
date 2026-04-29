"""Trackers: shared mutable state consulted by the BudgetWrapper.

The wrapper writes (token usage per chat) and reads (pending warnings,
at-limit decisions). Two implementations:

* ``TurnBudgetTracker`` — per-turn state, reset by ``TurnBudgetPolicy`` on
  the first iteration of each turn.
* ``DailyBudgetTracker`` — per-day state in UTC, auto-reset by the wrapper
  when the day boundary is crossed.

Both expose the same minimal interface the wrapper needs.

Day-boundary detection uses ``time.time()`` (seconds since epoch) instead
of ``datetime.now(UTC)`` because MicroPython 1.27's ``datetime`` shim
doesn't ship the ``UTC`` constant or timezone-aware datetime objects.
``time.time()`` is already epoch-based (UTC by definition), so day-key =
``int((t - reset_hour_utc * 3600) // 86400)`` is portable across runtimes.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from exoclaw_turn_budget.config import DailyBudgetConfig, TurnBudgetConfig
from exoclaw_turn_budget.enforcement import Enforcement

if TYPE_CHECKING:  # pragma: no cover (runtime)
    # ``Protocol`` and ``runtime_checkable`` aren't on MicroPython's
    # ``typing`` shim. Type-checking-only import keeps both runtimes
    # happy: ty/mypy see the protocol; MP never tries to load it.
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class BudgetTracker(Protocol):
        """Common interface the BudgetWrapper consumes."""

        def maybe_auto_reset(self) -> None: ...
        def record(self, usage: dict[str, int] | None, model: str | None = None) -> None: ...
        def utilization(self) -> float: ...
        def is_at_limit(self) -> bool: ...
        def consume_threshold_warning(self) -> str | None: ...
        def at_limit_message(self) -> str: ...
        def consume_at_limit_warning(self) -> str | None: ...


_SECONDS_PER_DAY = 86400


def _coerce_total_tokens(usage: "dict[str, int] | None") -> int:
    if not usage:
        return 0
    total = usage.get("total_tokens")
    if total is not None:
        return int(total)
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return int(prompt) + int(completion)


def _format(template: str, scope: str, pct: int, used: int, cap, unit: str) -> str:
    return template.format(
        scope=scope,
        pct=pct,
        used=used,
        cap=cap if cap is not None else "∞",
        unit=unit,
    )


class TurnBudgetTracker:
    """Per-turn iteration + token counter.

    Reset must be triggered externally by ``TurnBudgetPolicy`` when a new
    turn starts (it's the only component that sees the agent loop's
    iteration counter).
    """

    SCOPE = "turn"

    def __init__(self, config: TurnBudgetConfig | None = None) -> None:
        self._config = config if config is not None else TurnBudgetConfig()
        self.total_tokens: int = 0
        self.iterations_seen: int = 0
        self._warned_thresholds: set[float] = set()
        self._at_limit_warning_pending: bool = True

    @property
    def config(self) -> TurnBudgetConfig:
        return self._config

    def reset(self) -> None:
        """Clear per-turn counters. Called by the policy on iteration=0."""
        self.total_tokens = 0
        self.iterations_seen = 0
        self._warned_thresholds.clear()
        self._at_limit_warning_pending = True

    def maybe_auto_reset(self) -> None:
        """Turn boundaries are detected by the policy, not the wrapper."""
        return

    def record(self, usage: "dict[str, int] | None", model: str | None = None) -> None:
        self.iterations_seen += 1
        self.total_tokens += _coerce_total_tokens(usage)

    def utilization(self) -> float:
        cfg = self._config
        ratios: list[float] = []
        if cfg.iteration_budget and cfg.iteration_budget > 0:
            ratios.append(self.iterations_seen / cfg.iteration_budget)
        if cfg.token_budget and cfg.token_budget > 0:
            ratios.append(self.total_tokens / cfg.token_budget)
        return max(ratios) if ratios else 0.0

    def is_at_limit(self) -> bool:
        cfg = self._config
        if cfg.iteration_budget is not None and self.iterations_seen >= cfg.iteration_budget:
            return True
        if cfg.token_budget is not None and self.total_tokens >= cfg.token_budget:
            return True
        return False

    def consume_threshold_warning(self) -> str | None:
        """Return one warning message if a new threshold was crossed."""
        cfg = self._config
        util = self.utilization()
        for threshold in cfg.warning_thresholds:
            if threshold in self._warned_thresholds:
                continue
            if util >= threshold:
                self._warned_thresholds.add(threshold)
                return self._format_at(cfg.warning_template, int(threshold * 100))
        return None

    def at_limit_message(self) -> str:
        return self._format_at(self._config.cutoff_template, 100)

    def consume_at_limit_warning(self) -> str | None:
        """Return the at-limit message exactly once when first exhausted."""
        if self._at_limit_warning_pending and self.is_at_limit():
            self._at_limit_warning_pending = False
            return self.at_limit_message()
        return None

    def _format_at(self, template: str, pct: int) -> str:
        cfg = self._config
        # Use whichever axis is closer to exhaustion for the substitution.
        iter_ratio = self.iterations_seen / cfg.iteration_budget if cfg.iteration_budget else 0.0
        tok_ratio = self.total_tokens / cfg.token_budget if cfg.token_budget else 0.0
        if iter_ratio >= tok_ratio:
            return _format(
                template,
                self.SCOPE,
                pct,
                self.iterations_seen,
                cfg.iteration_budget,
                "iterations",
            )
        return _format(
            template,
            self.SCOPE,
            pct,
            self.total_tokens,
            cfg.token_budget,
            "tokens",
        )


def _day_key(now_epoch: float, reset_hour_utc: int) -> int:
    """Integer day-key (UTC) — same value for all timestamps within one
    rollover window. Offset by ``reset_hour_utc`` so the boundary lands
    at the configured hour instead of midnight UTC.
    """
    return int((now_epoch - reset_hour_utc * 3600) // _SECONDS_PER_DAY)


class DailyBudgetTracker:
    """Cumulative token counter for one calendar day in UTC.

    Auto-resets when the day boundary (``reset_hour_utc``) is crossed.
    Only counts tokens for models in ``primary_models`` (empty = all).
    """

    SCOPE = "daily"

    def __init__(
        self,
        config: DailyBudgetConfig | None = None,
        clock=None,
    ) -> None:
        # ``clock`` is a callable returning epoch seconds — tests inject a
        # deterministic one. Defaults to ``time.time``.
        self._config = config if config is not None else DailyBudgetConfig()
        self._clock = clock if clock is not None else time.time
        self._day = _day_key(self._clock(), self._config.reset_hour_utc)
        self.total_tokens: int = 0
        self._warned_thresholds: set[float] = set()
        self._at_limit_warning_pending: bool = True

    @property
    def config(self) -> DailyBudgetConfig:
        return self._config

    def maybe_auto_reset(self) -> None:
        current = _day_key(self._clock(), self._config.reset_hour_utc)
        if current != self._day:
            self._day = current
            self.total_tokens = 0
            self._warned_thresholds.clear()
            self._at_limit_warning_pending = True

    def _counts_against_budget(self, model: str | None) -> bool:
        primaries = self._config.primary_models
        if not primaries:
            return True
        if model is None:
            return False
        return model in primaries

    def record(self, usage: "dict[str, int] | None", model: str | None = None) -> None:
        if not self._counts_against_budget(model):
            return
        self.total_tokens += _coerce_total_tokens(usage)

    def utilization(self) -> float:
        budget = self._config.daily_budget
        if budget <= 0:
            return 0.0
        return self.total_tokens / budget

    def is_at_limit(self) -> bool:
        return self.total_tokens >= self._config.daily_budget

    def consume_threshold_warning(self) -> str | None:
        cfg = self._config
        util = self.utilization()
        for threshold in cfg.warning_thresholds:
            if threshold in self._warned_thresholds:
                continue
            if util >= threshold:
                self._warned_thresholds.add(threshold)
                return _format(
                    cfg.warning_template,
                    self.SCOPE,
                    int(threshold * 100),
                    self.total_tokens,
                    cfg.daily_budget,
                    "tokens",
                )
        return None

    def at_limit_message(self) -> str:
        cfg = self._config
        template = (
            cfg.fallback_template
            if cfg.enforcement == Enforcement.FALLBACK
            else cfg.cutoff_template
        )
        return _format(
            template,
            self.SCOPE,
            100,
            self.total_tokens,
            cfg.daily_budget,
            "tokens",
        )

    def consume_at_limit_warning(self) -> str | None:
        if self._at_limit_warning_pending and self.is_at_limit():
            self._at_limit_warning_pending = False
            return self.at_limit_message()
        return None
