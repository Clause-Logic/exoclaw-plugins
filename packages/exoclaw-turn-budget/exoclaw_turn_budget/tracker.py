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
from exoclaw_turn_budget.store import InMemoryBudgetStore

if TYPE_CHECKING:  # pragma: no cover (runtime)
    # ``Protocol`` and ``runtime_checkable`` aren't on MicroPython's
    # ``typing`` shim. Type-checking-only import keeps both runtimes
    # happy: ty/mypy see the protocol; MP never tries to load it.
    from typing import Protocol, runtime_checkable

    from exoclaw_turn_budget.store import BudgetStateStore

    @runtime_checkable
    class BudgetTracker(Protocol):
        """Common interface the BudgetWrapper consumes."""

        def maybe_auto_reset(self) -> None: ...
        def record(self, usage: dict[str, int] | None, model: str | None = None) -> None: ...
        def utilization(self) -> float: ...
        def is_at_limit(self) -> bool: ...
        def consume_threshold_warning(self, force: bool = False) -> str | None: ...
        def at_limit_message(self) -> str: ...
        def consume_at_limit_warning(self) -> str | None: ...
        @property
        def last_consumed_threshold(self) -> float | None: ...
        def dominant_axis(self) -> tuple[int, int | None, str]: ...


_SECONDS_PER_DAY = 86400


def _coerce_total_tokens(
    usage: "dict[str, int] | None",
    cached_token_weight: float = 1.0,
) -> int:
    """Sum chargeable tokens from a provider ``usage`` dict.

    Cached prompt tokens (``cached_tokens`` / ``cache_read_input_tokens``)
    are charged at ``cached_token_weight`` × their raw count — provider
    pricing for cache reads is roughly an order of magnitude cheaper than
    fresh input, so a weight of ``0.1`` keeps the budget tied to actual
    cost. ``cache_creation_input_tokens`` is *not* discounted: cache writes
    are billed at full input price (Anthropic) or above (1.25× on OpenAI).
    Providers that don't report a cache field fall through unchanged.
    """
    if not usage:
        return 0
    cached = usage.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0
    total = usage.get("total_tokens")
    if total is not None:
        raw = int(total)
    else:
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        raw = int(prompt) + int(completion)
    if not cached:
        return raw
    discount = int(cached) - int(int(cached) * cached_token_weight)
    chargeable = raw - discount
    return chargeable if chargeable > 0 else 0


def _format(template: str, scope: str, pct: int, used: int, cap, unit: str, **extra) -> str:
    # ``extra`` carries optional placeholders (e.g. ``disallow_clause``
    # for the tool-strip template) — pass through as kwargs so callers
    # can grow the template surface without touching every call site.
    return template.format(
        scope=scope,
        pct=pct,
        used=used,
        cap=cap if cap is not None else "∞",
        unit=unit,
        **extra,
    )


def _disallow_clause(disallow: "tuple[str, ...]") -> str:
    if not disallow:
        return "Tools are disabled"
    if len(disallow) == 1:
        return "Tool " + disallow[0] + " is disabled"
    if len(disallow) == 2:
        joined = disallow[0] + " and " + disallow[1]
    else:
        joined = ", ".join(disallow[:-1]) + ", and " + disallow[-1]
    return "Tools " + joined + " are disabled"


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
        # Records the threshold value that the most recent
        # ``consume_threshold_warning`` call fired for, so consumers
        # (e.g. ``BudgetWrapper``'s ``on_threshold_crossed`` hook) can
        # learn which threshold was crossed without re-deriving it.
        self._last_consumed_threshold: float | None = None

    @property
    def config(self) -> TurnBudgetConfig:
        return self._config

    def reset(self) -> None:
        """Clear per-turn counters. Called by the policy on iteration=0."""
        self.total_tokens = 0
        self.iterations_seen = 0
        self._warned_thresholds.clear()
        self._at_limit_warning_pending = True
        self._last_consumed_threshold = None

    def maybe_auto_reset(self) -> None:
        """Turn boundaries are detected by the policy, not the wrapper."""
        return

    def record(self, usage: "dict[str, int] | None", model: str | None = None) -> None:
        self.iterations_seen += 1
        chargeable = _coerce_total_tokens(usage, self._config.cached_token_weight)
        model_weight = self._config.model_weights.get(model, 1.0) if model else 1.0
        self.total_tokens += int(chargeable * model_weight)

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

    def consume_threshold_warning(self, force: bool = False) -> str | None:
        """Return one warning message if a new threshold was crossed.

        Once at the limit (``is_at_limit()`` true) threshold warnings are
        suppressed — the at-limit message takes over instead, so the agent
        doesn't get a stale "you're at 50%" notice when it's actually at
        100%. When utilization jumps past multiple thresholds in a single
        call (e.g. ``iteration_budget=1`` going 0→1), the *highest*
        crossed threshold fires and all lower thresholds are marked as
        already-fired so they don't surface later.

        ``force=True`` bypasses the at-limit suppression. The wrapper's
        ``CUTOFF`` branch passes it so the highest threshold a turn
        crossed (e.g. 90%) still surfaces in the synthetic cutoff
        response — without it, when the same call that crosses 90% also
        crosses 100%, the 90% notice gets stranded forever.
        """
        if not force and self.is_at_limit():
            return None
        cfg = self._config
        util = self.utilization()
        # Reverse iteration so the highest crossed threshold wins.
        # ``sorted(..., reverse=True)`` rather than ``reversed()`` —
        # callers may pass thresholds out of order.
        sorted_desc = sorted(cfg.warning_thresholds, reverse=True)
        for threshold in sorted_desc:
            if threshold in self._warned_thresholds:
                continue
            if util >= threshold:
                # Mark this threshold AND every lower threshold as
                # already fired — they were all crossed simultaneously
                # and shouldn't pop later.
                for t in cfg.warning_thresholds:
                    if t <= threshold:
                        self._warned_thresholds.add(t)
                self._last_consumed_threshold = threshold
                return self._format_at(cfg.warning_template, int(threshold * 100))
        return None

    @property
    def last_consumed_threshold(self) -> float | None:
        """The threshold value the most recent ``consume_threshold_warning``
        fired for, or ``None`` if no warning has fired yet (or since the
        last ``reset``). Used by the wrapper to surface the crossed
        threshold to ``on_threshold_crossed`` hooks."""
        return self._last_consumed_threshold

    def dominant_axis(self) -> "tuple[int, int | None, str]":
        """Return ``(used, cap, unit)`` for whichever axis is closer to
        exhaustion — same selection ``_format_at`` uses internally.
        Exposed so the wrapper can pass these values to observability
        hooks without duplicating the comparison logic."""
        cfg = self._config
        iter_ratio = self.iterations_seen / cfg.iteration_budget if cfg.iteration_budget else 0.0
        tok_ratio = self.total_tokens / cfg.token_budget if cfg.token_budget else 0.0
        if iter_ratio >= tok_ratio:
            return self.iterations_seen, cfg.iteration_budget, "iterations"
        return self.total_tokens, cfg.token_budget, "tokens"

    def should_strip_tools(self) -> bool:
        """``True`` once utilization crosses ``tool_strip_threshold``.

        Returns ``False`` when the threshold is unset (default — the
        feature is opt-in). Fires every call past the threshold rather
        than once-only because the strip needs to suppress *every*
        subsequent tool_calls response, not just the first.
        """
        cfg = self._config
        threshold = getattr(cfg, "tool_strip_threshold", None)
        if threshold is None:
            return False
        return self.utilization() >= threshold

    def tool_strip_message(self) -> str:
        cfg = self._config
        threshold = cfg.tool_strip_threshold
        return self._format_at(
            cfg.tool_strip_template,
            int(threshold * 100) if threshold is not None else 0,
            disallow_clause=_disallow_clause(cfg.tool_strip_disallow),
        )

    def at_limit_message(self) -> str:
        return self._format_at(self._config.cutoff_template, 100)

    def consume_at_limit_warning(self) -> str | None:
        """Return the at-limit message exactly once when first exhausted."""
        if self._at_limit_warning_pending and self.is_at_limit():
            self._at_limit_warning_pending = False
            return self.at_limit_message()
        return None

    def _format_at(self, template: str, pct: int, **extra) -> str:
        used, cap, unit = self.dominant_axis()
        return _format(template, self.SCOPE, pct, used, cap, unit, **extra)


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
        store: "BudgetStateStore | None" = None,
    ) -> None:
        # ``clock`` is a callable returning epoch seconds — tests inject a
        # deterministic one. Defaults to ``time.time``.
        # ``store`` lets callers swap the in-memory default for a durable
        # backing (``FileBudgetStore``, the DBOS-backed store, etc.) so a
        # container restart at 14:30 UTC after spending 7M of 10M doesn't
        # reset the counter back to zero. See ``store.py``.
        self._config = config if config is not None else DailyBudgetConfig()
        self._clock = clock if clock is not None else time.time
        self._day = _day_key(self._clock(), self._config.reset_hour_utc)
        self.total_tokens: int = 0
        self._warned_thresholds: set[float] = set()
        self._at_limit_warning_pending: bool = True
        # Same role as TurnBudgetTracker._last_consumed_threshold — surfaces
        # which threshold the most recent ``consume_threshold_warning`` fired
        # for. Not persisted: the persisted threshold dedup is enough to
        # prevent re-firing, and hooks would already have observed the prior
        # crossing in the previous process run.
        self._last_consumed_threshold: float | None = None
        self._store: BudgetStateStore = store if store is not None else InMemoryBudgetStore()
        self._restore()

    @property
    def config(self) -> DailyBudgetConfig:
        return self._config

    def _restore(self) -> None:
        """Hydrate from the store. Discards any state whose ``day_key``
        doesn't match the current day — that's just the persisted
        equivalent of ``maybe_auto_reset``."""
        data = self._store.load()
        if not data:
            return
        # ``data.get`` returns ``object`` from the protocol, so each
        # field needs an isinstance check before coercion. Catching
        # everything here is intentional: corrupt state should fall
        # back to a fresh tracker, not crash the agent loop on every
        # chat() call until the file is hand-fixed.
        raw_day = data.get("day_key")
        if not isinstance(raw_day, (int, float, str)):
            return
        try:
            day_key = int(raw_day)
        except (TypeError, ValueError):
            return
        if day_key != self._day:
            # Persisted state is from a previous day — drop it (and let
            # the next save overwrite).
            return
        raw_tokens = data.get("total_tokens", 0)
        if isinstance(raw_tokens, (int, float, str)):
            try:
                self.total_tokens = int(raw_tokens)
            except (TypeError, ValueError):
                self.total_tokens = 0
        warned = data.get("warned_thresholds")
        if isinstance(warned, list):
            try:
                self._warned_thresholds = {
                    float(x) for x in warned if isinstance(x, (int, float, str))
                }
            except (TypeError, ValueError):
                self._warned_thresholds = set()
        self._at_limit_warning_pending = bool(data.get("at_limit_warning_pending", True))

    def _persist(self) -> None:
        """Snapshot current state to the store. Called after every
        mutation so a SIGKILL only loses the most recent record()."""
        self._store.save(
            {
                "day_key": self._day,
                "total_tokens": self.total_tokens,
                "warned_thresholds": sorted(self._warned_thresholds),
                "at_limit_warning_pending": self._at_limit_warning_pending,
            }
        )

    def maybe_auto_reset(self) -> None:
        current = _day_key(self._clock(), self._config.reset_hour_utc)
        if current != self._day:
            self._day = current
            self.total_tokens = 0
            self._warned_thresholds.clear()
            self._at_limit_warning_pending = True
            self._last_consumed_threshold = None
            self._persist()

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
        self.total_tokens += _coerce_total_tokens(usage, self._config.cached_token_weight)
        self._persist()

    def utilization(self) -> float:
        budget = self._config.daily_budget
        if budget <= 0:
            return 0.0
        return self.total_tokens / budget

    def is_at_limit(self) -> bool:
        return self.total_tokens >= self._config.daily_budget

    def consume_threshold_warning(self, force: bool = False) -> str | None:
        # Highest-crossed-threshold-wins + suppress-when-at-limit logic
        # mirrors ``TurnBudgetTracker.consume_threshold_warning`` — see
        # that docstring for rationale (including the ``force`` flag).
        if not force and self.is_at_limit():
            return None
        cfg = self._config
        util = self.utilization()
        sorted_desc = sorted(cfg.warning_thresholds, reverse=True)
        for threshold in sorted_desc:
            if threshold in self._warned_thresholds:
                continue
            if util >= threshold:
                for t in cfg.warning_thresholds:
                    if t <= threshold:
                        self._warned_thresholds.add(t)
                self._last_consumed_threshold = threshold
                self._persist()
                return _format(
                    cfg.warning_template,
                    self.SCOPE,
                    int(threshold * 100),
                    self.total_tokens,
                    cfg.daily_budget,
                    "tokens",
                )
        return None

    @property
    def last_consumed_threshold(self) -> float | None:
        """The threshold value the most recent ``consume_threshold_warning``
        fired for, or ``None`` if no warning has fired yet (or since the
        last day-boundary auto-reset)."""
        return self._last_consumed_threshold

    def dominant_axis(self) -> "tuple[int, int | None, str]":
        """Return ``(used, cap, unit)`` for the daily tracker — always
        the token axis since iterations aren't part of the daily budget.
        Mirrors ``TurnBudgetTracker.dominant_axis`` so the wrapper can
        consume a uniform shape from either tracker.
        """
        return self.total_tokens, self._config.daily_budget, "tokens"

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
            self._persist()
            return self.at_limit_message()
        return None
