"""Configuration dataclasses for turn-level and daily-level budgets.

Dual-class pattern: ``@dataclass`` decorator on CPython, hand-written
``__init__`` on MicroPython (which strips annotations at compile time so
the runtime ``@dataclass`` decorator would produce an empty class).
"""

from __future__ import annotations

from exoclaw._compat import IS_MICROPYTHON

_DEFAULT_WARNING_TEMPLATE = (
    "[Budget notice] You've used {pct}% of your {scope} budget "
    "({used}/{cap} {unit}). "
    "Wrap up your current line of work and respond to the user soon."
)
_DEFAULT_CUTOFF_TEMPLATE = (
    "[Budget exhausted] {scope} budget reached "
    "({used}/{cap} {unit}). "
    "Stopping here — try breaking the task into smaller steps."
)
_DEFAULT_FALLBACK_TEMPLATE = (
    "[Budget exhausted] {scope} budget reached "
    "({used}/{cap} {unit}). "
    "Switching to fallback model for the rest of the {scope}."
)
_DEFAULT_TOOL_STRIP_TEMPLATE = (
    "[Budget critical] You've used {pct}%+ of your {scope} budget "
    "({used}/{cap} {unit}). {disallow_clause} for this response — "
    "wrap up and reply to the user now."
)


if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass

    from exoclaw_turn_budget.enforcement import Enforcement

    @dataclass
    class TurnBudgetConfig:
        """Per-turn budget caps + warning thresholds + enforcement strategy.

        A "turn" is one user query → final response loop. ``iteration_budget``
        and ``token_budget`` are consumed simultaneously; whichever exhausts
        first triggers enforcement. Set either field to ``None`` to disable
        that axis.

        ``tool_strip_threshold`` is an optional escalation point sitting
        between the highest warning threshold and the hard cutoff: once
        utilization crosses it, the wrapper drops the tools listed in
        ``tool_strip_disallow`` (or all tools when the list is empty) on
        every forwarded call. That forces the model into a text-only
        response so it actually wraps up instead of issuing more tool
        calls until the cutoff. Set to ``None`` to disable.

        ``cached_token_weight`` scales the contribution of cached prompt
        tokens (``usage["cached_tokens"]``) to the chargeable total —
        defaults to ``0.1`` because Anthropic and most OpenAI-compatible
        providers price cache reads at ~10% of fresh input. Set to ``1.0``
        to restore the legacy behavior where cached tokens count at full
        weight. Providers that don't surface a ``cached_tokens`` key are
        unaffected.
        """

        iteration_budget: int | None = 50
        token_budget: int | None = 1_500_000
        warning_thresholds: tuple[float, ...] = (0.5, 0.8, 0.9)
        enforcement: str = Enforcement.CUTOFF
        fallback_model: str | None = None
        warning_template: str = _DEFAULT_WARNING_TEMPLATE
        cutoff_template: str = _DEFAULT_CUTOFF_TEMPLATE
        tool_strip_threshold: float | None = None
        tool_strip_disallow: tuple[str, ...] = ()
        tool_strip_template: str = _DEFAULT_TOOL_STRIP_TEMPLATE
        cached_token_weight: float = 0.1
        model_weights: dict = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            if self.model_weights is None:
                self.model_weights = {}

    @dataclass
    class DailyBudgetConfig:
        """Per-day token cap with model fallback or full cutoff.

        Tracks tokens used by the configured *primary models* over a calendar
        day (UTC, offset by ``reset_hour_utc``). When the budget is exhausted,
        the wrapper takes the configured action; tokens spent on the fallback
        model do not count toward the budget.

        ``cached_token_weight`` mirrors the turn-config field — see
        ``TurnBudgetConfig`` for the rationale.
        """

        daily_budget: int = 35_000_000
        primary_models: tuple[str, ...] = ()
        warning_thresholds: tuple[float, ...] = ()
        enforcement: str = Enforcement.FALLBACK
        fallback_model: str | None = None
        reset_hour_utc: int = 0
        warning_template: str = _DEFAULT_WARNING_TEMPLATE
        cutoff_template: str = _DEFAULT_CUTOFF_TEMPLATE
        fallback_template: str = _DEFAULT_FALLBACK_TEMPLATE
        cached_token_weight: float = 0.1

else:  # pragma: no cover (cpython)
    from exoclaw_turn_budget.enforcement import Enforcement

    class TurnBudgetConfig:
        """MicroPython fallback — plain class with hand-written ``__init__``.
        Same shape as the CPython ``@dataclass`` branch above."""

        def __init__(
            self,
            iteration_budget: int | None = 50,
            token_budget: int | None = 1_500_000,
            warning_thresholds: tuple[float, ...] = (0.5, 0.8, 0.9),
            enforcement: str = Enforcement.CUTOFF,
            fallback_model: str | None = None,
            warning_template: str = _DEFAULT_WARNING_TEMPLATE,
            cutoff_template: str = _DEFAULT_CUTOFF_TEMPLATE,
            tool_strip_threshold: float | None = None,
            tool_strip_disallow: tuple[str, ...] = (),
            tool_strip_template: str = _DEFAULT_TOOL_STRIP_TEMPLATE,
            cached_token_weight: float = 0.1,
            model_weights: dict | None = None,
        ) -> None:
            self.iteration_budget = iteration_budget
            self.token_budget = token_budget
            self.warning_thresholds = warning_thresholds
            self.enforcement = enforcement
            self.fallback_model = fallback_model
            self.warning_template = warning_template
            self.cutoff_template = cutoff_template
            self.tool_strip_threshold = tool_strip_threshold
            self.tool_strip_disallow = tool_strip_disallow
            self.tool_strip_template = tool_strip_template
            self.cached_token_weight = cached_token_weight
            self.model_weights = model_weights or {}

    class DailyBudgetConfig:
        """MicroPython fallback — plain class with hand-written ``__init__``.
        Same shape as the CPython ``@dataclass`` branch above."""

        def __init__(
            self,
            daily_budget: int = 35_000_000,
            primary_models: tuple[str, ...] = (),
            warning_thresholds: tuple[float, ...] = (),
            enforcement: str = Enforcement.FALLBACK,
            fallback_model: str | None = None,
            reset_hour_utc: int = 0,
            warning_template: str = _DEFAULT_WARNING_TEMPLATE,
            cutoff_template: str = _DEFAULT_CUTOFF_TEMPLATE,
            fallback_template: str = _DEFAULT_FALLBACK_TEMPLATE,
            cached_token_weight: float = 0.1,
        ) -> None:
            self.daily_budget = daily_budget
            self.primary_models = primary_models
            self.warning_thresholds = warning_thresholds
            self.enforcement = enforcement
            self.fallback_model = fallback_model
            self.reset_hour_utc = reset_hour_utc
            self.warning_template = warning_template
            self.cutoff_template = cutoff_template
            self.fallback_template = fallback_template
            self.cached_token_weight = cached_token_weight
