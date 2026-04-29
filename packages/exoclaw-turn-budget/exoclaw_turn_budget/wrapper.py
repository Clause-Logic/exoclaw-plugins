"""BudgetWrapper — single LLMProvider wrapper used at both turn and daily levels.

Behavior on every ``chat()`` call:

1. Auto-reset the tracker if its boundary has rolled (daily only — turn
   resets are policy-driven).
2. If the tracker is already at the limit before this call:
   - ``CUTOFF``    → return a synthetic LLMResponse containing the cutoff
                     message and stop, *without* calling the inner provider.
   - ``FALLBACK``  → rewrite the ``model`` argument to the configured
                     fallback model and forward.
   - ``WARN``      → inject the cutoff message as a one-time user notice
                     and forward unchanged.
   - ``OBSERVE``   → forward unchanged.
3. Inject any pending threshold warning (50/80/90%) as a ``user``-role
   message at the end of ``messages``. Threshold warnings fire at most
   once per crossing.
4. Forward to the inner provider.
5. Record ``response.usage`` (and the *actual* model used after any
   fallback rewrite) into the tracker.

The ``model`` rewrite + ``primary_models`` filtering on
``DailyBudgetTracker.record`` together ensure tokens spent on the fallback
model don't deplete the daily budget that triggered the fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exoclaw.providers.types import LLMResponse

from exoclaw_turn_budget.enforcement import Enforcement

if TYPE_CHECKING:  # pragma: no cover (runtime)
    from exoclaw.providers.protocol import LLMProvider
    from exoclaw.providers.types import ResponseFormat

    from exoclaw_turn_budget.tracker import BudgetTracker


class BudgetWrapper:
    """Generic budget-aware wrapper around any ``LLMProvider``."""

    def __init__(
        self,
        inner: "LLMProvider",
        tracker: "BudgetTracker",
        enforcement: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        # Default ``enforcement`` and ``fallback_model`` from the tracker's
        # config when not explicitly passed, so callers only need to set
        # them in one place. Explicit args still win.
        cfg = getattr(tracker, "config", None)
        if enforcement is None and cfg is not None:
            enforcement = getattr(cfg, "enforcement", Enforcement.OBSERVE)
        if enforcement is None:
            enforcement = Enforcement.OBSERVE
        if fallback_model is None and cfg is not None:
            fallback_model = getattr(cfg, "fallback_model", None)
        if enforcement == Enforcement.FALLBACK and not fallback_model:
            raise ValueError(
                "Enforcement.FALLBACK requires a fallback_model.",
            )
        self._inner = inner
        self._tracker = tracker
        self._enforcement = enforcement
        self._fallback_model = fallback_model

    def get_default_model(self) -> str:
        return self._inner.get_default_model()

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        response_format: "ResponseFormat | None" = None,
    ) -> LLMResponse:
        self._tracker.maybe_auto_reset()

        at_limit = self._tracker.is_at_limit()
        used_model = model

        # --- Limit-hit actions ------------------------------------------
        if at_limit:
            if self._enforcement == Enforcement.CUTOFF:
                # Refuse the call. Return a synthetic response that the
                # agent loop reads as a normal "no tool calls" finish, so
                # the loop ends with the cutoff message as final content.
                return LLMResponse(
                    content=self._tracker.at_limit_message(),
                    finish_reason="stop",
                    usage={},
                )
            if self._enforcement == Enforcement.FALLBACK:
                used_model = self._fallback_model
            if self._enforcement == Enforcement.WARN:
                cutoff_warning = self._tracker.consume_at_limit_warning()
                if cutoff_warning:
                    # Plain ``+`` rather than ``[*messages, ...]`` —
                    # MicroPython 1.27 doesn't accept starred unpacking
                    # in list literals (raises ``SyntaxError: *x must be
                    # assignment target``).
                    messages = list(messages) + [{"role": "user", "content": cutoff_warning}]

        # --- Threshold warnings (50/80/90%) -----------------------------
        threshold_warning = self._tracker.consume_threshold_warning()
        if threshold_warning:
            messages = list(messages) + [{"role": "user", "content": threshold_warning}]

        # --- Forward ----------------------------------------------------
        response = await self._inner.chat(
            messages=messages,
            tools=tools,
            model=used_model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        )

        self._tracker.record(
            response.usage if response.usage else None,
            used_model,
        )
        return response
