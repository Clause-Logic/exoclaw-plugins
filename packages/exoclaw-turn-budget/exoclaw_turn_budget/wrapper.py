"""BudgetWrapper — single LLMProvider wrapper used at both turn and daily levels.

Behavior on every ``chat()`` call:

1. Auto-reset the tracker if its boundary has rolled (daily only — turn
   resets are policy-driven).
2. If the tracker is already at the limit before this call:
   - ``CUTOFF``    → return a synthetic LLMResponse containing the cutoff
                     message and stop, *without* calling the inner provider.
                     If the same call that crossed 100% also crossed a
                     warning threshold (e.g. 90%) that hadn't fired yet,
                     prepend that notice to the synthetic content so the
                     transcript shows the climb instead of jumping
                     straight from 80% to "exhausted".
   - ``FALLBACK``  → rewrite the ``model`` argument to the configured
                     fallback model and forward.
   - ``WARN``      → inject the cutoff message as a one-time user notice
                     and forward unchanged.
   - ``OBSERVE``   → forward unchanged.
3. Inject any pending threshold warning (50/80/90%) as a ``user``-role
   message at the end of ``messages``. Threshold warnings fire at most
   once per crossing.
4. If utilization is past ``tool_strip_threshold`` (turn tracker only,
   opt-in), inject the strip notice and either drop ``tools`` entirely
   (``tool_strip_disallow=()``) or filter it down to the tools NOT in
   that list. Forces the model to text-respond instead of issuing more
   tool_calls until cutoff.
5. Forward to the inner provider.
6. Record ``response.usage`` (and the *actual* model used after any
   fallback rewrite) into the tracker.

The ``model`` rewrite + ``primary_models`` filtering on
``DailyBudgetTracker.record`` together ensure tokens spent on the fallback
model don't deplete the daily budget that triggered the fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from exoclaw.providers.types import LLMResponse

from exoclaw_turn_budget.enforcement import Enforcement

if TYPE_CHECKING:  # pragma: no cover (runtime)
    from exoclaw.providers.protocol import LLMProvider
    from exoclaw.providers.types import ResponseFormat

    from exoclaw_turn_budget.tracker import BudgetTracker


def _tool_name(tool: object) -> str | None:
    """Best-effort tool-name extraction across provider conventions.

    OpenAI-style tool dicts wrap the name under ``function.name``;
    Anthropic-style tools (and exoclaw's flat dicts) put it at the top
    level. Returning ``None`` for unrecognized shapes keeps
    ``tool_strip_disallow`` filtering safe — an unknown-shape tool
    falls through and is preserved rather than silently dropped.
    """
    if not isinstance(tool, dict):
        return None
    # ``cast`` rather than annotated assignment — ty's narrowed
    # ``dict[Unknown, Unknown]`` isn't assignable to a more specific
    # ``dict[str, object]`` even though the runtime check guarantees it.
    tool_dict = cast("dict[str, object]", tool)
    fn = tool_dict.get("function")
    if isinstance(fn, dict):
        fn_dict = cast("dict[str, object]", fn)
        nested_name = fn_dict.get("name")
        if isinstance(nested_name, str):
            return nested_name
    flat_name = tool_dict.get("name")
    return flat_name if isinstance(flat_name, str) else None


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

    async def close(self) -> None:
        """Delegate close to the inner provider when present.

        Concrete providers (e.g. ``OpenAIStreamingProvider``) own HTTP
        clients that need to be released — without this delegation,
        wrapping leaks the inner client's resources at shutdown.
        """
        inner_close = getattr(self._inner, "close", None)
        if inner_close is None:
            return
        result = inner_close()
        # Inner ``close`` is conventionally async on this codebase, but
        # be defensive: if a sync implementation slips through, don't
        # try to await its return value.
        if hasattr(result, "__await__"):
            await result

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
                # ``force=True`` pulls any threshold warning that was
                # crossed by the same call that pushed past 100% — the
                # tracker normally suppresses threshold warnings once
                # at_limit is true, so without the force the highest
                # unreported threshold would get stranded forever.
                stranded = self._tracker.consume_threshold_warning(force=True)
                cutoff_msg = self._tracker.at_limit_message()
                content = (stranded + "\n\n" + cutoff_msg) if stranded else cutoff_msg
                return LLMResponse(
                    content=content,
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

        # --- Tool stripping (opt-in escalation between warning + cutoff)
        # ``getattr`` + ``callable`` rather than a ``hasattr`` + protocol
        # check so this is a no-op for tracker types that don't expose
        # the turn-only API (DailyBudgetTracker today; the shared
        # ``BudgetTracker`` protocol stays slim instead of growing
        # methods only one tracker implements).
        forwarded_tools = tools
        should_strip = getattr(self._tracker, "should_strip_tools", None)
        if tools is not None and callable(should_strip) and should_strip():
            tracker_cfg = getattr(self._tracker, "config", None)
            disallow = getattr(tracker_cfg, "tool_strip_disallow", ())
            strip_message = getattr(self._tracker, "tool_strip_message", None)
            if callable(strip_message):
                messages = list(messages) + [{"role": "user", "content": strip_message()}]
            if disallow:
                kept = [t for t in tools if _tool_name(t) not in disallow]
                forwarded_tools = kept if kept else None
            else:
                forwarded_tools = None

        # --- Forward ----------------------------------------------------
        response = await self._inner.chat(
            messages=messages,
            tools=forwarded_tools,
            model=used_model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        )

        # Resolve to a concrete model name for tracking purposes —
        # callers commonly omit ``model`` (the agent loop relies on the
        # inner provider's default), but ``DailyBudgetTracker`` with
        # ``primary_models`` set ignores ``None`` and would silently
        # undercount. Forward the original (possibly None) value to the
        # inner provider so its own defaulting still applies.
        tracked_model = used_model if used_model is not None else self._inner.get_default_model()
        self._tracker.record(
            response.usage if response.usage else None,
            tracked_model,
        )
        return response
