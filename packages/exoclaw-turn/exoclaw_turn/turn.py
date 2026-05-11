"""``run_turn``: drive one exoclaw agent turn synchronously from library code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.protocol import Tool
from exoclaw.bus.queue import MessageBus
from exoclaw.iteration_policy import IterationPolicy
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import ToolCallRequest

_SESSION_ID = "exoclaw-turn"


class _EphemeralConversation:
    """Minimal in-memory Conversation that drops state at end of turn.

    Builds the prompt as ``[system?, user]`` and discards everything the
    loop produces. Anything that would normally go to disk (assistant
    replies, tool results) is unreachable after ``run_turn`` returns, but
    the loop's ``new_messages`` return is still propagated to the caller
    via ``TurnResult.messages``.
    """

    def __init__(self, system: str | None) -> None:
        self._system = system

    async def build_prompt(
        self,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system_parts: list[str] = []
        if self._system:
            system_parts.append(self._system)
        if plugin_context:
            system_parts.extend(plugin_context)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": message})
        return messages

    async def record(
        self, session_id: str, new_messages: list[dict[str, Any]]
    ) -> None:  # pragma: no cover - intentional no-op
        pass

    async def clear(self, session_id: str) -> bool:  # pragma: no cover - intentional no-op
        return True

    def list_sessions(self) -> list[dict[str, Any]]:  # pragma: no cover - intentional no-op
        return []

    def active_tools(self) -> set[str]:  # pragma: no cover - intentional no-op
        return set()


@dataclass
class TurnResult:
    """The output of one ``run_turn`` invocation."""

    text: str | None
    """Final assistant text. ``None`` if the model finished without
    producing user-visible content (rare; usually only on hard failure)."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    """All new messages produced this turn — user, assistant, and tool
    result messages, in order. The first entry is the user message that
    seeded the turn; the last is typically the final assistant reply."""

    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    """Tool calls the model issued during the turn, in the order they
    were executed. Convenience view — also reconstructable from
    ``messages``."""


def _extract_tool_calls(messages: list[dict[str, Any]]) -> list[ToolCallRequest]:
    """Pull tool calls out of the assistant messages produced this turn.

    The loop records tool calls inline on assistant messages with shape
    ``{"role": "assistant", "tool_calls": [{"id", "function": {"name",
    "arguments"}}]}``. Arguments are JSON strings on the wire; this
    helper parses them back to dicts so the returned ``ToolCallRequest``
    matches what tool implementations actually receive.
    """
    import json

    calls: list[ToolCallRequest] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        raw_calls = msg.get("tool_calls") or []
        for raw in raw_calls:
            fn = raw.get("function", {}) if isinstance(raw, dict) else {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if not name:
                continue
            args_raw = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            except (ValueError, TypeError):
                args = {}
            calls.append(
                ToolCallRequest(
                    id=raw.get("id", "") if isinstance(raw, dict) else "",
                    name=name,
                    arguments=args,
                )
            )
    return calls


async def run_turn(
    *,
    provider: LLMProvider,
    message: str,
    system: str | None = None,
    tools: list[Tool] | None = None,
    model: str | None = None,
    max_iterations: int = 40,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
    iteration_policy: IterationPolicy | None = None,
    media: list[str] | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
) -> TurnResult:
    """Drive one agent turn to completion and return the result.

    Spins up an ephemeral ``AgentLoop`` with a throwaway ``MessageBus``
    and an in-memory ``Conversation`` seeded with ``system`` and ``message``.
    Inherits compaction, loop detection, plugin context collection, tool
    dispatch, and subagent chain tracking from the underlying loop — this
    function adds no agent behaviour of its own, it just assembles the
    loop and reshapes the result.

    Parameters
    ----------
    provider:
        LLM provider (any ``LLMProvider`` — LiteLLM, OpenAI, etc.).
    message:
        The user message that seeds the turn.
    system:
        Optional system prompt. Plugin context collected from tools is
        appended to this string.
    tools:
        Tools the model can call. Each must implement the
        ``exoclaw.agent.tools.protocol.Tool`` protocol — every tool
        plugin in ``exoclaw-plugins`` works unchanged.
    model:
        Override the provider's default model for this turn.
    max_iterations:
        Hard cap on tool-call iterations. Default 40 matches ``AgentLoop``.
    temperature, max_tokens, reasoning_effort:
        Standard LLM parameters, forwarded to the provider.
    iteration_policy:
        Replace the hard ``max_iterations`` counter with a pattern-based
        termination strategy (e.g. ``exoclaw-loop-detection``).
    media:
        Optional list of media references (image/file paths) the
        provider/tools can resolve. Same shape as ``InboundMessage.media``.
    on_progress:
        Optional async callback invoked by the loop for streaming progress.
    """
    bus = MessageBus()
    conversation = _EphemeralConversation(system=system)
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=conversation,
        model=model,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        tools=list(tools) if tools else None,
        iteration_policy=iteration_policy,
    )
    text, new_messages = await loop.process_turn(
        session_id=_SESSION_ID,
        message=message,
        media=media,
        on_progress=on_progress,
    )
    return TurnResult(
        text=text,
        messages=new_messages,
        tool_calls=_extract_tool_calls(new_messages),
    )
