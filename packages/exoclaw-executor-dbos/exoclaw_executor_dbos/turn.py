"""Durable turn — runs an agent turn as a DBOS workflow.

Equivalent to standd_agent's _run_turn() but using DBOS instead of Temporal.
The agent loop runs inside the workflow, with each chat/tool call as a step.
"""

from __future__ import annotations

import contextvars
from typing import Any

from dbos import DBOS
from exoclaw.agent.conversation import Conversation
from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.protocol import Tool
from exoclaw.providers.protocol import LLMProvider

from .executor import DBOSExecutor


class _NullBus:
    """No-op bus — AgentLoop requires one but DBOSExecutor bypasses it."""

    async def publish_inbound(self, msg: Any) -> None:
        pass

    async def publish_outbound(self, msg: Any) -> None:
        pass

    async def get_inbound(self) -> Any:
        raise NotImplementedError

    async def get_outbound(self) -> Any:
        raise NotImplementedError


# ── Per-task context for non-serializable deps ───────────────────────────────
# ContextVars are safe for concurrent workflows — each asyncio Task gets
# its own copy.

_ctx_provider: contextvars.ContextVar[LLMProvider | None] = contextvars.ContextVar(
    "_ctx_provider", default=None
)
_ctx_conversation: contextvars.ContextVar[Conversation | None] = contextvars.ContextVar(
    "_ctx_conversation", default=None
)
_ctx_tools: contextvars.ContextVar[list[Tool] | None] = contextvars.ContextVar(
    "_ctx_tools", default=None
)
_ctx_on_tool_calls: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_ctx_on_tool_calls", default=None
)
_ctx_on_tool_result: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_ctx_on_tool_result", default=None
)
_ctx_on_pre_tool: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_ctx_on_pre_tool", default=None
)


def set_turn_context(
    *,
    provider: LLMProvider,
    conversation: Conversation,
    tools: list[Tool] | None = None,
    on_tool_calls: Any = None,
    on_tool_result: Any = None,
    on_pre_tool: Any = None,
) -> None:
    """Set the non-serializable context for durable turns.

    Call once at startup. The same provider/conversation/tools are used
    for all turns and workflow recovery. Safe for concurrent turns via
    ContextVar inheritance.
    """
    _ctx_provider.set(provider)
    _ctx_conversation.set(conversation)
    _ctx_tools.set(tools)
    _ctx_on_tool_calls.set(on_tool_calls)
    _ctx_on_tool_result.set(on_tool_result)
    _ctx_on_pre_tool.set(on_pre_tool)


@DBOS.workflow()
async def run_durable_turn(
    session_id: str,
    message: str,
    *,
    channel: str = "cli",
    chat_id: str = "direct",
    media: list[str] | None = None,
    plugin_context: list[str] | None = None,
    max_iterations: int = 40,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: str | None = None,
    reasoning_effort: str | None = None,
    turn_context: list[str] | None = None,
    skills: list[str] | None = None,
) -> str | None:
    """Run one full agent turn as a durable DBOS workflow.

    Every LLM call and tool execution within the turn is a DBOS step.
    If the process restarts, DBOS replays completed steps and continues.

    Returns the final assistant content, or None if max iterations reached.
    """
    provider = _ctx_provider.get()
    conversation = _ctx_conversation.get()
    tools = _ctx_tools.get()

    if provider is None:
        raise RuntimeError("provider must be set via set_turn_context()")
    if conversation is None:
        raise RuntimeError("conversation must be set via set_turn_context()")

    executor = DBOSExecutor()

    loop = AgentLoop(
        bus=_NullBus(),  # type: ignore[arg-type]
        provider=provider,
        conversation=conversation,
        executor=executor,
        tools=tools,
        model=model,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        on_tool_calls=_ctx_on_tool_calls.get(),
        on_tool_result=_ctx_on_tool_result.get(),
        on_pre_tool=_ctx_on_pre_tool.get(),
    )

    kwargs: dict[str, Any] = {}
    if skills:
        kwargs["skills"] = skills

    initial = await executor.build_prompt(
        conversation,
        session_id,
        message,
        channel=channel,
        chat_id=chat_id,
        media=media,
        plugin_context=plugin_context,
        turn_context=turn_context,
        **kwargs,
    )

    final_content, _, all_msgs = await loop._run_agent_loop(initial)

    # Persist the turn
    new_msgs = all_msgs[len(initial) - 1 :]
    await executor.record(conversation, session_id, new_msgs)

    return final_content
