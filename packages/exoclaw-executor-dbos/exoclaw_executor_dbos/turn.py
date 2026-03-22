"""Durable turn — runs an agent turn as a DBOS workflow.

Equivalent to standd_agent's _run_turn() but using DBOS instead of Temporal.
The agent loop runs inside the workflow, with each chat/tool call as a step.
"""

from __future__ import annotations

from typing import Any

import structlog
from dbos import DBOS
from exoclaw.agent.conversation import Conversation
from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.protocol import Tool
from exoclaw.providers.protocol import LLMProvider

from .executor import DBOSExecutor

logger = structlog.get_logger()


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
    # These are set by the caller before invoking the workflow
    # via the module-level _TurnContext.
    provider = _TurnContext.provider
    conversation = _TurnContext.conversation
    tools = _TurnContext.tools
    on_tool_calls = _TurnContext.on_tool_calls
    on_tool_result = _TurnContext.on_tool_result
    on_pre_tool = _TurnContext.on_pre_tool

    assert provider is not None, "provider must be set via set_turn_context()"
    assert conversation is not None, "conversation must be set via set_turn_context()"

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
        on_tool_calls=on_tool_calls,
        on_tool_result=on_tool_result,
        on_pre_tool=on_pre_tool,
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


class _TurnContext:
    """Module-level holder for non-serializable turn dependencies.

    Set these before calling run_durable_turn(). They can't be passed
    as workflow arguments because they aren't serializable.
    """

    provider: LLMProvider | None = None
    conversation: Conversation | None = None
    tools: list[Tool] | None = None
    on_tool_calls: Any = None
    on_tool_result: Any = None
    on_pre_tool: Any = None


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
    for all turns and workflow recovery.
    """
    _TurnContext.provider = provider
    _TurnContext.conversation = conversation
    _TurnContext.tools = tools
    _TurnContext.on_tool_calls = on_tool_calls
    _TurnContext.on_tool_result = on_tool_result
    _TurnContext.on_pre_tool = on_pre_tool
