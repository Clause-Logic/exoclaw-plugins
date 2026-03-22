"""DBOS-backed durable executor for exoclaw.

Drop-in replacement for DirectExecutor. Every LLM call and tool execution
is a DBOS step, automatically checkpointed to SQLite. If the process
restarts mid-turn, DBOS replays completed steps from the journal.

Architecture follows the same pattern as standd_agent's TemporalExecutor:
the agent loop runs inside a @DBOS.workflow(), and each chat/tool call
is a @DBOS.step().

Usage in nanobot wiring:
    from exoclaw_executor_dbos import run_durable_turn, DBOSExecutor

    # In message processing, instead of calling AgentLoop._process_message:
    await run_durable_turn(session_id, message, ...)
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from dbos import DBOS
from exoclaw.agent.conversation import Conversation
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import LLMResponse, ToolCallRequest

logger = structlog.get_logger()


# ── Serialization helpers ────────────────────────────────────────────────────


def _response_to_dict(resp: LLMResponse) -> dict[str, Any]:
    return dataclasses.asdict(resp)


def _dict_to_response(d: dict[str, Any]) -> LLMResponse:
    tool_calls = [ToolCallRequest(**tc) for tc in d.pop("tool_calls", [])]
    return LLMResponse(tool_calls=tool_calls, **d)


# ── DBOS step functions ──────────────────────────────────────────────────────
# These are module-level so DBOS can register and replay them.
# They receive serializable inputs and produce serializable outputs.
# The actual provider/registry references are passed via a module-level holder
# that the executor sets before calling.


class _Refs:
    """Holds non-serializable references for the current turn."""

    provider: LLMProvider | None = None
    registry: ToolRegistry | None = None


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2)
async def _chat_step(
    messages: list[dict[str, Any]],
    tools_json: str | None,
    model: str | None,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Durable LLM call. Result is cached by DBOS on completion."""
    assert _Refs.provider is not None, "provider not set"
    tools = json.loads(tools_json) if tools_json else None
    resp = await _Refs.provider.chat(
        messages=messages,
        tools=tools,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    return _response_to_dict(resp)


@DBOS.step(retries_allowed=True, max_attempts=2, interval_seconds=1)
async def _tool_step(
    name: str,
    params: dict[str, Any],
    ctx_data: dict[str, str] | None,
) -> str:
    """Durable tool execution. Result is cached by DBOS on completion."""
    assert _Refs.registry is not None, "registry not set"
    ctx = ToolContext(**ctx_data) if ctx_data else None
    return await _Refs.registry.execute(name, params, ctx)


# ── DBOSExecutor ─────────────────────────────────────────────────────────────


class DBOSExecutor:
    """Executor that routes AgentLoop operations through DBOS steps.

    Must be used inside a @DBOS.workflow() — see run_durable_turn().
    """

    async def chat(
        self,
        provider: LLMProvider,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        _Refs.provider = provider
        tools_json = json.dumps(tools) if tools else None
        result = await _chat_step(
            messages=list(messages),
            tools_json=tools_json,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return _dict_to_response(result)

    async def execute_tool(
        self,
        registry: ToolRegistry,
        name: str,
        params: dict[str, object],
        ctx: ToolContext | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> str:
        _Refs.registry = registry
        ctx_data = dataclasses.asdict(ctx) if ctx else None
        return await _tool_step(
            name=name,
            params=dict(params),
            ctx_data=ctx_data,
        )

    async def build_prompt(
        self,
        conversation: Conversation,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        **kwargs: list[str] | None,
    ) -> list[dict[str, object]]:
        # build_prompt is not a step — it's cheap and idempotent.
        # We call conversation directly so it can trigger consolidation etc.
        return await conversation.build_prompt(
            session_id,
            message,
            channel=channel,
            chat_id=chat_id,
            media=media,
            plugin_context=plugin_context,
            **kwargs,
        )

    async def record(
        self,
        conversation: Conversation,
        session_id: str,
        new_messages: list[dict[str, object]],
    ) -> None:
        # record is not a step — it's the persistence itself
        await conversation.record(session_id, new_messages)

    async def clear(
        self,
        conversation: Conversation,
        session_id: str,
    ) -> bool:
        return await conversation.clear(session_id)

    async def run_hook(
        self,
        fn: Callable[..., Awaitable[object]],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        return await fn(*args, **kwargs)
