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

import contextvars
import dataclasses
import json
from collections.abc import Awaitable, Callable
from typing import Any

from dbos import DBOS, SetWorkflowID
from exoclaw.agent.conversation import Conversation
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import LLMResponse, ToolCallRequest
from uuid_utils import uuid7

# ── Serialization helpers ────────────────────────────────────────────────────


def _response_to_dict(resp: LLMResponse) -> dict[str, Any]:
    return dataclasses.asdict(resp)


def _dict_to_response(d: dict[str, Any]) -> LLMResponse:
    d = dict(d)  # don't mutate caller's dict
    tool_calls = [ToolCallRequest(**tc) for tc in d.pop("tool_calls", [])]
    return LLMResponse(tool_calls=tool_calls, **d)


# ── Per-task context for non-serializable refs ───────────────────────────────
# ContextVars are safe for concurrent workflows — each asyncio Task gets
# its own copy, so parallel turns don't stomp on each other.

_provider_var: contextvars.ContextVar[LLMProvider | None] = contextvars.ContextVar(
    "_provider_var", default=None
)
_registry_var: contextvars.ContextVar[ToolRegistry | None] = contextvars.ContextVar(
    "_registry_var", default=None
)


# ── DBOS step functions ──────────────────────────────────────────────────────
# Module-level so DBOS can register and replay them.


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
    provider = _provider_var.get()
    if provider is None:
        raise RuntimeError("provider not set — call set_turn_context() before running turns")
    tools = json.loads(tools_json) if tools_json else None
    resp = await provider.chat(
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
    ctx_data: dict[str, Any] | None,
) -> str:
    """Durable tool execution. Result is cached by DBOS on completion."""
    registry = _registry_var.get()
    if registry is None:
        raise RuntimeError("registry not set — call set_turn_context() before running turns")
    ctx = ToolContext(**ctx_data) if ctx_data else None
    return await registry.execute(name, params, ctx)


# ── DBOSExecutor ─────────────────────────────────────────────────────────────


class DBOSExecutor:
    """Executor that routes AgentLoop operations through DBOS steps.

    Must be used inside a @DBOS.workflow() — see run_durable_turn().
    Sets ContextVar refs so steps can access provider/registry safely
    across concurrent workflows.
    """

    def __init__(self) -> None:
        # Per-turn message buffer. Mirrors DirectExecutor — the buffer
        # lives for one turn and does not need to be durable across DBOS
        # recovery because run_durable_turn encapsulates the whole turn.
        self._messages: list[dict[str, object]] = []

    def append_messages(self, messages: list[dict[str, object]]) -> None:
        self._messages.extend(messages)

    def load_messages(self) -> list[dict[str, object]]:
        return list(self._messages)

    def set_messages(self, messages: list[dict[str, object]]) -> None:
        self._messages = list(messages)

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
        _provider_var.set(provider)
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
        _registry_var.set(registry)
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
        messages = await conversation.build_prompt(
            session_id,
            message,
            channel=channel,
            chat_id=chat_id,
            media=media,
            plugin_context=plugin_context,
            **kwargs,
        )
        self.set_messages(messages)
        return messages

    async def record(
        self,
        conversation: Conversation,
        session_id: str,
        new_messages: list[dict[str, object]],
    ) -> None:
        await conversation.record(session_id, new_messages)

    async def clear(
        self,
        conversation: Conversation,
        session_id: str,
    ) -> bool:
        return await conversation.clear(session_id)

    async def run_turn(
        self,
        loop: Any,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        on_progress: Any = None,
        **kwargs: Any,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Run a full agent turn inside a DBOS workflow.

        Called by AgentLoop.process_turn() when the executor provides this
        method. Sets the loop context (for crash recovery) and wraps the
        turn in a @DBOS.workflow() so it is recoverable on restart.
        """
        from .turn import run_durable_turn, set_loop_context

        # Ensure the loop reference is available for DBOS recovery
        set_loop_context(loop)

        from . import turn

        turn._on_progress = on_progress

        wfid = f"turn:{session_id}:{uuid7().hex}"
        with SetWorkflowID(wfid):
            return await run_durable_turn(
                session_id,
                message,
                channel=channel or "",
                chat_id=chat_id or "",
                media=media,
                plugin_context=plugin_context,
            )

    async def run_hook(
        self,
        fn: Callable[..., Awaitable[object]],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        return await fn(*args, **kwargs)
