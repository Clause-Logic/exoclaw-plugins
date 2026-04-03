"""Durable turn — runs an agent turn as a DBOS workflow.

When the AgentLoop calls process_turn(), the DBOSExecutor intercepts it
and runs it inside this @DBOS.workflow(). Every LLM call and tool execution
within process_turn is a DBOS step, so if the process restarts, DBOS
replays completed steps and continues from where it left off.
"""

from __future__ import annotations

import contextvars
from typing import Any

from dbos import DBOS

# ── Per-process context (set once at startup, available during recovery) ────

_ctx_loop: contextvars.ContextVar[Any] = contextvars.ContextVar("_ctx_loop", default=None)
_ctx_on_progress: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_ctx_on_progress", default=None
)


def set_loop_context(loop: Any) -> None:
    """Store the AgentLoop reference for durable turn recovery.

    Call once at startup, after creating the AgentLoop. On recovery,
    DBOS re-enters run_durable_turn() and uses this loop to call
    process_turn().
    """
    _ctx_loop.set(loop)


@DBOS.workflow()
async def run_durable_turn(
    session_id: str,
    message: str,
    channel: str = "",
    chat_id: str = "",
    media: list[str] | None = None,
    plugin_context: list[str] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Run one full agent turn as a durable DBOS workflow.

    Delegates to AgentLoop.process_turn(), which calls build_prompt,
    _run_agent_loop (with durable chat/tool steps), and record.

    If the process restarts, DBOS replays completed steps and continues.
    Returns ``(final_content, new_messages)``.
    """
    loop = _ctx_loop.get()
    if loop is None:
        raise RuntimeError("AgentLoop not set — call set_loop_context() at startup")

    on_progress = _ctx_on_progress.get()

    return await loop._process_turn_inline(
        session_id,
        message,
        channel=channel or None,
        chat_id=chat_id or None,
        media=media,
        plugin_context=plugin_context,
        on_progress=on_progress,
    )
