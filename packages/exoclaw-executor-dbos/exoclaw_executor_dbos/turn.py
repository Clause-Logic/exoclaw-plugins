"""Durable turn — runs an agent turn as a DBOS workflow.

When the AgentLoop calls process_turn(), the DBOSExecutor intercepts it
and runs it inside this @DBOS.workflow(). Every LLM call and tool execution
within the turn is a DBOS step, so if the process restarts, DBOS
replays completed steps and continues from where it left off.
"""

from __future__ import annotations

from typing import Any

from dbos import DBOS

# ── Module-level globals (set once at startup, available during recovery) ────
# These are plain globals, not ContextVars, because the loop reference is
# per-process and must survive thread/context switches during DBOS recovery.

_loop: Any = None
_on_progress: Any = None


def set_loop_context(loop: Any) -> None:
    """Store the AgentLoop reference for durable turn recovery.

    Call once at startup, after creating the AgentLoop. On recovery,
    DBOS re-enters run_durable_turn() and uses this loop to call
    _process_turn_inline().
    """
    global _loop
    _loop = loop


@DBOS.workflow()
async def run_durable_turn(
    session_id: str,
    message: str,
    channel: str = "",
    chat_id: str = "",
    media: list[str] | None = None,
    plugin_context: list[str] | None = None,
    model: str | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Run one full agent turn as a durable DBOS workflow.

    Delegates to AgentLoop._process_turn_inline(), which calls build_prompt,
    _run_agent_loop (with durable chat/tool steps), and record.

    ``model`` overrides the loop's default model for this turn; ``None``
    inherits. The override is part of the workflow argument set so replays
    on crash recovery reuse the same model.

    If the process restarts, DBOS replays completed steps and continues.
    Returns ``(final_content, new_messages)``.
    """
    loop = _loop
    if loop is None:
        raise RuntimeError("AgentLoop not set — call set_loop_context() at startup")

    on_progress = _on_progress

    return await loop._process_turn_inline(
        session_id,
        message,
        channel=channel or None,
        chat_id=chat_id or None,
        media=media,
        plugin_context=plugin_context,
        on_progress=on_progress,
        model=model,
    )
