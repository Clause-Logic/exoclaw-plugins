"""Durable turn — runs an agent turn as a DBOS workflow.

When the AgentLoop calls process_turn(), the DBOSExecutor intercepts it
and runs it inside this @DBOS.workflow(). Every LLM call and tool execution
within the turn is a DBOS step, so if the process restarts, DBOS
replays completed steps and continues from where it left off.
"""

from __future__ import annotations

from typing import Any

from dbos import DBOS, Queue
from exoclaw.bus.events import InboundMessage, OutboundMessage

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


@DBOS.step()
async def _check_sent_in_turn_step(chat_id: str) -> bool:
    """Snapshot "did any tool already send a user-facing message to this
    chat during the turn" as a durable step.

    The ``message`` tool tracks ``sent_in_turn`` on its in-memory instance
    and that flag would be lost across a container restart — on recovery
    the journaled ``_tool_step`` outputs are replayed without re-running
    the tool body, so the flag would read False and we'd publish a
    redundant final reply. Wrapping the read in a step captures the
    first-run answer in the journal; replay returns the same boolean.
    """
    loop = _loop
    if loop is None:
        raise RuntimeError("AgentLoop not set — call set_loop_context() at startup")
    return any(getattr(t, "sent_in_turn", False) for t in loop.tools._tools.values())


@DBOS.step()
async def _publish_outbound_step(
    session_id: str,
    channel: str,
    chat_id: str,
    content: str,
) -> None:
    """Publish the final turn reply to the bus as a durable step.

    Wrapping ``bus.publish_outbound`` in a step lets DBOS replay recover
    the send across container restarts — the outer ``_dispatch`` used to
    own this, but the outer coroutine dies on OOM and never re-runs on
    workflow recovery, so the reply would sit in ``workflow_status.output``
    unsent.

    Logging lives inside the step so observability matches reality — on
    replay DBOS returns the journaled completion without re-executing the
    body, so we don't double-log a send that didn't actually happen.

    Idempotency: on partial completion (bus write succeeded, step record
    didn't commit) recovery re-sends, which means one duplicate message.
    Accepting that risk — the send window is ~50 ms and OOMs during it
    are vanishingly rare compared to OOMs during the LLM loop.
    """
    loop = _loop
    if loop is None:
        raise RuntimeError("AgentLoop not set — call set_loop_context() at startup")
    preview = content[:120] + "..." if len(content) > 120 else content
    loop._log.info("response_send", preview=preview)
    await loop.bus.publish_outbound(
        OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            metadata={"session_key": session_id},
        )
    )


@DBOS.workflow()
async def run_durable_turn(
    session_id: str,
    message: str,
    channel: str = "",
    chat_id: str = "",
    media: list[str] | None = None,
    plugin_context: list[str] | None = None,
    model: str | None = None,
    publish_response: bool = False,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Run one full agent turn as a durable DBOS workflow.

    Delegates to AgentLoop._process_turn_inline(), which calls build_prompt,
    _run_agent_loop (with durable chat/tool steps), and record.

    ``model`` overrides the loop's default model for this turn; ``None``
    inherits. The override is part of the workflow argument set so replays
    on crash recovery reuse the same model.

    ``publish_response`` controls whether the workflow publishes the final
    reply to the bus as a durable step. Top-level user turns (via
    ``_dispatch``) set it to True; subagent chains (via ``process_direct``)
    leave it False and let the caller read the returned content.

    If the process restarts, DBOS replays completed steps and continues.
    Returns ``(final_content, new_messages)``.
    """
    loop = _loop
    if loop is None:
        raise RuntimeError("AgentLoop not set — call set_loop_context() at startup")

    on_progress = _on_progress

    final_content, new_msgs = await loop._process_turn_inline(
        session_id,
        message,
        channel=channel or None,
        chat_id=chat_id or None,
        media=media,
        plugin_context=plugin_context,
        on_progress=on_progress,
        model=model,
    )

    if publish_response and channel and chat_id:
        # Skip publish if any tool already sent a user-facing message to
        # this chat during the turn. Reading through a @DBOS.step keeps
        # the answer replay-stable — tool flags are in-memory and wouldn't
        # survive a container restart.
        already_sent = await _check_sent_in_turn_step(chat_id)
        if not already_sent:
            reply = final_content
            if reply is None:
                reply = "I've completed processing but have no response to give."
            await _publish_outbound_step(
                session_id=session_id,
                channel=channel,
                chat_id=chat_id,
                content=reply,
            )

    return final_content, new_msgs


# ── Durable inbound enqueue ─────────────────────────────────────────────────
#
# Closes the crash window between "channel received the message" and
# "agent started processing it". Before this, ``AgentLoop.run`` drained
# an in-memory ``asyncio.Queue`` and a container OOM in that window
# dropped the message — the channel's "eyes" ack had already been sent
# but no DBOS workflow existed to replay. Now channels flow through
# ``DBOSExecutor.enqueue_inbound`` (wired by AgentLoop when the executor
# advertises ``handles_inbound_enqueue=True``), which puts the message
# on the queue below. DBOS persists the enqueue record to SQLite
# immediately, so the message survives any subsequent crash.
#
# ``concurrency=1`` replaces ``AgentLoop._processing_lock``: DBOS runs
# at most one inbound turn at a time, matching the previous serial
# behavior. If per-session parallelism is later wanted, revisit.

INBOUND_QUEUE_NAME = "exoclaw-inbound"

# Declared at module import so DBOS sees the queue before
# ``DBOS.launch()`` starts its queue manager thread. A lazy
# construction path (first ``enqueue_inbound`` call) ran into
# ``Listening to 0 queues`` at launch and then failed submissions
# against the queue manager's thread pool — the manager only
# picks up queues that exist at launch time.
#
# The module is imported once per process, so this single
# construction is safe. ``DBOS.destroy()`` + a new ``DBOS()`` in
# the same process (test fixtures only; production constructs
# DBOS once) keeps the same Queue instance, which continues to
# work because the new ``DBOS.launch()`` discovers the existing
# global queue registry.
_INBOUND_QUEUE: Queue = Queue(INBOUND_QUEUE_NAME, concurrency=1)


def _get_inbound_queue() -> Queue:
    """Accessor kept as a function so tests can monkeypatch it if
    they want a fake queue, and to localize the import site."""
    return _INBOUND_QUEUE


@DBOS.workflow()
async def run_inbound_turn(
    channel: str,
    sender_id: str,
    chat_id: str,
    content: str,
    media: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    session_key_override: str | None = None,
    model_override: str | None = None,
) -> None:
    """Durable top-of-funnel for an inbound channel message.

    Started via ``queue.enqueue_async`` from
    ``DBOSExecutor.enqueue_inbound`` the moment a channel hands the
    message over. The workflow args are persisted to the DBOS system
    DB synchronously so a crash after enqueue and before the body
    runs still replays with the full message.

    Reconstructs the ``InboundMessage`` and drives it through
    ``AgentLoop._dispatch``. ``_dispatch`` calls ``_process_message``
    which in turn calls the executor's ``run_turn`` — which is
    ``run_durable_turn`` — as a child workflow, so the LLM-loop side
    of durability is unaffected.

    Args are flat primitives (matching the subagent workflow shape)
    rather than a packed dict so DBOS's argument codec sees only
    types it serializes without surprises.
    """
    loop = _loop
    if loop is None:
        raise RuntimeError("AgentLoop not set — call set_loop_context() at startup")

    msg = InboundMessage(
        channel=channel,
        sender_id=sender_id,
        chat_id=chat_id,
        content=content,
        media=list(media or []),
        metadata=dict(metadata or {}),
        session_key_override=session_key_override,
        model_override=model_override,
    )
    await loop._dispatch(msg)
