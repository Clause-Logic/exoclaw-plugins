"""Deferred workflow-start intents.

DBOS forbids ``DBOS.start_workflow_async`` from inside a ``@DBOS.step()``
(``_context.py`` asserts ``cur_ctx.is_workflow()`` before letting one
through). That means tools running inside ``_tool_step`` cannot spin up
DBOS child workflows directly — yet that is exactly what the spawn tool
needs to do.

The workaround is two-phase: tools queue an intent into a contextvar
during step execution, and ``DBOSExecutor.execute_tool`` drains the
queue *after* the step body returns. At that point the executor is back
in workflow context, so ``start_workflow_async`` is legal.

This file is a private contract between ``DBOSExecutor`` and the
DBOS-flavoured spawner. Tools never import it directly — they go
through ``DBOSSubagentSpawner`` (or future analogues), which knows
about this intent buffer.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass


@dataclass(frozen=True)
class StartChildWorkflow:
    """A request to start a DBOS child workflow once the active step exits.

    ``workflow_key`` is a registry key resolved by ``DBOSExecutor`` to
    the actual workflow function — keeps the intent free of DBOS
    function references so it can be added/queued/inspected from
    contexts that don't import DBOS.

    ``workflow_id`` is the DBOS workflow ID to use when starting the
    child. Must be deterministic across step retries so DBOS can dedup
    duplicate dispatches as no-ops.
    """

    workflow_key: str
    kwargs: dict[str, object]
    workflow_id: str


# Bound to a fresh list by ``DBOSExecutor.execute_tool`` before each step
# body runs. ``None`` means "not running under DBOSExecutor" — callers
# must fall back to a non-DBOS path.
_pending_intents: contextvars.ContextVar[list[StartChildWorkflow] | None] = contextvars.ContextVar(
    "_dbos_pending_intents", default=None
)


def try_queue_child_workflow(intent: StartChildWorkflow) -> bool:
    """Append a child-workflow start request to the active step's intent buffer.

    Returns ``True`` if an intent buffer was bound (we're inside a step
    wrapped by ``DBOSExecutor.execute_tool``) and the intent has been
    queued for deferred dispatch. Returns ``False`` if no buffer is
    bound — caller is responsible for dispatching directly, which is
    legal from workflow context but not from step context.
    """
    pending = _pending_intents.get()
    if pending is None:
        return False
    pending.append(intent)
    return True


def _bind_intent_buffer() -> tuple[list[StartChildWorkflow], contextvars.Token]:
    """Bind a fresh empty intent buffer to the current context.

    Returns ``(buffer, token)`` — the executor passes the token to
    ``_release_intent_buffer`` after the step body exits.
    """
    buffer: list[StartChildWorkflow] = []
    token = _pending_intents.set(buffer)
    return buffer, token


def _release_intent_buffer(token: contextvars.Token) -> None:
    _pending_intents.reset(token)
