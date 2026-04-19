"""DBOS-backed spawner for ``exoclaw-subagent``.

``DBOSSubagentSpawner`` implements the ``SubagentSpawner`` protocol by
dispatching each subagent as its own DBOS child workflow. This gives
every subagent its own wfid + step journal so concurrent subagents
can't race into the parent workflow's log and poison determinism.

Wiring:

    from exoclaw_executor_dbos import DBOSSubagentSpawner
    from exoclaw_subagent import SubagentManager

    mgr = SubagentManager(
        provider=...,
        bus=...,
        conversation_factory=...,
        spawner_factory=DBOSSubagentSpawner,
    )

Single-spawner-per-process: the workflow entrypoint must be module-level
so DBOS can register it at import time and look it up by name on
recovery. The spawner stores a reference to the manager's runner
adapter in a module-level global that the workflow reads at invocation.
Constructing more than one ``DBOSSubagentSpawner`` in the same process
is unsupported — the last one wins.
"""

from __future__ import annotations

from dbos import DBOS, Queue, SetWorkflowID
from exoclaw_subagent import Runner, SubagentHandle

from .executor import register_intent_queue, register_intent_workflow, unregister_intent_queue
from .intents import StartChildWorkflow, try_queue_child_workflow

# ── Module-level runner ref (set at DBOSSubagentSpawner construction) ───────
# The decorated workflow function reads this at invocation time. Replay does
# not serialize it — the host process always boots, constructs the spawner,
# and sets this before DBOS processes any pending work.

_active_runner: Runner | None = None


@DBOS.workflow()
async def _subagent_workflow(
    task_id: str,
    task: str,
    label: str,
    origin_channel: str,
    origin_chat_id: str,
    session_key: str | None,
    batch: str | None,
    skills: list[str] | None,
    model: str | None,
    parent_turn_chain: str | None = None,
    parent_turn_id: str | None = None,
) -> None:
    """Durable child workflow wrapping one subagent run.

    Invoked via ``DBOS.start_workflow_async``; each invocation gets its
    own wfid and step journal so concurrent subagents don't interleave
    writes into a shared log.

    ``parent_turn_chain`` and ``parent_turn_id`` are explicit workflow
    arguments — DBOS journals workflow arguments durably so on
    crash-recovery the child workflow re-enters with the *same* parent
    ancestry it had on the original run. That is the load-bearing
    invariant for stage-3 turn observability: trace lines emitted by
    a subagent before vs after a recovery boundary still land in the
    same ``turn.root_id`` query.

    The ``runner`` (``SubagentManager._run``) is responsible for
    re-binding these into structlog contextvars before invoking the
    child agent loop.
    """
    runner = _active_runner
    if runner is None:
        raise RuntimeError(
            "DBOSSubagentSpawner runner not bound — construct "
            "DBOSSubagentSpawner during app startup before any subagent "
            "spawns or DBOS workflow recovery runs."
        )
    await runner(
        task_id=task_id,
        task=task,
        label=label,
        origin_channel=origin_channel,
        origin_chat_id=origin_chat_id,
        session_key=session_key,
        batch=batch,
        skills=skills,
        model=model,
        parent_turn_chain=parent_turn_chain,
        parent_turn_id=parent_turn_id,
    )


_TERMINAL_STATUSES = frozenset({"SUCCESS", "ERROR", "MAX_RECOVERY_ATTEMPTS_EXCEEDED", "CANCELLED"})


class _DeferredHandle:
    """``SubagentHandle`` for a child workflow whose dispatch is deferred.

    ``DBOSSubagentSpawner`` cannot start the workflow inline because it
    runs from inside ``_tool_step``, where DBOS forbids
    ``start_workflow_async`` (the assertion at ``_context.py:183``).
    Instead it queues a ``StartChildWorkflow`` intent and returns this
    handle. ``DBOSExecutor.execute_tool`` dispatches the queued intent
    after the wrapping step exits.

    Status methods query DBOS by ``workflow_id``. Between ``start()``
    returning and the executor dispatching the intent, the workflow does
    not yet exist in the system DB — both ``done()`` and ``wait()`` treat
    that window as "not yet done", which matches what
    ``SubagentManager.get_running_count`` and
    ``SubagentManager.cancel_by_session`` need to behave correctly.
    """

    def __init__(self, workflow_id: str) -> None:
        self._id = workflow_id

    @property
    def id(self) -> str:
        return self._id

    def done(self) -> bool:
        try:
            status = DBOS.get_workflow_status(self._id)
        except Exception:
            return False
        if status is None:
            return False  # not yet dispatched, or already cleaned up
        return status.status in _TERMINAL_STATUSES

    async def wait(self) -> None:
        try:
            handle = DBOS.retrieve_workflow(self._id)
            await handle.get_result()
        except Exception:
            pass

    async def cancel(self) -> None:
        try:
            await DBOS.cancel_workflow_async(self._id)
        except Exception:
            pass


def _intent_workflow_id(task_id: str) -> str:
    """Deterministic workflow ID so DBOS dedups duplicate dispatches.

    ``task_id`` is set by ``SubagentManager.spawn`` per spawn call. On
    step retry, the spawner re-queues with the same ``task_id`` and the
    executor re-dispatches with this same ID — DBOS treats the second
    ``start_workflow_async`` as a no-op.
    """
    return f"subagent:{task_id}"


SUBAGENT_WORKFLOW_KEY = "exoclaw_subagent"
SUBAGENT_QUEUE_NAME = "exoclaw-subagent"
register_intent_workflow(SUBAGENT_WORKFLOW_KEY, _subagent_workflow)

# DBOS ``Queue`` declares itself into a process-global registry on
# construction and raises if declared twice under the same name. The
# spawner may be reconstructed (config reloads, tests) so we cache the
# queue object here and reuse it across reconstructions. Concurrency is
# fixed at first-declare; changing the cap requires a process restart —
# we remember the first cap and raise on any subsequent mismatch so
# misconfigurations fail loudly instead of silently ignoring the new value.
_cached_queue: Queue | None = None
_cached_queue_concurrency: int | None = None


class DBOSSubagentSpawner:
    """Queues subagent dispatches as deferred DBOS child workflows.

    Matches the ``SpawnerFactory`` signature: ``SubagentManager`` calls
    ``DBOSSubagentSpawner(runner)`` during its own ``__init__``. The
    runner is an async adapter around ``SubagentManager._run`` that
    handles per-task cleanup; storing it in ``_active_runner`` is safe
    because it's a live in-process reference, not serialized into the
    workflow journal.

    ``start()`` does *not* call ``DBOS.start_workflow_async`` — it runs
    inside ``_tool_step`` where that's illegal. It queues a
    ``StartChildWorkflow`` intent on a contextvar that
    ``DBOSExecutor.execute_tool`` drains once the wrapping step exits.

    ``max_concurrent`` caps how many subagent workflows DBOS runs at
    once. When set, the spawner creates a ``DBOS.Queue`` and attaches it
    to the subagent workflow key so dispatch goes through
    ``queue.enqueue_async`` instead of ``start_workflow_async``.
    Subagents beyond the cap stay ``ENQUEUED`` in the DBOS system DB and
    drain as slots open — behavior survives crash recovery natively.
    ``None`` preserves the uncapped behavior.
    """

    def __init__(self, runner: Runner, max_concurrent: int | None = None) -> None:
        global _active_runner, _cached_queue, _cached_queue_concurrency
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1 or None, got {max_concurrent!r}")
        _active_runner = runner
        self._queue: Queue | None = None
        if max_concurrent is not None:
            if _cached_queue is None:
                _cached_queue = Queue(SUBAGENT_QUEUE_NAME, concurrency=max_concurrent)
                _cached_queue_concurrency = max_concurrent
            elif _cached_queue_concurrency != max_concurrent:
                raise RuntimeError(
                    f"DBOSSubagentSpawner already declared queue "
                    f"{SUBAGENT_QUEUE_NAME!r} with concurrency="
                    f"{_cached_queue_concurrency}; cannot rebind to "
                    f"concurrency={max_concurrent} — DBOS queue caps are "
                    f"fixed at first declaration. Restart the process to "
                    f"change the cap."
                )
            self._queue = _cached_queue
            register_intent_queue(SUBAGENT_WORKFLOW_KEY, self._queue)
        else:
            # Clear any stale registration from a previous spawner (tests,
            # re-init). Harmless if nothing is registered.
            unregister_intent_queue(SUBAGENT_WORKFLOW_KEY)

    async def start(
        self,
        *,
        task_id: str,
        task: str,
        label: str,
        origin_channel: str,
        origin_chat_id: str,
        session_key: str | None,
        batch: str | None,
        skills: list[str] | None,
        model: str | None,
        parent_turn_chain: str | None = None,
        parent_turn_id: str | None = None,
    ) -> SubagentHandle:
        wfid = _intent_workflow_id(task_id)
        kwargs: dict[str, object] = {
            "task_id": task_id,
            "task": task,
            "label": label,
            "origin_channel": origin_channel,
            "origin_chat_id": origin_chat_id,
            "session_key": session_key,
            "batch": batch,
            "skills": skills,
            "model": model,
            "parent_turn_chain": parent_turn_chain,
            "parent_turn_id": parent_turn_id,
        }

        # If we're inside a `_tool_step` wrapped by `DBOSExecutor`, the
        # executor has bound an intent buffer; queue the dispatch and let
        # the executor start the child workflow after the step exits.
        # Otherwise we're being called from workflow context directly
        # (e.g. from a top-level @DBOS.workflow), where DBOS allows
        # `start_workflow_async` — fall through to inline dispatch.
        intent = StartChildWorkflow(
            workflow_key=SUBAGENT_WORKFLOW_KEY,
            kwargs=kwargs,
            workflow_id=wfid,
        )
        if try_queue_child_workflow(intent):
            return _DeferredHandle(wfid)

        with SetWorkflowID(wfid):
            if self._queue is not None:
                await self._queue.enqueue_async(_subagent_workflow, **kwargs)
            else:
                await DBOS.start_workflow_async(_subagent_workflow, **kwargs)
        return _DeferredHandle(wfid)


__all__ = ["DBOSSubagentSpawner"]
