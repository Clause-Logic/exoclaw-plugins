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

from dbos import DBOS, SetWorkflowID
from exoclaw_subagent import Runner, SubagentHandle

from .executor import register_intent_workflow
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
) -> None:
    """Durable child workflow wrapping one subagent run.

    Invoked via ``DBOS.start_workflow_async``; each invocation gets its
    own wfid and step journal so concurrent subagents don't interleave
    writes into a shared log.
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
    )


class _DeferredHandle:
    """Stub ``SubagentHandle`` returned by ``DBOSSubagentSpawner.start``.

    ``DBOSSubagentSpawner`` cannot start the workflow inline because it
    runs from inside ``_tool_step``, where DBOS forbids
    ``start_workflow_async`` (the assertion at ``_context.py:183``).
    Instead it queues a ``StartChildWorkflow`` intent that
    ``DBOSExecutor`` dispatches after the step exits, and returns this
    stub so ``SubagentManager`` can carry on with its bookkeeping.

    The stub is reported as ``done`` immediately. The manager only ever
    awaits these handles via ``/stop``, and even then it just wants to
    join — best-effort cancellation goes through DBOS by workflow ID.
    """

    def __init__(self, workflow_id: str) -> None:
        self._id = workflow_id

    @property
    def id(self) -> str:
        return self._id

    def done(self) -> bool:
        return True

    async def wait(self) -> None:
        return

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
register_intent_workflow(SUBAGENT_WORKFLOW_KEY, _subagent_workflow)


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
    """

    def __init__(self, runner: Runner) -> None:
        global _active_runner
        _active_runner = runner

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
            await DBOS.start_workflow_async(_subagent_workflow, **kwargs)
        return _DeferredHandle(wfid)


__all__ = ["DBOSSubagentSpawner"]
