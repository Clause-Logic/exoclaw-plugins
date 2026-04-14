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

from typing import Any

from dbos import DBOS
from exoclaw_subagent import Runner, SubagentHandle

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


class _DBOSHandle:
    """``SubagentHandle`` backed by a DBOS ``WorkflowHandleAsync``."""

    def __init__(self, wf_handle: Any) -> None:
        self._wf = wf_handle
        self._id: str = wf_handle.get_workflow_id()
        self._done = False

    @property
    def id(self) -> str:
        return self._id

    def done(self) -> bool:
        return self._done

    async def wait(self) -> None:
        try:
            await self._wf.get_result()
        except Exception:
            pass
        finally:
            self._done = True

    async def cancel(self) -> None:
        try:
            await DBOS.cancel_workflow_async(self._id)
        except Exception:
            pass
        self._done = True


class DBOSSubagentSpawner:
    """Dispatches subagents as DBOS child workflows.

    Matches the ``SpawnerFactory`` signature: the ``SubagentManager`` calls
    ``DBOSSubagentSpawner(runner)`` during its own ``__init__``. The runner
    is an async adapter around ``SubagentManager._run`` that handles
    per-task cleanup; storing it in ``_active_runner`` is safe because it's
    a live in-process reference, not serialized into the workflow journal.
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
        wf_handle = await DBOS.start_workflow_async(
            _subagent_workflow,
            task_id,
            task,
            label,
            origin_channel,
            origin_chat_id,
            session_key,
            batch,
            skills,
            model,
        )
        return _DBOSHandle(wf_handle)


__all__ = ["DBOSSubagentSpawner"]
