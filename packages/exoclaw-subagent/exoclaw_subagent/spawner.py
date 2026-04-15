"""Pluggable subagent dispatch.

``SubagentManager`` delegates background execution of each subagent to a
``SubagentSpawner``. The default ``AsyncioSpawner`` runs work as a bare
``asyncio`` task. Durable-workflow backends (DBOS, Temporal, …) live in
their own plugin packages and implement the same protocol, so the subagent
package itself depends on neither.

Durability-friendly contract:

- All kwargs passed to ``SubagentSpawner.start`` are JSON-serializable
  primitives, so a durable backend can journal them and replay on recovery.
- The substrate-specific workflow/activity functions must be pre-registered
  with their backend (decorators run at import time); the spawner owns
  that registration.
- The manager-side runner (typically ``SubagentManager._run``) is passed to
  the spawner factory at construction, not on every call, so durable
  backends can stash a module-level reference and reuse it across replays.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable

Runner = Callable[..., Coroutine[Any, Any, None]]
"""Async callable that executes one subagent given its keyword args.

Shape matches ``SubagentManager._run``'s parameters — the manager passes a
small adapter here rather than a direct method reference so attribute-level
patches (e.g. in tests) are still picked up per-call.
"""


@runtime_checkable
class SubagentHandle(Protocol):
    """Opaque handle to a running subagent."""

    @property
    def id(self) -> str: ...

    def done(self) -> bool: ...

    async def wait(self) -> None:
        """Block until the subagent finishes. Must not raise on failure."""
        ...

    async def cancel(self) -> None:
        """Best-effort cancellation. Idempotent."""
        ...


@runtime_checkable
class SubagentSpawner(Protocol):
    """Dispatches subagents onto a background execution substrate.

    Implementations wrap whatever durability layer is in use (bare
    asyncio, DBOS, Temporal, …). The subagent package never imports any
    of them — the wiring layer injects a concrete spawner factory.

    ``parent_turn_chain`` and ``parent_turn_id`` carry the parent
    turn's trace ancestry so the child workflow inherits it across the
    spawn boundary. Durable backends (DBOS, Temporal) must journal
    these as workflow arguments so the ancestry survives recovery.
    """

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
    ) -> SubagentHandle: ...


SpawnerFactory = Callable[[Runner], SubagentSpawner]
"""Builds a ``SubagentSpawner`` given the manager's runner adapter.

Factory form lets the manager construct the spawner during its own
``__init__`` — cleaner than a two-phase ``bind()`` and avoids the
chicken-and-egg between spawner and manager.
"""


class _AsyncioHandle:
    """``SubagentHandle`` backed by an ``asyncio.Task``."""

    def __init__(self, task_id: str, task: asyncio.Task[None]) -> None:
        self._id = task_id
        self._task = task

    @property
    def id(self) -> str:
        return self._id

    def done(self) -> bool:
        return self._task.done()

    async def wait(self) -> None:
        try:
            await self._task
        except (Exception, asyncio.CancelledError):
            pass

    async def cancel(self) -> None:
        if not self._task.done():
            self._task.cancel()


class AsyncioSpawner:
    """Non-durable default: each subagent runs as a bare ``asyncio`` task.

    Appropriate for CLI/one-shot use and for host apps that don't need
    crash-recovery of in-flight subagents. Durable hosts should inject a
    backend-specific spawner factory via ``SubagentManager(spawner_factory=…)``.
    """

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

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
        bg_task: asyncio.Task[None] = asyncio.create_task(
            self._runner(
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
        )
        return _AsyncioHandle(task_id, bg_task)


__all__ = [
    "AsyncioSpawner",
    "Runner",
    "SpawnerFactory",
    "SubagentHandle",
    "SubagentSpawner",
]
