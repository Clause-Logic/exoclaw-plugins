"""Batch lifecycle state — pluggable so durable backends own durability.

``SubagentManager`` tracks how many subagents in a batch have completed and
fires one announcement when all are done. That bookkeeping used to live in
a ``dict[str, _BatchState]`` on the manager — in-memory, per-process.

That worked for ``AsyncioSpawner`` (everything in one process) but not for
durable backends. A DBOS-backed spawner survives process restarts by
replaying pending workflows, but an in-memory ``_batches`` dict does not.
Any subagent that completes on a recovered process finds the batch state
missing and its completion vanishes silently — no ``batch_progress``, no
final announcement.

``BatchStore`` is the fix: a small protocol the manager delegates to.
``InMemoryBatchStore`` preserves the pre-refactor behaviour for CLI use
and tests. Durable backends (DBOS today, Temporal on the roadmap) ship
their own ``BatchStore`` implementations — this package depends on
neither.

Idempotency contract
--------------------
Durable executors replay completed operations on recovery. Every method
on ``BatchStore`` MUST be safe to invoke twice with the same arguments:

* ``register(batch, task)`` — second call is a no-op; never inflates
  ``total``.
* ``record_completion_and_maybe_announce(batch, task, …)`` — second
  call with the same ``task_id`` does not re-increment the completed
  count. ``announce`` is invoked **at-least-once** per batch — the
  implementation may re-run it under replay (matching the final-reply
  posture in PR #44: accept one duplicate message rather than risk a
  silent drop).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, runtime_checkable


@dataclass
class BatchSnapshot:
    """State of a batch at the moment a member completion was recorded.

    ``results`` carries what ``_announce_batch`` needs to build its
    message — label, status, and result file path per completed member.
    Routing fields (``origin_channel`` / ``origin_chat_id`` /
    ``session_key``) come from ``register`` — the batch inherits them
    from the first task to join.
    """

    batch_id: str
    total: int
    completed: int
    results: list[dict[str, str]] = field(default_factory=list)
    origin_channel: str = "cli"
    origin_chat_id: str = "direct"
    session_key: str | None = None


AnnounceCallback = Callable[[BatchSnapshot], Awaitable[None]]
"""Publishes the batch announcement. The store invokes it at-least-once
when a completion pushes ``completed`` to ``total``."""


def _copy_snapshot(snap: BatchSnapshot) -> BatchSnapshot:
    return BatchSnapshot(
        batch_id=snap.batch_id,
        total=snap.total,
        completed=snap.completed,
        results=list(snap.results),
        origin_channel=snap.origin_channel,
        origin_chat_id=snap.origin_chat_id,
        session_key=snap.session_key,
    )


@runtime_checkable
class BatchStore(Protocol):
    """Durable-ready batch lifecycle state.

    The manager treats the store as the source of truth for batch state.
    Implementations keep the data wherever durability demands — a local
    dict (``InMemoryBatchStore``), a DBOS step (``DBOSBatchStore``), or a
    Temporal activity.
    """

    async def register(
        self,
        batch_id: str,
        task_id: str,
        *,
        session_key: str | None,
        origin_channel: str,
        origin_chat_id: str,
    ) -> None:
        """Add ``task_id`` to ``batch_id``'s roster. Idempotent."""

    async def record_completion_and_maybe_announce(
        self,
        batch_id: str,
        task_id: str,
        *,
        status: str,
        label: str,
        result_path: str | None,
        announce: AnnounceCallback,
    ) -> BatchSnapshot:
        """Mark ``task_id`` complete within ``batch_id`` and return the
        post-completion snapshot.

        Idempotent on ``(batch_id, task_id)``. Invokes ``announce`` when
        this completion pushes the batch to ``completed == total`` — and
        only then. Under replay the callback may fire again; ``announce``
        should therefore tolerate at-least-once delivery (the batch
        message may be published more than once).

        Raises ``KeyError`` if ``batch_id`` was never registered — the
        caller (SubagentManager) turns that into a ``subagent_done_orphaned``
        warning. Durable backends should use this signal to flag the
        specific prod failure mode that motivated this protocol:
        in-memory state lost across process restart.
        """

    def list_active(self) -> list[BatchSnapshot]:
        """Return snapshots of batches that have not yet been announced.

        Best-effort — durable backends that can't cheaply enumerate
        active batches may return ``[]``. Consumed only by the
        ``SpawnTool`` status action for human debugging; never on a
        hot path.
        """


class InMemoryBatchStore:
    """Default implementation. Preserves pre-refactor semantics.

    Not durable across process restarts — any backend that runs subagents
    as durable workflows should ship its own store so recovered
    completions still reach the parent.
    """

    def __init__(self) -> None:
        self._batches: dict[str, BatchSnapshot] = {}
        self._registered: dict[str, set[str]] = {}
        self._completed: dict[str, set[str]] = {}
        self._announced: set[str] = set()
        self._lock = asyncio.Lock()

    async def register(
        self,
        batch_id: str,
        task_id: str,
        *,
        session_key: str | None,
        origin_channel: str,
        origin_chat_id: str,
    ) -> None:
        async with self._lock:
            registered = self._registered.setdefault(batch_id, set())
            if task_id in registered:
                return
            registered.add(task_id)
            snap = self._batches.get(batch_id)
            if snap is None:
                snap = BatchSnapshot(
                    batch_id=batch_id,
                    total=0,
                    completed=0,
                    origin_channel=origin_channel,
                    origin_chat_id=origin_chat_id,
                    session_key=session_key,
                )
                self._batches[batch_id] = snap
            snap.total += 1

    async def record_completion_and_maybe_announce(
        self,
        batch_id: str,
        task_id: str,
        *,
        status: str,
        label: str,
        result_path: str | None,
        announce: AnnounceCallback,
    ) -> BatchSnapshot:
        async with self._lock:
            snap = self._batches.get(batch_id)
            if snap is None:
                # Never registered. Caller logs ``subagent_done_orphaned``.
                raise KeyError(batch_id)
            completed_set = self._completed.setdefault(batch_id, set())
            if task_id not in completed_set:
                completed_set.add(task_id)
                snap.completed += 1
                snap.results.append(
                    {
                        "label": label,
                        "status": status,
                        "path": result_path or "(no file)",
                    }
                )
            # Copy under the lock so concurrent completions see coherent
            # snapshots.
            post_snapshot = _copy_snapshot(snap)
            should_announce = (
                snap.completed >= snap.total and batch_id not in self._announced
            )

        if not should_announce:
            return post_snapshot

        # Publish BEFORE marking announced — matches PR #44's at-least-
        # once posture (duplicate on crash between publish and mark, but
        # never a silent drop). For the in-memory store there is no
        # crash window, but keeping the order consistent with durable
        # backends makes the contract uniform.
        await announce(post_snapshot)
        async with self._lock:
            self._announced.add(batch_id)
        return post_snapshot

    def list_active(self) -> list[BatchSnapshot]:
        return [
            _copy_snapshot(snap)
            for bid, snap in self._batches.items()
            if bid not in self._announced
        ]


__all__ = [
    "BatchSnapshot",
    "BatchStore",
    "InMemoryBatchStore",
    "AnnounceCallback",
]
