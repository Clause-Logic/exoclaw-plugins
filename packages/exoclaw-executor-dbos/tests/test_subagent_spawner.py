"""Regression tests: DBOSSubagentSpawner gives each subagent its own wfid.

These cover the 2026-04-13 Feed curator incident:

    DBOSUnexpectedStepError: step 7, _chat_step recorded,
                             run_durable_turn expected

Root cause was ``SubagentManager.spawn`` dispatching background work via
``asyncio.create_task``, which inherits the parent's DBOS ContextVar. Every
step the subagent executed (LLM calls, tool calls, nested turns) was
recorded into the **parent workflow's** step journal. With 4 concurrent
subagents in the same parent session, the steps interleaved
non-deterministically and poisoned the parent's replay log.

The fix — now encoded as the ``SubagentSpawner`` protocol in
``exoclaw-subagent`` and implemented by ``DBOSSubagentSpawner`` here — is
that each subagent runs as its own DBOS child workflow via
``DBOS.start_workflow_async``, so each gets its own wfid + journal and
only a single deterministic "started child" entry is recorded in the
parent.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dbos import DBOS, DBOSConfig, SetWorkflowID

_DB_PATH = f"/tmp/dbos_subagent_spawner_test_{os.getpid()}.sqlite"


@pytest.fixture(scope="session", autouse=True)
def dbos_instance() -> Any:
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "dbos-subagent-spawner-test",
        "system_database_url": f"sqlite:///{_DB_PATH}",
        "enable_otlp": False,
    }
    # Importing the module registers @DBOS.workflow() decorators.
    import exoclaw_executor_dbos.subagent  # noqa: F401

    dbos = DBOS(config=config)
    DBOS.launch()
    yield dbos
    DBOS.destroy()
    if os.path.exists(_DB_PATH):
        os.unlink(_DB_PATH)


def _make_manager() -> Any:
    from exoclaw_executor_dbos import DBOSSubagentSpawner
    from exoclaw_subagent import SubagentManager

    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    return SubagentManager(
        provider=MagicMock(),
        bus=bus,
        conversation_factory=MagicMock,
        max_iterations=2,
        spawner_factory=DBOSSubagentSpawner,
    )


@pytest.mark.asyncio(loop_scope="session")
class TestDBOSSubagentSpawner:
    async def test_spawned_subagent_has_independent_workflow_id(self, dbos_instance: Any) -> None:
        """The subagent's work must run under a DBOS workflow id distinct from its parent."""
        observed: dict[str, str | None] = {}
        child_ran = asyncio.Event()

        async def observe_process_direct(task: str, **kwargs: Any) -> str:
            observed["child"] = DBOS.workflow_id
            child_ran.set()
            return "done"

        mock_loop = MagicMock()
        mock_loop.process_direct = observe_process_direct

        mgr = _make_manager()
        parent_wfid = f"parent-{uuid.uuid4()}"

        @DBOS.workflow()
        async def parent_workflow() -> None:
            observed["parent"] = DBOS.workflow_id
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr.spawn(task="work", label="child")
                await asyncio.wait_for(child_ran.wait(), timeout=2.0)

        with SetWorkflowID(parent_wfid):
            await parent_workflow()

        assert observed.get("parent") == parent_wfid
        assert observed.get("child") is not None, (
            "subagent executed outside any DBOS workflow — "
            "DBOSSubagentSpawner should have started a child workflow"
        )
        assert observed["child"] != observed["parent"], (
            f"subagent inherited the parent workflow id {observed['parent']!r} — "
            "concurrent subagents will race into the parent's step journal and "
            "break determinism (see 2026-04-13 Feed curator failure)."
        )

    async def test_concurrent_spawns_get_distinct_workflow_ids(self, dbos_instance: Any) -> None:
        """Four subagents spawned from the same parent must each get their own wfid.

        This mirrors the 2026-04-13 incident shape: the user spawned 4 parallel
        subagents (web-search, calendar, feed-fetch, notes-search) in one Zulip
        topic and they interleaved writes into the same step journal.
        """
        n = 4
        child_wfids: list[str | None] = []
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def observe_process_direct(task: str, **kwargs: Any) -> str:
            child_wfids.append(DBOS.workflow_id)
            if len(child_wfids) == n:
                all_started.set()
            await release.wait()
            return "done"

        mock_loop = MagicMock()
        mock_loop.process_direct = observe_process_direct

        mgr = _make_manager()

        @DBOS.workflow()
        async def parent_workflow() -> None:
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                for i in range(n):
                    await mgr.spawn(task=f"t{i}", label=f"child-{i}")
                try:
                    await asyncio.wait_for(all_started.wait(), timeout=2.0)
                finally:
                    release.set()

        with SetWorkflowID(f"parent-{uuid.uuid4()}"):
            await parent_workflow()

        assert len(child_wfids) == n
        assert all(w is not None for w in child_wfids), (
            f"subagents ran outside any DBOS workflow: {child_wfids}"
        )
        assert len(set(child_wfids)) == n, (
            f"concurrent subagents shared workflow ids {child_wfids!r} — "
            "they will race into the same DBOS step journal"
        )

    async def test_parent_turn_chain_threaded_into_child_workflow(self, dbos_instance: Any) -> None:
        """``parent_turn_chain`` and ``parent_turn_id`` flow from
        ``SubagentManager.spawn`` → ``DBOSSubagentSpawner.start`` →
        ``_subagent_workflow`` workflow arguments → ``_run`` →
        rebound into structlog contextvars before the child agent
        loop starts.

        This is the stage-3 propagation contract end-to-end on the
        DBOS substrate. We observe the contextvars from inside the
        child's mocked ``process_direct`` to confirm the chain
        actually arrives at the agent loop boundary.
        """
        import structlog
        import structlog.contextvars

        observed: dict[str, object] = {}
        child_ran = asyncio.Event()

        async def observe_process_direct(task: str, **kwargs: Any) -> str:
            observed.update(structlog.contextvars.get_contextvars())
            child_ran.set()
            return "done"

        mock_loop = MagicMock()
        mock_loop.process_direct = observe_process_direct

        mgr = _make_manager()

        @DBOS.workflow()
        async def parent_workflow() -> None:
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr.spawn(
                    task="work",
                    label="child",
                    parent_turn_chain="rootA:parentB",
                    parent_turn_id="parentB",
                )
                await asyncio.wait_for(child_ran.wait(), timeout=2.0)

        with SetWorkflowID(f"parent-{uuid.uuid4()}"):
            await parent_workflow()

        assert observed.get("turn.chain") == "rootA:parentB", (
            "child agent loop did not see parent_turn_chain as turn.chain — "
            "DBOSSubagentSpawner failed to thread the workflow argument through"
        )
        assert observed.get("turn.id") == "parentB"
        assert observed.get("turn.root_id") == "rootA"

    async def test_parent_turn_chain_replayed_on_recovery(self, dbos_instance: Any) -> None:
        """The hard invariant: after a workflow crash the recovered
        child workflow re-enters with the *same* parent turn ancestry
        it had on the original run.

        Proves that ``parent_turn_chain`` is journaled as a durable
        workflow argument rather than derived from ambient state at
        execution time — the latter would leave recovered subagents
        bound to nothing, and their post-recovery log lines would
        drop out of ``turn.root_id`` queries.

        Mirrors the stage-2 ``test_mint_turn_id_replayed_on_recovery``
        template: run once, force status back to PENDING, call
        ``DBOS._recover_pending_workflows()``, assert the recovered
        workflow saw the same ancestry as the original.
        """
        import sqlalchemy as sa
        import structlog
        import structlog.contextvars
        from dbos._schemas.system_database import SystemSchema
        from exoclaw_executor_dbos.subagent import _subagent_workflow

        observed_chains: list[object] = []
        observed_ids: list[object] = []
        observed_roots: list[object] = []

        async def observe(task: str, **kwargs: Any) -> str:
            ctx = structlog.contextvars.get_contextvars()
            observed_chains.append(ctx.get("turn.chain"))
            observed_ids.append(ctx.get("turn.id"))
            observed_roots.append(ctx.get("turn.root_id"))
            return "done"

        mock_loop = MagicMock()
        mock_loop.process_direct = observe

        # _make_manager() has the side effect of binding the runner
        # adapter as the module-level _active_runner that
        # _subagent_workflow reads — construct it even though we
        # don't hold the reference.
        _make_manager()

        wfid = f"replay-test-{uuid.uuid4()}"
        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            with SetWorkflowID(wfid):
                await _subagent_workflow(
                    task_id="t1",
                    task="task",
                    label="label",
                    origin_channel="cli",
                    origin_chat_id="user1",
                    session_key="cli:user1",
                    batch=None,
                    skills=None,
                    model=None,
                    parent_turn_chain="rootA:parentB",
                    parent_turn_id="parentB",
                )

            assert observed_chains == ["rootA:parentB"]
            assert observed_ids == ["parentB"]
            assert observed_roots == ["rootA"]

            # Force workflow status back to PENDING to simulate a
            # crash mid-workflow. DBOS recovery re-enters the body
            # with the originally-journaled workflow arguments.
            with dbos_instance._sys_db.engine.begin() as conn:
                conn.execute(
                    sa.update(SystemSchema.workflow_status)
                    .values({"status": "PENDING"})
                    .where(SystemSchema.workflow_status.c.workflow_uuid == wfid)
                )

            handles = DBOS._recover_pending_workflows()
            recovered = [h for h in handles if h.workflow_id == wfid]
            assert len(recovered) == 1, (
                f"expected to recover the target workflow, got handles "
                f"{[h.workflow_id for h in handles]}"
            )
            # ``get_result`` is sync on DBOS workflow handles — same
            # pattern as the stage-2 ``test_mint_turn_id_replayed_on_recovery``
            # test. Await the event instead to know the replay body
            # has actually run and updated ``observed_*``.
            recovered[0].get_result()

        # The recovered run must have observed the exact same ancestry
        # as the original. If this fails, ``parent_turn_chain`` was
        # being sourced from something non-durable on replay.
        assert observed_chains[-1] == "rootA:parentB", (
            f"recovered workflow saw turn.chain={observed_chains[-1]!r}, "
            f"expected 'rootA:parentB' — workflow arguments were not "
            f"re-journaled across replay"
        )
        assert observed_ids[-1] == "parentB"
        assert observed_roots[-1] == "rootA"


@pytest.mark.asyncio(loop_scope="session")
class TestDBOSSubagentSpawnerConcurrencyCap:
    async def test_no_queue_registered_by_default(self, dbos_instance: Any) -> None:
        """Without ``max_concurrent``, no queue is attached — dispatch
        falls through to ``DBOS.start_workflow_async`` so the new code
        path is fully opt-in."""
        from exoclaw_executor_dbos.executor import _queue_registry
        from exoclaw_executor_dbos.subagent import SUBAGENT_WORKFLOW_KEY, DBOSSubagentSpawner

        async def noop_runner(**kwargs: Any) -> None:
            return None

        DBOSSubagentSpawner(noop_runner)  # max_concurrent defaults to None

        assert SUBAGENT_WORKFLOW_KEY not in _queue_registry

    async def test_max_concurrent_registers_queue(self, dbos_instance: Any) -> None:
        """With ``max_concurrent=N``, the spawner owns a ``DBOS.Queue``
        and registers it under the subagent workflow key so intent
        dispatch routes through ``queue.enqueue_async``."""
        from dbos import Queue
        from exoclaw_executor_dbos.executor import _queue_registry
        from exoclaw_executor_dbos.subagent import SUBAGENT_WORKFLOW_KEY, DBOSSubagentSpawner

        async def noop_runner(**kwargs: Any) -> None:
            return None

        spawner = DBOSSubagentSpawner(noop_runner, max_concurrent=2)
        try:
            registered = _queue_registry.get(SUBAGENT_WORKFLOW_KEY)
            assert registered is not None
            assert isinstance(registered, Queue)
            assert registered is spawner._queue
        finally:
            # Cleanup so other tests don't see a stale queue registration.
            DBOSSubagentSpawner(noop_runner, max_concurrent=None)

    async def test_reconstruction_without_cap_clears_stale_registration(
        self, dbos_instance: Any
    ) -> None:
        """If a capped spawner is replaced by an uncapped one in the same
        process, the stale queue registration is removed. Important for
        tests and config reloads that re-init the manager."""
        from exoclaw_executor_dbos.executor import _queue_registry
        from exoclaw_executor_dbos.subagent import SUBAGENT_WORKFLOW_KEY, DBOSSubagentSpawner

        async def noop_runner(**kwargs: Any) -> None:
            return None

        DBOSSubagentSpawner(noop_runner, max_concurrent=4)
        assert SUBAGENT_WORKFLOW_KEY in _queue_registry

        DBOSSubagentSpawner(noop_runner, max_concurrent=None)
        assert SUBAGENT_WORKFLOW_KEY not in _queue_registry
