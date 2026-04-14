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
