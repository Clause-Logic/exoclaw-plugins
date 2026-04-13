"""Regression test: subagents must run as their own DBOS child workflows.

Background: on 2026-04-13 a Feed curator cron run failed with
``DBOSUnexpectedStepError: step 7, _chat_step recorded, run_durable_turn
expected``. Root cause was ``SubagentManager.spawn`` dispatching work via
``asyncio.create_task``, which inherits the parent's DBOS ContextVar. Every
step the subagent executed (LLM calls, tool calls, nested turns) was
recorded into the **parent workflow's** journal. With 4 concurrent subagents
in the same parent session, the steps interleaved non-deterministically and
poisoned the parent's replay log.

The fix is to dispatch subagents via ``DBOS.start_workflow_async`` so each
gets its own wfid + journal and only a single deterministic "started child"
entry is recorded in the parent.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dbos import DBOS, DBOSConfig, SetWorkflowID

_DB_PATH = f"/tmp/dbos_subagent_test_{os.getpid()}.sqlite"


@pytest.fixture(scope="session", autouse=True)
def dbos_instance() -> Any:
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "subagent-child-workflow-test",
        "system_database_url": f"sqlite:///{_DB_PATH}",
        "enable_otlp": False,
    }
    dbos = DBOS(config=config)
    DBOS.launch()
    yield dbos
    DBOS.destroy()
    if os.path.exists(_DB_PATH):
        os.unlink(_DB_PATH)


@pytest.mark.asyncio(loop_scope="session")
class TestSubagentRunsAsChildWorkflow:
    async def test_spawned_subagent_has_independent_workflow_id(self, dbos_instance: Any) -> None:
        """The subagent's work must observe a DBOS workflow id distinct from its parent.

        Today this fails: ``asyncio.create_task`` copies the parent's context
        so ``DBOS.workflow_id`` inside the subagent equals the parent's.
        After switching to ``DBOS.start_workflow_async`` each subagent runs
        under its own wfid.
        """
        from exoclaw_subagent.manager import SubagentManager

        observed: dict[str, str | None] = {}

        async def observe_process_direct(task: str, **kwargs: Any) -> str:
            observed["child"] = DBOS.workflow_id
            return "done"

        mock_loop = MagicMock()
        mock_loop.process_direct = observe_process_direct

        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        mgr = SubagentManager(
            provider=MagicMock(),
            bus=bus,
            conversation_factory=MagicMock,
            max_iterations=2,
        )

        parent_wfid = f"parent-{uuid.uuid4()}"

        @DBOS.workflow()
        async def parent_workflow() -> None:
            observed["parent"] = DBOS.workflow_id
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr.spawn(task="work", label="child")
                # allow the dispatched child to run
                for _ in range(20):
                    if "child" in observed:
                        break
                    await asyncio.sleep(0.01)

        with SetWorkflowID(parent_wfid):
            await parent_workflow()

        assert observed.get("parent") == parent_wfid, (
            f"parent workflow id not captured (got {observed.get('parent')!r})"
        )
        assert observed.get("child") is not None, (
            "subagent executed outside any DBOS workflow — expected it to run "
            "inside its own child workflow started via DBOS.start_workflow_async"
        )
        assert observed["child"] != observed["parent"], (
            f"subagent inherited the parent workflow id {observed['parent']!r} — "
            "concurrent subagents will race into the parent's step journal and "
            "break determinism (see 2026-04-13 Feed curator failure). "
            "Dispatch via DBOS.start_workflow_async so each subagent gets its "
            "own wfid."
        )

    async def test_concurrent_spawns_get_distinct_workflow_ids(self, dbos_instance: Any) -> None:
        """Two subagents spawned from the same parent must each get their own wfid.

        This is the exact shape of the original failure — the user spawned 4
        parallel subagents (web-search, calendar, feed-fetch, notes-search)
        in one Zulip topic and they interleaved writes into the same journal.
        """
        from exoclaw_subagent.manager import SubagentManager

        child_wfids: list[str | None] = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def observe_process_direct(task: str, **kwargs: Any) -> str:
            child_wfids.append(DBOS.workflow_id)
            if len(child_wfids) == 2:
                started.set()
            await release.wait()
            return "done"

        mock_loop = MagicMock()
        mock_loop.process_direct = observe_process_direct

        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        mgr = SubagentManager(
            provider=MagicMock(),
            bus=bus,
            conversation_factory=MagicMock,
            max_iterations=2,
        )

        @DBOS.workflow()
        async def parent_workflow() -> None:
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr.spawn(task="a", label="a")
                await mgr.spawn(task="b", label="b")
                try:
                    await asyncio.wait_for(started.wait(), timeout=2.0)
                finally:
                    release.set()
                # drain the running tasks
                for _ in range(50):
                    if mgr.get_running_count() == 0:
                        break
                    await asyncio.sleep(0.01)

        with SetWorkflowID(f"parent-{uuid.uuid4()}"):
            await parent_workflow()

        assert len(child_wfids) == 2
        assert all(w is not None for w in child_wfids), (
            f"subagents ran outside any DBOS workflow: {child_wfids}"
        )
        assert child_wfids[0] != child_wfids[1], (
            f"concurrent subagents shared a workflow id {child_wfids[0]!r} — "
            "they will race into the same DBOS step journal"
        )
