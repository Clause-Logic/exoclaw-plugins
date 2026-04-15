"""End-to-end smoke test for the spawn dispatch path.

This test wires the real composition that runs in production:

    DBOS workflow → DBOSExecutor.execute_tool → _tool_step (DBOS step)
                  → ToolRegistry.execute → SpawnTool → SubagentManager.spawn
                  → DBOSSubagentSpawner.start → child DBOS workflow

Three bugs in two weeks shipped to production because no test exercised
this whole stack together:

1. ``ToolRegistry`` swallowed every tool exception into the empty string
   ``"Error executing spawn:"`` (exoclaw < 0.14).
2. ``DBOSExecutor`` was missing the ``set_messages`` /
   ``append_messages`` / ``load_messages`` methods that the executor
   protocol added in exoclaw 0.13 (exoclaw-executor-dbos < 0.5.1).
3. ``DBOSSubagentSpawner.start`` called ``DBOS.start_workflow_async``
   from inside ``_tool_step``, which DBOS forbids — assertion at
   ``_context.py:183`` fired with no message
   (exoclaw-executor-dbos < 0.5.2).

Each of those bugs was caught at deploy by tracing production logs.
This test fails immediately if any equivalent regression lands again.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dbos import DBOS, DBOSConfig, SetWorkflowID
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_executor_dbos import DBOSExecutor, DBOSSubagentSpawner
from exoclaw_subagent import SpawnTool, SubagentManager

_DB_PATH = f"/tmp/dbos_spawn_integration_test_{os.getpid()}.sqlite"


@pytest.fixture(scope="module")
def dbos_instance() -> Any:
    """Module-scoped DBOS fixture.

    Not session-scoped — DBOS is a process-global singleton and other
    test modules in this package may have their own DBOS fixtures with
    their own SQLite paths. Module scope keeps the lifetime contained
    so we don't fight other modules over the singleton.
    """
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "spawn-integration-test",
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


def _build_stack() -> tuple[ToolRegistry, DBOSExecutor]:
    """Wire the real composition: registry + SpawnTool + manager + executor."""
    bus = MagicMock()
    bus.publish_inbound = AsyncMock()

    manager = SubagentManager(
        provider=MagicMock(),
        bus=bus,
        conversation_factory=MagicMock,
        max_iterations=2,
        spawner_factory=DBOSSubagentSpawner,
    )

    spawn_tool = SpawnTool(manager=manager)

    registry = ToolRegistry()
    registry.register(spawn_tool)

    executor = DBOSExecutor()
    return registry, executor


@pytest.mark.asyncio(loop_scope="session")
class TestSpawnIntegration:
    async def test_spawn_via_execute_tool_dispatches_child_workflow(
        self, dbos_instance: Any
    ) -> None:
        """Spawn called via ``DBOSExecutor.execute_tool`` (i.e. from a
        ``_tool_step``) must successfully dispatch a DBOS child workflow.

        Pre-fix this asserted with no message inside DBOS context plumbing
        because ``DBOSSubagentSpawner.start`` called
        ``start_workflow_async`` from step context.
        """
        observed: dict[str, Any] = {}
        child_ran = asyncio.Event()

        async def observe_process_direct(task: str, **kwargs: Any) -> str:
            observed["task"] = task
            observed["child_wfid"] = DBOS.workflow_id
            child_ran.set()
            return "subagent done"

        registry, executor = _build_stack()

        mock_loop = MagicMock()
        mock_loop.process_direct = observe_process_direct

        @DBOS.workflow()
        async def parent_turn() -> str:
            observed["parent_wfid"] = DBOS.workflow_id
            return await executor.execute_tool(
                registry,
                "spawn",
                {"task": "do real work", "label": "child-1"},
                ToolContext(session_key="cli:test", channel="cli", chat_id="test"),
                tool_call_id="tc-1",
            )

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            with SetWorkflowID("parent-1"):
                result = await parent_turn()
            await asyncio.wait_for(child_ran.wait(), timeout=3.0)

        # The spawn tool's return string is the LLM-visible message.
        assert "child-1" in result or "started" in result.lower(), result

        # The child actually ran and got its own DBOS workflow id.
        assert observed["task"] == "do real work"
        assert observed["parent_wfid"] == "parent-1"
        assert observed["child_wfid"] is not None
        assert observed["child_wfid"] != observed["parent_wfid"], (
            "child workflow inherited the parent's wfid — concurrent "
            "subagents will race into the parent's step journal"
        )

    async def test_executor_implements_message_buffer_protocol(self, dbos_instance: Any) -> None:
        """``DBOSExecutor`` must implement the message-buffer methods the
        Executor protocol requires — without them, subagent.process_direct
        explodes with ``AttributeError: 'DBOSExecutor' object has no
        attribute 'set_messages'`` the moment it tries to build a prompt.
        """
        executor = DBOSExecutor()
        msgs: list[dict[str, object]] = [{"role": "user", "content": "hi"}]
        executor.set_messages(msgs)
        assert executor.load_messages() == msgs
        executor.append_messages([{"role": "assistant", "content": "ok"}])
        assert len(executor.load_messages()) == 2
