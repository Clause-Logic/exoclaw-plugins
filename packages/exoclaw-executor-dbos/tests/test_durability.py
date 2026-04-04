"""Integration test: prove DBOS step replay on workflow recovery.

Runs a workflow containing a _chat_step, forces the workflow status back
to PENDING (simulating a crash), recovers, and asserts the LLM provider
was NOT called again — DBOS replayed the step result from its journal.
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from dbos import DBOS, DBOSConfig, SetWorkflowID

_DB_PATH = f"/tmp/dbos_test_{os.getpid()}.sqlite"

# Session-scoped so DBOS stays alive across all async tests
# (avoids thread pool shutdown between function-scoped event loops)


@pytest.fixture(autouse=True, scope="session")
def dbos_instance() -> Any:
    """Spin up a DBOS instance for the session."""
    DBOS.destroy()

    config: DBOSConfig = {
        "name": "durability-test",
        "system_database_url": f"sqlite:///{_DB_PATH}",
        "enable_otlp": False,
    }
    import exoclaw_executor_dbos.executor  # noqa: F401
    import exoclaw_executor_dbos.turn  # noqa: F401

    dbos = DBOS(config=config)
    DBOS.launch()
    yield dbos
    DBOS.destroy()
    if os.path.exists(_DB_PATH):
        os.unlink(_DB_PATH)


def _make_response(content: str = "hello") -> Any:
    from exoclaw.providers.types import LLMResponse

    return LLMResponse(content=content, finish_reason="stop", tool_calls=[])


@pytest.mark.asyncio(loop_scope="session")
class TestDurability:
    async def test_chat_step_replayed_on_recovery(self, dbos_instance: Any) -> None:
        """On recovery, _chat_step returns the journaled result without calling provider.chat again."""
        from dbos._schemas.system_database import SystemSchema
        from exoclaw_executor_dbos.executor import _chat_step, _provider_var

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=_make_response("original answer"))

        @DBOS.workflow()
        async def single_chat_workflow() -> dict[str, Any]:
            _provider_var.set(provider)
            return await _chat_step(
                messages=[{"role": "user", "content": "hi"}],
                tools_json=None,
                model=None,
                temperature=0.7,
                max_tokens=4096,
                reasoning_effort=None,
            )

        wfid = str(uuid.uuid4())
        with SetWorkflowID(wfid):
            result = await single_chat_workflow()

        assert result["content"] == "original answer"
        assert provider.chat.await_count == 1

        # Simulate crash: force workflow back to PENDING
        with dbos_instance._sys_db.engine.begin() as conn:
            conn.execute(
                sa.update(SystemSchema.workflow_status)
                .values({"status": "PENDING"})
                .where(SystemSchema.workflow_status.c.workflow_uuid == wfid)
            )

        # Recover — DBOS re-enters the workflow but replays the step from journal
        handles = DBOS._recover_pending_workflows()
        assert len(handles) == 1
        recovered = handles[0].get_result()

        assert recovered["content"] == "original answer"
        # provider.chat was NOT called again — step was replayed
        assert provider.chat.await_count == 1

    async def test_tool_step_replayed_on_recovery(self, dbos_instance: Any) -> None:
        """On recovery, _tool_step returns the journaled result without calling registry.execute again."""
        from dbos._schemas.system_database import SystemSchema
        from exoclaw_executor_dbos.executor import _registry_var, _tool_step

        registry = MagicMock()
        registry.execute = AsyncMock(return_value="tool output")

        @DBOS.workflow()
        async def single_tool_workflow() -> str:
            _registry_var.set(registry)
            return await _tool_step(
                name="my_tool",
                params={"q": "test"},
                ctx_data=None,
            )

        wfid = str(uuid.uuid4())
        with SetWorkflowID(wfid):
            result = await single_tool_workflow()

        assert result == "tool output"
        assert registry.execute.await_count == 1

        # Simulate crash
        with dbos_instance._sys_db.engine.begin() as conn:
            conn.execute(
                sa.update(SystemSchema.workflow_status)
                .values({"status": "PENDING"})
                .where(SystemSchema.workflow_status.c.workflow_uuid == wfid)
            )

        # Recover
        handles = DBOS._recover_pending_workflows()
        assert len(handles) == 1
        assert handles[0].get_result() == "tool output"
        # registry.execute was NOT called again
        assert registry.execute.await_count == 1
