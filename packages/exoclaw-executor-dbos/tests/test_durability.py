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

    async def test_mint_turn_id_replayed_on_recovery(self, dbos_instance: Any) -> None:
        """On recovery, ``_mint_turn_id_step`` returns the journaled id.

        This is the load-bearing invariant for stage-1 turn observability:
        if a workflow crashes mid-turn and DBOS replays it, the rebound
        ``turn.id`` contextvar must contain the SAME uuid as the original
        run — otherwise log lines emitted before vs after the crash land
        in two different ``turn.root_id`` buckets and the whole
        single-query-debug story falls apart.

        We mint two ids inside a workflow (proving they're distinct on
        first run), force the workflow back to PENDING, recover, and
        assert the recovered workflow returned the *exact same pair* of
        ids — not freshly minted ones.
        """
        from dbos._schemas.system_database import SystemSchema
        from exoclaw_executor_dbos.executor import _mint_turn_id_step

        @DBOS.workflow()
        async def two_mints_workflow() -> tuple[str, str]:
            first = await _mint_turn_id_step()
            second = await _mint_turn_id_step()
            return first, second

        wfid = str(uuid.uuid4())
        with SetWorkflowID(wfid):
            original_first, original_second = await two_mints_workflow()

        # Sanity: each call inside the workflow body produces a distinct id.
        assert original_first != original_second
        assert len(original_first) == 36  # uuid string format
        assert len(original_second) == 36

        # Simulate crash mid-turn — the workflow has already journaled
        # both step results, but we force it back to PENDING so DBOS
        # treats it as needing recovery.
        with dbos_instance._sys_db.engine.begin() as conn:
            conn.execute(
                sa.update(SystemSchema.workflow_status)
                .values({"status": "PENDING"})
                .where(SystemSchema.workflow_status.c.workflow_uuid == wfid)
            )

        handles = DBOS._recover_pending_workflows()
        assert len(handles) == 1
        recovered_first, recovered_second = handles[0].get_result()

        assert recovered_first == original_first, (
            "first turn id was re-minted on replay — DBOS step journal not honored"
        )
        assert recovered_second == original_second, (
            "second turn id was re-minted on replay — DBOS step journal not honored"
        )

    async def test_dbos_executor_mint_turn_id_returns_uuid7(self, dbos_instance: Any) -> None:
        """``DBOSExecutor.mint_turn_id`` (the public entrypoint exoclaw's
        ``_process_turn_inline`` calls) must return a uuidv7 string when
        invoked from inside a workflow.

        Validates the executor method end-to-end: it has to delegate to
        the step (so it's replay-safe) AND return a real uuidv7 (so the
        log-sort-by-turn.id property holds).
        """
        from uuid import UUID

        from exoclaw_executor_dbos.executor import DBOSExecutor

        executor = DBOSExecutor()

        @DBOS.workflow()
        async def call_executor_mint() -> str:
            return await executor.mint_turn_id()

        with SetWorkflowID(str(uuid.uuid4())):
            value = await call_executor_mint()

        parsed = UUID(value)
        assert parsed.version == 7, f"expected uuidv7, got version {parsed.version}"
