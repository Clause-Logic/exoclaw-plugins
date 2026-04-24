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

    async def test_publish_outbound_step_replayed_on_recovery(self, dbos_instance: Any) -> None:
        """``_publish_outbound_step`` is a @DBOS.step, so on recovery DBOS
        replays the journaled completion without calling ``bus.publish_outbound``
        a second time. This is what makes the final reply send survive a
        container restart mid-turn."""
        from dbos._schemas.system_database import SystemSchema
        from exoclaw_executor_dbos import turn as turn_mod
        from exoclaw_executor_dbos.turn import _publish_outbound_step

        fake_loop = MagicMock()
        fake_loop.bus.publish_outbound = AsyncMock()
        prior_loop = turn_mod._loop
        turn_mod._loop = fake_loop
        try:

            @DBOS.workflow()
            async def single_publish_workflow() -> None:
                await _publish_outbound_step(
                    session_id="sess1",
                    channel="cli",
                    chat_id="main",
                    content="final reply",
                )

            wfid = str(uuid.uuid4())
            with SetWorkflowID(wfid):
                await single_publish_workflow()

            assert fake_loop.bus.publish_outbound.await_count == 1
            call_args = fake_loop.bus.publish_outbound.await_args
            assert call_args is not None
            sent = call_args[0][0]
            assert sent.content == "final reply"
            assert sent.channel == "cli"
            assert sent.chat_id == "main"

            # Force workflow back to PENDING and recover.
            with dbos_instance._sys_db.engine.begin() as conn:
                conn.execute(
                    sa.update(SystemSchema.workflow_status)
                    .values({"status": "PENDING"})
                    .where(SystemSchema.workflow_status.c.workflow_uuid == wfid)
                )

            handles = DBOS._recover_pending_workflows()
            assert len(handles) == 1
            handles[0].get_result()

            # Step was replayed from the journal — bus.publish_outbound
            # must NOT have been called a second time.
            assert fake_loop.bus.publish_outbound.await_count == 1, (
                "publish_outbound was called on replay — step journal not honored"
            )
        finally:
            turn_mod._loop = prior_loop

    async def test_chat_context_exceeded_not_retried(self, dbos_instance: Any) -> None:
        """``ContextWindowExceededError`` from the provider must surface
        to the caller *immediately* and *as itself* — not retried three
        times by DBOS and then re-raised as
        ``DBOSMaxStepRetriesExceeded``.

        Retries on this class burn 4+ seconds and the provider's quota
        for a deterministic failure, and the wrapper exception class
        doesn't match ``AgentLoop._run_agent_loop``'s
        ``except ContextWindowExceededError`` guard — which is where
        ``on_context_overflow`` compaction lives. Silently breaking
        compaction this way is exactly how openclaw's 2026-04-24
        feed-digest turn landed on ``Sorry, I encountered an error``
        instead of compacting and retrying.
        """
        from exoclaw.providers.types import ContextWindowExceededError
        from exoclaw_executor_dbos.executor import DBOSExecutor

        provider = MagicMock()
        provider.chat = AsyncMock(side_effect=ContextWindowExceededError("prompt too long"))

        executor = DBOSExecutor()

        @DBOS.workflow()
        async def run_once() -> str:
            try:
                await executor.chat(
                    provider,
                    messages=[{"role": "user", "content": "hi"}],
                )
            except ContextWindowExceededError as e:
                return f"ok: {e}"
            return "no exception"

        with SetWorkflowID(str(uuid.uuid4())):
            result = await run_once()

        assert result.startswith("ok:"), (
            f"expected ContextWindowExceededError to surface, got {result!r}"
        )
        assert provider.chat.await_count == 1, (
            f"provider.chat should be invoked exactly once (no retries on "
            f"deterministic failure), got {provider.chat.await_count}"
        )

    async def test_executor_advertises_handles_response_send(self) -> None:
        """``DBOSExecutor`` tells the core it will own the publish so
        ``_process_message`` returns None and the outer agent loop skips
        the non-workflow publish path."""
        from exoclaw_executor_dbos.executor import DBOSExecutor

        assert DBOSExecutor.handles_response_send is True

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

    async def test_append_message_step_replayed_on_recovery(self, dbos_instance: Any) -> None:
        """``_append_message_step`` journals its completion, so on
        workflow recovery DBOS returns the recorded result without
        re-invoking ``conversation.append``. Without the step
        boundary, a crash between the append body finishing and the
        workflow returning would re-append the same message to the
        session JSONL on replay (DefaultConversation.append is not
        idempotent at the filesystem level — two writes → two lines).
        """
        from dbos._schemas.system_database import SystemSchema
        from exoclaw_executor_dbos.executor import (
            _append_message_step,
            _conversation_var,
        )

        conversation = MagicMock()
        conversation.append = AsyncMock()
        _conversation_var.set(conversation)

        @DBOS.workflow()
        async def single_append_workflow() -> None:
            await _append_message_step(
                "sess-append-1",
                {"role": "assistant", "content": "hi"},
            )

        wfid = str(uuid.uuid4())
        with SetWorkflowID(wfid):
            await single_append_workflow()

        assert conversation.append.await_count == 1

        # Force the workflow back to PENDING to simulate a crash
        # between step-body completion and workflow return.
        with dbos_instance._sys_db.engine.begin() as conn:
            conn.execute(
                sa.update(SystemSchema.workflow_status)
                .values({"status": "PENDING"})
                .where(SystemSchema.workflow_status.c.workflow_uuid == wfid)
            )

        _conversation_var.set(conversation)
        handles = DBOS._recover_pending_workflows()
        assert len(handles) == 1
        handles[0].get_result()

        # Step was replayed from the journal — conversation.append
        # must NOT have been called a second time.
        assert conversation.append.await_count == 1, (
            "append was called on replay — step journal not honored"
        )

    async def test_post_turn_step_replayed_on_recovery(self, dbos_instance: Any) -> None:
        """``_post_turn_step`` must replay from the journal so the
        end-of-turn hooks (agent_end, consolidation triggers) don't
        fire twice across a crash boundary."""
        from dbos._schemas.system_database import SystemSchema
        from exoclaw_executor_dbos.executor import (
            _conversation_var,
            _post_turn_step,
        )

        conversation = MagicMock()
        conversation.post_turn = AsyncMock()
        _conversation_var.set(conversation)

        @DBOS.workflow()
        async def single_post_turn_workflow() -> None:
            await _post_turn_step("sess-post-1")

        wfid = str(uuid.uuid4())
        with SetWorkflowID(wfid):
            await single_post_turn_workflow()

        assert conversation.post_turn.await_count == 1

        with dbos_instance._sys_db.engine.begin() as conn:
            conn.execute(
                sa.update(SystemSchema.workflow_status)
                .values({"status": "PENDING"})
                .where(SystemSchema.workflow_status.c.workflow_uuid == wfid)
            )

        _conversation_var.set(conversation)
        handles = DBOS._recover_pending_workflows()
        assert len(handles) == 1
        handles[0].get_result()

        assert conversation.post_turn.await_count == 1, (
            "post_turn hooks fired twice on replay — step journal not honored"
        )

    async def test_dbos_batch_store_survives_simulated_restart(
        self, dbos_instance: Any, tmp_path: Any
    ) -> None:
        """Reproduces the production failure fixed by moving batch state
        onto disk.

        Setup mirrors the incident from 2026-04-23: 6 subagents registered
        into one batch, 3 complete on the original process, the process
        is restarted (in the real incident DBOS replayed the remaining 3
        subagent workflows on a fresh ``SubagentManager`` whose in-memory
        ``_batches`` dict was empty), the last 3 complete. With the old
        ``InMemoryBatchStore`` no announcement fires — the test would
        catch that as an assertion on ``announce_calls``. With
        ``DBOSBatchStore`` the state lives in ``tmp_path`` and survives.
        """
        from exoclaw_executor_dbos.batch_store import DBOSBatchStore
        from exoclaw_subagent import BatchSnapshot

        store = DBOSBatchStore(workspace=tmp_path)
        batch_id = "feed-digest-retry"
        task_ids = [f"t{i}" for i in range(6)]
        announce_calls: list[BatchSnapshot] = []

        async def _announce(snap: BatchSnapshot) -> None:
            announce_calls.append(snap)

        # Register all 6 — wrapped in one workflow so the register steps
        # journal against the same wfid.
        @DBOS.workflow()
        async def register_all() -> None:
            for tid in task_ids:
                await store.register(
                    batch_id,
                    tid,
                    session_key="zulip:test",
                    origin_channel="zulip",
                    origin_chat_id="test",
                )

        with SetWorkflowID(str(uuid.uuid4())):
            await register_all()

        # Complete the first 3 subagents — each as its own workflow (that's
        # how real subagents run under DBOSSubagentSpawner).
        async def _complete_one(task_id: str) -> None:
            @DBOS.workflow()
            async def wf() -> None:
                await store.record_completion_and_maybe_announce(
                    batch_id,
                    task_id,
                    status="completed",
                    label=f"enrich-{task_id}",
                    result_path=f"/tmp/{task_id}.md",
                    announce=_announce,
                )

            with SetWorkflowID(str(uuid.uuid4())):
                await wf()

        for tid in task_ids[:3]:
            await _complete_one(tid)

        assert announce_calls == [], (
            "batch announced before all 6 members finished — should_announce decision is wrong"
        )

        # "Restart": instantiate a fresh DBOSBatchStore pointing at the
        # same on-disk state. This simulates the production failure
        # mode — a new SubagentManager on a recovered process gets a
        # fresh in-memory ``_batches`` dict, and the fix (file-backed
        # state) means this new store still sees the prior 6 members.
        #
        # Deliberately does NOT destroy+relaunch DBOS: the original
        # test did, but that invalidated the session-scoped fixture
        # for any tests that ran afterwards. The batch-state survival
        # is a DBOSBatchStore invariant, not a DBOS-journal invariant,
        # so the simpler simulation is the honest one.
        restarted_store = DBOSBatchStore(workspace=tmp_path)
        store = restarted_store  # pick up from the new store for the rest

        # Complete the remaining 3.
        for tid in task_ids[3:]:
            await _complete_one(tid)

        assert len(announce_calls) == 1, (
            f"expected exactly one batch announcement post-restart, got "
            f"{len(announce_calls)} — the in-memory-state bug would have "
            f"produced zero announcements here"
        )
        announced = announce_calls[0]
        assert announced.batch_id == batch_id
        assert announced.total == 6
        assert announced.completed == 6
