"""Reproducers for cron concurrency / context-leak bugs.

A class of bugs in tools and background-task plumbing where per-call
state is owned by whoever last touched a singleton, not by the run
that should own it. The bugs split into two flavors:

* **Singleton instance-attr destinations** — tools store routing
  state (channel, chat_id, session_key) as plain instance
  attributes mutated by ``set_context``. The instance is shared
  across every concurrent turn, so whichever caller last wrote
  the attrs owns routing for every other caller until someone
  else stomps them. Manifests as cross-channel routing: a cron
  callback's mid-turn message goes to the user's active topic
  instead of the cron's intended destination.

* **DBOS context leak via ``asyncio.create_task``** — Python's
  ``create_task`` copies the calling task's contextvars,
  including DBOS's ``_dbos_context_var`` (workflow_id +
  function_id snapshot). A background task spawned from inside
  a workflow body inherits that context. When the task later
  invokes a ``@DBOS.workflow`` of its own, DBOS misclassifies
  the call as a child step of the long-finished parent and
  raises ``DBOSUnexpectedStepError`` against the parent's
  already-journaled step at that function_id (or, if the parent
  is still alive, the bare assertion at ``dbos/_context.py``
  forbidding ``start_workflow`` from inside a step).

Tests 1-3 cover the original incident shape: cron timer leak,
MessageTool singleton, SpawnTool singleton.

Tests 4-6 cover three more call sites with the same root causes:

4. **CronTool destination is a singleton** (same pattern as Tests
   2 & 3). ``execute_with_context`` writes ``self._channel`` /
   ``self._chat_id`` and ``execute`` reads them. Concurrent calls
   don't actually race in current code (the read happens
   synchronously before any await yields), but a non-context call
   that follows a context-bound call inherits the leftover state.
5. **Conversation consolidation create_task DBOS leak** (same
   pattern as Test 1). ``DefaultConversation.build_prompt`` spawns
   ``_consolidate_and_unlock`` via plain ``asyncio.create_task``,
   which copies the calling task's contextvars. Reached from
   inside a workflow, the spawned task carries DBOSContext.
6. **AsyncioSpawner DBOS leak** (same pattern as Test 1). Already
   replaced in production by ``DBOSSubagentSpawner``, but
   ``AsyncioSpawner`` is still the default for tests / CLI.

Other places exhibiting the same patterns:

* ``HeartbeatService`` shares the same MessageTool / SpawnTool
  singletons constructed by the nanobot — concurrent heartbeat +
  cron + user turn races the same way as Tests 2 & 3.
* ``AgentLoop.run`` dispatch
  (``exoclaw/agent/loop.py``): top-of-funnel for inbound
  messages. Today ``run()`` lives in the bus consumer task with
  no workflow on the stack so the leak doesn't manifest;
  intentionally untested because writing a failing case requires
  artificially placing ``run()`` inside a workflow. Worth a
  defense-in-depth ``create_isolated_task`` at this call site
  alongside the other fixes — it's the busiest ``create_task``
  in the codebase.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Importing the module registers the @DBOS.workflow() decorators we
# rely on (run_durable_turn lives there). Imported at module scope so
# DBOS sees them before launch.
import exoclaw_executor_dbos.turn  # noqa: F401, E402
import pytest
from dbos import DBOS, DBOSConfig, SetWorkflowID
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_subagent import SpawnTool, SubagentManager
from exoclaw_subagent.spawner import AsyncioSpawner
from exoclaw_tools_cron.service import CronService
from exoclaw_tools_cron.tool import CronTool
from exoclaw_tools_cron.types import CronJob, CronJobState, CronPayload, CronSchedule
from exoclaw_tools_message.tool import MessageTool

_DB_PATH = f"/tmp/dbos_cron_concurrency_test_{os.getpid()}.sqlite"

# Module-global handle to the cron service the test installs. The
# parent workflow can't take CronService as an argument because DBOS
# pickles workflow args into the system DB. Lookup-by-key keeps the
# arguments to plain ints/strs.
_SERVICES: dict[str, CronService] = {}


@pytest.fixture(scope="module")
def dbos_instance() -> Any:
    """Module-scoped DBOS fixture.

    Mirrors ``test_spawn_integration.py`` — DBOS is a process-global
    singleton; module scope keeps the lifetime contained without
    stepping on other test modules' SQLite paths.
    """
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "cron-concurrency-test",
        "system_database_url": f"sqlite:///{_DB_PATH}",
        "enable_otlp": False,
    }
    dbos = DBOS(config=config)
    DBOS.launch()
    yield dbos
    DBOS.destroy()
    if os.path.exists(_DB_PATH):
        os.unlink(_DB_PATH)


# ── Module-level workflows / steps for Test 1 ────────────────────────
# DBOS requires @DBOS.workflow / @DBOS.step decorators at import time
# so the registry knows about them before launch. They can't be defined
# inside test functions.


@DBOS.step()
async def _noop_step() -> None:
    """A step that records function_id but does nothing."""
    return None


@DBOS.step()
async def _add_cron_step(svc_key: str, when_ms: int, name: str) -> None:
    """Schedule a cron job from inside a DBOS step.

    This is the production-faithful call site: the cron tool's
    ``_add_job`` runs inside ``_tool_step`` (a DBOS step), and from
    there it calls into ``CronService.add_job`` which arms an
    asyncio timer task.

    Looks the service up by key from a module-global registry so the
    workflow's arguments stay picklable.
    """
    svc = _SERVICES[svc_key]
    svc.add_job(
        name=name,
        schedule=CronSchedule(kind="at", at_ms=when_ms),
        message=name,
    )


@DBOS.workflow()
async def _scheduling_parent_workflow(svc_key: str, when_ms: int) -> None:
    """Mimics a user turn that schedules a cron job mid-flight.

    The leading and trailing no-op steps advance the workflow's
    function_id counter so that when the timer fires later carrying
    a stale DBOSContext snapshot, the function_id it tries to
    record at collides with steps the parent already journaled.
    """
    await _noop_step()
    await _noop_step()
    await _add_cron_step(svc_key, when_ms, "test-cron")
    await _noop_step()
    await _noop_step()


@DBOS.workflow()
async def _cron_inner_workflow(label: str) -> str:
    """Stand-in for ``run_durable_turn``: a top-level workflow the
    cron callback wants to start. If it ever runs as a child step of
    the parent (the bug), DBOS raises ``DBOSUnexpectedStepError``."""
    return f"cron ran: {label}"


# ── Tests 5 & 6: shared scaffolding for create_task DBOS leaks ──────


@DBOS.workflow()
async def _bg_inner_workflow(label: str) -> str:
    """Same role as ``_cron_inner_workflow`` but used by the
    consolidation / spawner leak tests. Reusing one workflow across
    both tests is fine because each invocation gets a fresh wfid via
    ``SetWorkflowID`` — DBOS only collides when the wfid matches an
    existing journal entry."""
    return f"bg ran: {label}"


@DBOS.step()
async def _consolidation_trigger_step(memory_holder_key: str) -> None:
    """Drive ``DefaultConversation.build_prompt`` from inside a DBOS
    step so the conversation's background ``asyncio.create_task``
    fires while a real workflow context is on the stack.

    Looks the conversation up by key for the same picklability reason
    as ``_add_cron_step``."""
    conv: DefaultConversation = _SERVICES[memory_holder_key]  # type: ignore[assignment]
    await conv.build_prompt("test-session", "hi", channel="zulip", chat_id="topic-A")


@DBOS.workflow()
async def _consolidation_parent_workflow(memory_holder_key: str) -> None:
    """Mirror of ``_scheduling_parent_workflow`` for the consolidation
    leak. Leading no-ops advance function_id so the spawned
    consolidation task — if it inherits the workflow context — would
    collide on ``_bg_inner_workflow`` against an already-journaled
    step."""
    await _noop_step()
    await _noop_step()
    await _consolidation_trigger_step(memory_holder_key)
    await _noop_step()
    await _noop_step()


@DBOS.step()
async def _spawner_start_step(spawner_key: str) -> None:
    """Drive ``AsyncioSpawner.start`` from inside a DBOS step so the
    spawner's ``asyncio.create_task`` captures a live workflow
    context."""
    spawner: AsyncioSpawner = _SERVICES[spawner_key]  # type: ignore[assignment]
    handle = await spawner.start(
        task_id="t1",
        task="do work",
        label="lbl",
        origin_channel="zulip",
        origin_chat_id="topic-A",
        session_key="s",
        batch=None,
        skills=None,
        model=None,
    )
    # Hold the handle so the test fixture can await it
    _SERVICES[f"{spawner_key}:handle"] = handle  # type: ignore[assignment]


@DBOS.workflow()
async def _spawner_parent_workflow(spawner_key: str) -> None:
    await _noop_step()
    await _noop_step()
    await _spawner_start_step(spawner_key)
    await _noop_step()
    await _noop_step()


@pytest.mark.asyncio(loop_scope="session")
class TestCronInheritsDBOSContextFromSchedulingTurn:
    """Test 1 — the contextvar leak that produces ``DBOSUnexpectedStepError``.

    Scheduling a cron from inside a DBOS workflow leaks the parent
    workflow's ``DBOSContext`` into the asyncio timer task. When the
    timer fires later, the cron's invocation of a top-level
    ``@DBOS.workflow`` is misclassified as a child step of the
    parent and crashes against the parent's already-journaled steps.
    """

    async def test_cron_fired_after_parent_workflow_runs_as_top_level(
        self, dbos_instance: Any
    ) -> None:
        cron_fired = asyncio.Event()
        cron_error: list[BaseException] = []

        async def on_job(job: CronJob) -> None:
            try:
                # Production path: ``_on_cron_job`` invokes
                # ``executor.run_turn`` which wraps the call in
                # ``with SetWorkflowID(...): await run_durable_turn(...)``
                # We mirror that exactly with a stand-in workflow.
                wfid = f"cron-{job.id}-{uuid.uuid4().hex[:8]}"
                with SetWorkflowID(wfid):
                    await _cron_inner_workflow(job.name)
            except BaseException as e:
                cron_error.append(e)
            finally:
                cron_fired.set()

        store_path = Path(tempfile.mkdtemp(prefix="cron-test-")) / "jobs.json"
        svc_key = f"svc-{uuid.uuid4().hex[:8]}"
        svc = CronService(store_path=store_path, on_job=on_job)
        _SERVICES[svc_key] = svc
        await svc.start()
        try:
            when_ms = int(time.time() * 1000) + 500  # fire ~0.5s out
            with SetWorkflowID(f"parent-{uuid.uuid4().hex[:8]}"):
                await _scheduling_parent_workflow(svc_key, when_ms)

            # Parent workflow has now completed and its steps are journaled.
            # The timer task — created during the parent's _add_cron_step —
            # is still pending, carrying the parent's DBOSContext snapshot.
            await asyncio.wait_for(cron_fired.wait(), timeout=5.0)
        finally:
            svc.stop()
            _SERVICES.pop(svc_key, None)

        assert not cron_error, (
            f"Cron firing failed with {type(cron_error[0]).__name__}: "
            f"{cron_error[0]}.\n"
            f"The timer task "
            f"inherited the scheduling workflow's DBOSContext via "
            f"asyncio.create_task's contextvars copy, so the cron's "
            f"top-level workflow start was treated as a child step of the "
            f"long-finished parent."
        )


@pytest.mark.asyncio(loop_scope="session")
class TestMessageToolDestinationIsSingleton:
    """Test 2 — MessageTool's destination is process-global mutable state.

    The nanobot constructs one ``MessageTool`` (``app.py:332``) and
    shares it across every turn — user, cron, heartbeat, subagent.
    ``set_context`` mutates instance attrs in place. ``execute``
    falls back to those defaults when the agent doesn't pass channel
    / chat_id explicitly. There is no per-turn isolation.

    Fixing this means making the destination per-task (a
    ``ContextVar``) rather than per-instance. After that fix both
    tests below should pass.
    """

    async def test_concurrent_turns_each_send_to_their_own_destination(
        self,
    ) -> None:
        """Two concurrent turns each ``set_context`` to their own
        destination, then ``execute``. Each send should reach the
        destination its own turn set — they must not collide on
        shared singleton state."""
        sent: list[Any] = []

        async def capture(msg: Any) -> None:
            sent.append(msg)

        tool = MessageTool(send_callback=capture)

        async def turn(channel: str, chat_id: str, content: str) -> None:
            tool.set_context(channel=channel, chat_id=chat_id)
            # Yield so the other turn can interleave its set_context
            # before this one calls execute. This is exactly the
            # interleaving that happens when an inbound user message
            # arrives while a cron's agent loop is mid-flight.
            await asyncio.sleep(0.01)
            await tool.execute(content=content)

        await asyncio.gather(
            turn("zulip", "topic-A-user", "user message"),
            turn("zulip", "topic-B-cron", "cron message"),
        )

        by_content = {m.content: m.chat_id for m in sent}
        assert by_content.get("user message") == "topic-A-user", (
            f"user's message was routed to "
            f"{by_content.get('user message')!r}, not topic-A-user. "
            f"MessageTool's destination is shared instance state, so "
            f"the cron's set_context overwrote the user's."
        )
        assert by_content.get("cron message") == "topic-B-cron", (
            f"cron's message was routed to "
            f"{by_content.get('cron message')!r}, not topic-B-cron — "
            f"same singleton race in the opposite direction."
        )

    async def test_destination_does_not_leak_across_sequential_turns(
        self,
    ) -> None:
        """Two turns run sequentially, each in its own task — production
        shape: the cron callback runs in the timer's task, an inbound
        user message dispatches in ``AgentLoop.run``'s per-message
        task. Each turn binds its destination via ``set_context`` and
        sends. Each send must reach its own binding; the second turn
        must not inherit the first turn's binding."""
        sent: list[Any] = []

        async def capture(msg: Any) -> None:
            sent.append(msg)

        tool = MessageTool(send_callback=capture)

        async def cron_turn() -> None:
            tool.set_context(channel="zulip", chat_id="cron-target")
            await tool.execute(content="cron output")

        async def user_turn() -> None:
            tool.set_context(channel="zulip", chat_id="user-target")
            await tool.execute(content="user output")

        await asyncio.create_task(cron_turn())
        await asyncio.create_task(user_turn())

        by_content = {m.content: m.chat_id for m in sent}
        assert by_content.get("cron output") == "cron-target", (
            f"cron's send went to {by_content.get('cron output')!r}"
        )
        assert by_content.get("user output") == "user-target", (
            f"user's send went to {by_content.get('user output')!r} — "
            f"cron's destination from the previous task leaked across "
            f"task boundaries. Per-task ContextVar isolation would "
            f"keep cron's set_context inside its own task."
        )


@pytest.mark.asyncio(loop_scope="session")
class TestSpawnToolDestinationIsSingleton:
    """Test 3 — SpawnTool has the same singleton design as MessageTool
    and IS exercised in production by ``_on_cron_job``.

    ``execute_with_context`` reads from the per-call ``ToolContext``
    so the registry path is fine when ``ctx is not None``. But
    ``execute`` (the no-ctx fallback path) reads from instance
    state, and ``set_context`` mutates that state in place.
    Concurrent callers race on those instance attrs.
    """

    def _make_tool(self) -> tuple[SpawnTool, list[dict[str, Any]]]:
        spawned: list[dict[str, Any]] = []

        async def fake_spawn(**kwargs: Any) -> str:
            spawned.append(dict(kwargs))
            return "spawned"

        manager = MagicMock(spec=SubagentManager)
        manager.spawn = AsyncMock(side_effect=fake_spawn)
        return SpawnTool(manager=manager), spawned

    async def test_concurrent_turns_each_spawn_to_their_own_destination(
        self,
    ) -> None:
        """Two concurrent turns each set spawn context for their own
        destination, then dispatch a spawn through ``execute()``.
        Each subagent should be queued with its caller's destination,
        not the other caller's."""
        tool, spawned = self._make_tool()

        async def turn(channel: str, chat_id: str, task: str) -> None:
            tool.set_context(channel=channel, chat_id=chat_id)
            # Yield so the other turn's set_context can interleave
            await asyncio.sleep(0.01)
            await tool.execute(action="spawn", task=task)

        await asyncio.gather(
            turn("zulip", "topic-A-user", "user task"),
            turn("zulip", "topic-B-cron", "cron task"),
        )

        by_task = {s["task"]: s["origin_chat_id"] for s in spawned}
        assert by_task.get("user task") == "topic-A-user", (
            f"user's subagent was queued with destination "
            f"{by_task.get('user task')!r}, not topic-A-user — "
            f"SpawnTool's instance attrs were overwritten by the "
            f"concurrent cron set_context call."
        )
        assert by_task.get("cron task") == "topic-B-cron", (
            f"cron's subagent was queued with destination "
            f"{by_task.get('cron task')!r}, not topic-B-cron — "
            f"same race in the opposite direction."
        )

    async def test_spawn_destination_does_not_leak_across_sequential_turns(
        self,
    ) -> None:
        """``_on_cron_job`` (``app.py:492``) sets spawn context to the
        cron's destination. Its finally clause re-calls set_context
        with the same channel/chat_id and ``skills=None`` — channel
        / chat_id are never restored to a prior value. With singleton
        instance attrs, a later turn that dispatches through
        ``execute()`` inherits the cron's destination. With per-task
        isolation, the cron callback's writes don't escape its task."""
        tool, spawned = self._make_tool()

        async def cron_callback() -> None:
            # _on_cron_job: prologue + finally
            tool.set_context(channel="zulip", chat_id="cron-target", session_key="cron:abc")
            tool.set_context(
                channel="zulip",
                chat_id="cron-target",
                session_key="cron:abc",
                skills=None,
            )
            # Cron's agent dispatches a spawn correctly to its own destination
            await tool.execute(action="spawn", task="cron task")

        async def user_turn() -> None:
            tool.set_context(channel="zulip", chat_id="user-target", session_key="user:xyz")
            await tool.execute(action="spawn", task="user task")

        await asyncio.create_task(cron_callback())
        await asyncio.create_task(user_turn())

        by_task = {s["task"]: s["origin_chat_id"] for s in spawned}
        assert by_task.get("cron task") == "cron-target"
        assert by_task.get("user task") == "user-target", (
            f"User's subagent was queued with origin_chat_id="
            f"{by_task.get('user task')!r} — cron's leftover destination. "
            f"SpawnTool's singleton state outlived the cron callback."
        )


@pytest.mark.asyncio(loop_scope="session")
class TestCronToolDestinationLeaksWhenCtxAbsent:
    """Test 4 — CronTool stores destination on instance attrs.

    ``execute_with_context`` writes ``self._channel = ctx.channel`` /
    ``self._chat_id = ctx.chat_id`` and synchronously hands off to
    ``execute()`` which reads them in ``_add_job``. Because the read
    happens synchronously before any ``await`` yields, the
    *concurrent* race window is empty in current code — concurrent
    callers don't actually trip.

    The leak that DOES manifest: once any context-bound call has
    populated the instance state, subsequent calls that go through
    ``execute()`` directly (registry path when ``ctx is None``)
    inherit the leftover state. Per-task ContextVars would isolate
    each caller's binding to its own task.
    """

    def _make_tool(self) -> tuple[CronTool, list[dict[str, Any]]]:
        added: list[dict[str, Any]] = []

        class MockBackend:
            async def add(self, **kwargs: Any) -> CronJob:
                added.append(dict(kwargs))
                return CronJob(
                    id="x",
                    name=kwargs["name"],
                    schedule=kwargs["schedule"],
                    payload=CronPayload(message=kwargs["message"]),
                    state=CronJobState(),
                )

            async def list_jobs(self, **_: Any) -> list[CronJob]:
                return []

            async def get(self, _job_id: str) -> CronJob | None:
                return None

            async def update(self, _job_id: str, **_: Any) -> CronJob | None:
                return None

            async def remove(self, _job_id: str) -> bool:
                return False

            async def enable(self, _job_id: str, enabled: bool = True) -> CronJob | None:
                return None

        return CronTool(backend=MockBackend()), added  # type: ignore[arg-type]

    async def test_destination_does_not_leak_to_no_ctx_call(self) -> None:
        """First add goes through ``execute_with_context`` (binds the
        instance to A's destination). Second add goes through
        ``execute`` directly (no ctx). With singleton state, the
        second add inherits A's destination."""
        tool, added = self._make_tool()

        async def turn_with_ctx() -> None:
            ctx = ToolContext(session_key="zulip:topic-A", channel="zulip", chat_id="topic-A")
            await tool.execute_with_context(ctx, action="add", message="task A", every_seconds=60)

        async def turn_without_ctx() -> None:
            # No ctx: registry would call execute() directly. Without
            # per-task isolation, the cron tool reads A's leftover
            # state from its instance attrs and sends task B to A.
            await tool.execute(action="add", message="task B", every_seconds=60)

        await asyncio.create_task(turn_with_ctx())
        await asyncio.create_task(turn_without_ctx())

        by_msg = {j["message"]: j["to"] for j in added}
        assert by_msg.get("task A") == "topic-A"
        assert by_msg.get("task B") != "topic-A", (
            f"task B was queued with destination "
            f"{by_msg.get('task B')!r} — turn_with_ctx's binding "
            f"leaked through CronTool's singleton instance state."
        )


@pytest.mark.asyncio(loop_scope="session")
class TestConversationConsolidationInheritsDBOSContext:
    """Test 5 — ``DefaultConversation.build_prompt`` spawns a
    background consolidation task via ``asyncio.create_task``
    (``conversation.py:169``). The task copies the calling
    contextvars, including DBOS's ``_dbos_context_var`` when
    ``build_prompt`` is reached from inside a workflow body.

    The leak is currently *dormant* in production —
    ``memory.consolidate_messages`` calls a provider directly, not a
    DBOS step — but the inherited context is still there, ready to
    misclassify any future workflow-aware call inside the
    consolidation path as a child step of the parent.

    The test forces the latent leak: it patches
    ``memory.consolidate_messages`` to attempt a fresh top-level
    workflow. With context inheritance, that becomes a child-step
    attempt against the parent and fails with
    ``DBOSUnexpectedStepError``.
    """

    async def test_consolidation_task_runs_as_top_level_workflow(self, dbos_instance: Any) -> None:
        consolidate_done = asyncio.Event()
        consolidate_error: list[BaseException] = []

        async def consolidate(*_: Any, **__: Any) -> bool:
            try:
                wfid = f"consolidate-{uuid.uuid4().hex[:8]}"
                with SetWorkflowID(wfid):
                    await _bg_inner_workflow("consolidate")
            except BaseException as e:
                consolidate_error.append(e)
            finally:
                consolidate_done.set()
            return True

        memory = MagicMock()
        memory.consolidate_messages = AsyncMock(side_effect=consolidate)
        memory.system_context = MagicMock(return_value=None)

        # Build a HistoryStore stand-in whose session reports
        # unconsolidated > memory_window so consolidation fires.
        session = MagicMock()
        session.total_messages = 200
        session.last_consolidated = 0
        session.metadata = {}
        session.get_history = MagicMock(return_value=[])
        session.key = "test-session"
        history = MagicMock()
        history.get_or_create.return_value = session
        history.load_range = MagicMock(return_value=[{"role": "user", "content": "old"}])
        history.save_metadata = MagicMock()

        prompt = MagicMock()
        prompt.build = AsyncMock(return_value=[{"role": "system", "content": ""}])

        conv = DefaultConversation(history=history, memory=memory, prompt=prompt, memory_window=10)

        key = f"conv-{uuid.uuid4().hex[:8]}"
        _SERVICES[key] = conv  # type: ignore[assignment]
        try:
            with SetWorkflowID(f"parent-{uuid.uuid4().hex[:8]}"):
                await _consolidation_parent_workflow(key)

            # The parent has finished. The consolidation task spawned
            # inside build_prompt is still running with a copy of the
            # parent's contextvars (including DBOSContext).
            await asyncio.wait_for(consolidate_done.wait(), timeout=5.0)
        finally:
            _SERVICES.pop(key, None)
            # Drain any other consolidation tasks the conversation may
            # have left around (e.g. on a different session) so they
            # don't bleed into other tests.
            for t in list(conv._consolidation_tasks):
                t.cancel()

        assert not consolidate_error, (
            f"Consolidation failed with "
            f"{type(consolidate_error[0]).__name__}: "
            f"{consolidate_error[0]!r}.\n"
            f"DefaultConversation's background consolidation task "
            f"inherited the parent workflow's DBOSContext via the "
            f"asyncio.create_task contextvars copy at "
            f"conversation.py:169. The error may surface as "
            f"DBOSUnexpectedStepError (stale context once the parent "
            f"finishes) or as a bare ``AssertionError`` from "
            f"dbos/_context.py (live step context — DBOS asserts you "
            f"can't start a workflow from inside a step). Both have "
            f"the same root cause: the spawned task should run with "
            f"a fresh ``contextvars.Context()``."
        )


@pytest.mark.asyncio(loop_scope="session")
class TestAsyncioSpawnerInheritsDBOSContext:
    """Test 6 — ``AsyncioSpawner.start`` spawns the subagent's
    coroutine via ``asyncio.create_task`` (``spawner.py:178``)
    without isolating the contextvars. When called from inside a
    DBOS workflow, the subagent task carries the parent's
    DBOSContext and any DBOS-aware work it does is misclassified.

    Production routes around this by using ``DBOSSubagentSpawner``
    (which establishes its own DBOS workflow context per spawn).
    But ``AsyncioSpawner`` is still the default for tests / CLI,
    and any consumer that picks it up while inside a workflow
    re-acquires the same bug.
    """

    async def test_spawner_task_runs_as_top_level_workflow(self, dbos_instance: Any) -> None:
        runner_done = asyncio.Event()
        runner_error: list[BaseException] = []

        async def runner(**kwargs: Any) -> None:
            try:
                wfid = f"subagent-{kwargs['task_id']}-{uuid.uuid4().hex[:8]}"
                with SetWorkflowID(wfid):
                    await _bg_inner_workflow(kwargs["label"])
            except BaseException as e:
                runner_error.append(e)
            finally:
                runner_done.set()

        spawner = AsyncioSpawner(runner=runner)
        key = f"spawner-{uuid.uuid4().hex[:8]}"
        _SERVICES[key] = spawner  # type: ignore[assignment]
        try:
            with SetWorkflowID(f"parent-{uuid.uuid4().hex[:8]}"):
                await _spawner_parent_workflow(key)

            await asyncio.wait_for(runner_done.wait(), timeout=5.0)
        finally:
            handle = _SERVICES.pop(f"{key}:handle", None)
            _SERVICES.pop(key, None)
            if handle is not None:
                # Drain the background task so it doesn't leak across
                # tests. The handle's task is the gated wrapper.
                with __import__("contextlib").suppress(BaseException):
                    await handle._task  # type: ignore[attr-defined]

        assert not runner_error, (
            f"Subagent runner failed with "
            f"{type(runner_error[0]).__name__}: "
            f"{runner_error[0]!r}.\n"
            f"AsyncioSpawner's background task inherited the parent "
            f"workflow's DBOSContext via the asyncio.create_task "
            f"contextvars copy at spawner.py. Error may surface as "
            f"DBOSUnexpectedStepError or a bare ``AssertionError`` "
            f"from dbos/_context.py depending on whether the parent "
            f"is still alive when the spawned task tries to start a "
            f"workflow."
        )
