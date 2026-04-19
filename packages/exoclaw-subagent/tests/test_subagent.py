"""Tests for exoclaw-subagent package."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from exoclaw.bus.events import InboundMessage
from exoclaw_subagent.manager import SubagentManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    return bus


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return provider


def _make_conversation() -> MagicMock:
    conv = MagicMock()
    return conv


def _make_manager(
    bus: MagicMock | None = None,
    provider: MagicMock | None = None,
    process_direct_result: str = "task done",
) -> SubagentManager:
    bus = bus or _make_bus()
    provider = provider or _make_provider()

    mgr = SubagentManager(
        provider=provider,
        bus=bus,
        conversation_factory=_make_conversation,
        max_iterations=5,
    )
    return mgr


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSubagentManagerInit:
    def test_defaults(self) -> None:
        bus = _make_bus()
        provider = _make_provider()
        mgr = SubagentManager(
            provider=provider,
            bus=bus,
            conversation_factory=_make_conversation,
        )
        assert mgr._max_iterations == 15
        assert mgr._model is None
        assert mgr._tools == []
        assert mgr.get_running_count() == 0

    def test_custom_params(self) -> None:
        mgr = SubagentManager(
            provider=_make_provider(),
            bus=_make_bus(),
            conversation_factory=_make_conversation,
            model="claude-3",
            max_iterations=5,
        )
        assert mgr._model == "claude-3"
        assert mgr._max_iterations == 5

    def test_satisfies_spawn_manager_protocol(self) -> None:
        from exoclaw_subagent import SpawnManager

        mgr = _make_manager()
        assert isinstance(mgr, SpawnManager)


# ---------------------------------------------------------------------------
# spawn
# ---------------------------------------------------------------------------


class TestSpawn:
    async def test_spawn_returns_immediately(self) -> None:
        mgr = _make_manager()
        with patch("exoclaw_subagent.manager.SubagentManager._run", new=AsyncMock()):
            result = await mgr.spawn(task="do something")
        assert "started" in result

    async def test_spawn_with_label(self) -> None:
        mgr = _make_manager()
        with patch("exoclaw_subagent.manager.SubagentManager._run", new=AsyncMock()):
            result = await mgr.spawn(task="do something", label="my task")
        assert "my task" in result

    async def test_spawn_creates_background_task(self) -> None:
        mgr = _make_manager()
        ran: list[str] = []

        async def fake_run(*args: object, **kwargs: object) -> None:
            await asyncio.sleep(0)
            ran.append("done")

        with patch.object(mgr, "_run", new=fake_run):
            await mgr.spawn(task="work")
            assert mgr.get_running_count() == 1
            await asyncio.sleep(0.01)  # let task run

        assert ran == ["done"]

    async def test_spawn_task_cleaned_up_after_completion(self) -> None:
        mgr = _make_manager()

        async def fake_run(*args: object, **kwargs: object) -> None:
            await asyncio.sleep(0)

        with patch.object(mgr, "_run", new=fake_run):
            await mgr.spawn(task="work")
            await asyncio.sleep(0.05)

        assert mgr.get_running_count() == 0

    async def test_spawn_generates_unique_ids(self) -> None:
        mgr = _make_manager()
        results = []
        with patch("exoclaw_subagent.manager.SubagentManager._run", new=AsyncMock()):
            for _ in range(3):
                r = await mgr.spawn(task="task")
                results.append(r)
        # All results should reference different IDs
        ids = [r.split("id: ")[1].rstrip(").") for r in results]
        assert len(set(ids)) == 3

    async def test_label_truncated_from_task(self) -> None:
        mgr = _make_manager()
        long_task = "a" * 50
        with patch("exoclaw_subagent.manager.SubagentManager._run", new=AsyncMock()):
            result = await mgr.spawn(task=long_task)
        assert "..." in result


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------


class TestRun:
    async def test_run_calls_process_direct(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="result text")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "do task", "do task", "cli", "user1", "cli:user1", None)

        mock_loop.process_direct.assert_called_once()
        args, kwargs = mock_loop.process_direct.call_args
        assert args == ("do task",)
        assert kwargs["session_key"] == "subagent:cli:user1:t1"
        # Origin channel/chat_id flow through so ToolContext-consuming
        # tools (cron, nested spawn) still route deliveries back to the
        # originating conversation.
        assert kwargs["channel"] == "cli"
        assert kwargs["chat_id"] == "user1"

    async def test_run_isolates_child_session_from_parent(self) -> None:
        """Every child gets its own on-disk session derived from the
        parent's session_key and the task_id. Two spawns from the same
        parent must not share a session key — otherwise build_prompt
        would load sibling-subagent history as the child's context.
        """
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "a", "a", "telegram", "chat99", "telegram:chat99", None)
            await mgr._run("t2", "b", "b", "telegram", "chat99", "telegram:chat99", None)

        first_key = mock_loop.process_direct.call_args_list[0].kwargs["session_key"]
        second_key = mock_loop.process_direct.call_args_list[1].kwargs["session_key"]
        assert first_key == "subagent:telegram:chat99:t1"
        assert second_key == "subagent:telegram:chat99:t2"
        assert first_key != second_key

    async def test_run_preserves_origin_channel_and_chat_id(self) -> None:
        """Child inherits parent's channel/chat_id. ToolContext-consuming
        tools (cron scheduling, nested SpawnTool) read those fields for
        delivery routing; if we pass anything else, cron jobs scheduled
        from a subagent land at a dead destination instead of the real
        user conversation.
        """
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "task", "label", "telegram", "chat99", "telegram:chat99", None)

        kwargs = mock_loop.process_direct.call_args.kwargs
        assert kwargs["channel"] == "telegram"
        assert kwargs["chat_id"] == "chat99"

    async def test_run_falls_back_when_parent_session_key_missing(self) -> None:
        """If the parent didn't supply a session_key, derive one from
        channel:chat_id so the child still gets a unique on-disk session.
        """
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "do task", "label", "cli", "user1", None, None)

        kwargs = mock_loop.process_direct.call_args.kwargs
        assert kwargs["session_key"] == "subagent:cli:user1:t1"

    async def test_run_announces_result(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="task completed")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "do task", "label", "cli", "user1", "cli:user1", None)

        bus.publish_inbound.assert_called_once()
        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert msg.channel == "system"
        assert msg.sender_id == "subagent"
        assert msg.session_key_override == "cli:user1"
        assert "task completed" in msg.content
        assert "label" in msg.content

    async def test_run_on_exception_announces_failure(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "do task", "label", "cli", "user1", None, None)

        bus.publish_inbound.assert_called_once()
        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert "failed" in msg.content
        assert "boom" in msg.content

    async def test_run_uses_conversation_factory(self) -> None:
        bus = _make_bus()
        factory_calls: list[int] = []

        def factory() -> MagicMock:
            factory_calls.append(1)
            return _make_conversation()

        mgr = SubagentManager(
            provider=_make_provider(),
            bus=bus,
            conversation_factory=factory,
            max_iterations=5,
        )

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "task", "label", "cli", "user1", None, None)

        assert factory_calls == [1]

    async def test_run_passes_model_to_loop(self) -> None:
        bus = _make_bus()
        mgr = SubagentManager(
            provider=_make_provider(),
            bus=bus,
            conversation_factory=_make_conversation,
            model="claude-opus",
            max_iterations=5,
        )
        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop) as MockLoop:  # noqa: N806
            await mgr._run("t1", "task", "label", "cli", "user1", None, None)

        _, kwargs = MockLoop.call_args
        assert kwargs["model"] == "claude-opus"

    async def test_run_per_spawn_model_overrides_default(self) -> None:
        bus = _make_bus()
        mgr = SubagentManager(
            provider=_make_provider(),
            bus=bus,
            conversation_factory=_make_conversation,
            model="claude-opus",
            max_iterations=5,
        )
        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop) as MockLoop:  # noqa: N806
            await mgr._run(
                "t1", "task", "label", "cli", "user1", None, None, model="claude-haiku-4-5"
            )

        _, kwargs = MockLoop.call_args
        assert kwargs["model"] == "claude-haiku-4-5"

    async def test_run_none_model_falls_back_to_manager_default(self) -> None:
        bus = _make_bus()
        mgr = SubagentManager(
            provider=_make_provider(),
            bus=bus,
            conversation_factory=_make_conversation,
            model="claude-opus",
            max_iterations=5,
        )
        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop) as MockLoop:  # noqa: N806
            await mgr._run("t1", "task", "label", "cli", "user1", None, None, model=None)

        _, kwargs = MockLoop.call_args
        assert kwargs["model"] == "claude-opus"

    async def test_spawn_forwards_model_to_run(self) -> None:
        mgr = _make_manager()
        captured: dict[str, object] = {}

        async def fake_run(*args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        with patch.object(mgr, "_run", new=fake_run):
            await mgr.spawn(task="work", model="claude-haiku-4-5")
            await asyncio.sleep(0.01)

        assert captured.get("model") == "claude-haiku-4-5"

    async def test_run_session_key_none_in_announcement(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)
        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "task", "label", "telegram", "chat99", None, None)

        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert msg.chat_id == "telegram:chat99"
        assert msg.session_key_override is None


# ---------------------------------------------------------------------------
# Turn ancestry propagation (stage 3)
# ---------------------------------------------------------------------------


class TestParentTurnAncestry:
    """``SubagentManager._run`` rebinds parent ``turn.*`` contextvars
    before the child agent loop starts, so the child's
    ``_process_turn_inline`` extends the trace chain instead of
    starting a fresh root.

    Without these tests the propagation chain has no end-to-end gate;
    any future refactor that drops the bind/unbind would silently
    break ``turn.root_id:<uuid>`` queries across subagent boundaries.
    """

    async def test_run_binds_parent_chain_before_loop(self) -> None:
        import structlog
        import structlog.contextvars

        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        captured: dict[str, object] = {}

        async def capture_inside_loop(_task: str, **_kwargs: object) -> str:
            captured.update(structlog.contextvars.get_contextvars())
            return "child done"

        mock_loop = MagicMock()
        mock_loop.process_direct = capture_inside_loop

        structlog.contextvars.clear_contextvars()
        try:
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr._run(
                    "t1",
                    "child task",
                    "label",
                    "cli",
                    "user1",
                    "cli:user1",
                    parent_turn_chain="rootA:parentB",
                    parent_turn_id="parentB",
                )
        finally:
            structlog.contextvars.clear_contextvars()

        assert captured.get("turn.chain") == "rootA:parentB"
        assert captured.get("turn.id") == "parentB"
        assert captured.get("turn.root_id") == "rootA", (
            "root_id must be derived from the first segment of the chain"
        )

    async def test_run_unbinds_after_completion(self) -> None:
        import structlog
        import structlog.contextvars

        bus = _make_bus()
        mgr = _make_manager(bus=bus)
        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        structlog.contextvars.clear_contextvars()
        try:
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr._run(
                    "t1",
                    "task",
                    "label",
                    "cli",
                    "user1",
                    None,
                    parent_turn_chain="root:parent",
                    parent_turn_id="parent",
                )
            after = structlog.contextvars.get_contextvars()
        finally:
            structlog.contextvars.clear_contextvars()

        for key in ("turn.id", "turn.chain", "turn.root_id"):
            assert key not in after, f"{key} leaked out of _run"

    async def test_run_unbinds_on_exception(self) -> None:
        import structlog
        import structlog.contextvars

        bus = _make_bus()
        mgr = _make_manager(bus=bus)
        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(side_effect=RuntimeError("boom"))

        structlog.contextvars.clear_contextvars()
        try:
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr._run(
                    "t1",
                    "task",
                    "label",
                    "cli",
                    "user1",
                    None,
                    parent_turn_chain="root:parent",
                    parent_turn_id="parent",
                )
            after = structlog.contextvars.get_contextvars()
        finally:
            structlog.contextvars.clear_contextvars()

        for key in ("turn.id", "turn.chain", "turn.root_id"):
            assert key not in after, (
                f"{key} leaked out of _run after exception — must unbind in finally"
            )

    async def test_run_binds_root_id_from_chain_even_without_turn_id(self) -> None:
        """Passing ``parent_turn_chain`` alone (without
        ``parent_turn_id``) must still bind ``turn.root_id`` from the
        first segment. Otherwise log lines emitted by the subagent
        before the child mints its own ``turn.id`` would drop out of
        ``turn.root_id`` queries entirely — a silent observability
        hole. Caught by Copilot review on PR #36.
        """
        import structlog
        import structlog.contextvars

        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        captured: dict[str, object] = {}

        async def capture_inside_loop(_task: str, **_kwargs: object) -> str:
            captured.update(structlog.contextvars.get_contextvars())
            return "done"

        mock_loop = MagicMock()
        mock_loop.process_direct = capture_inside_loop

        structlog.contextvars.clear_contextvars()
        try:
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr._run(
                    "t1",
                    "task",
                    "label",
                    "cli",
                    "user1",
                    None,
                    parent_turn_chain="rootA:parentB",
                    parent_turn_id=None,
                )
        finally:
            structlog.contextvars.clear_contextvars()

        assert captured.get("turn.chain") == "rootA:parentB"
        assert captured.get("turn.root_id") == "rootA", (
            "root_id must be derived from the chain whether or not "
            "the caller also passed parent_turn_id"
        )
        assert "turn.id" not in captured, (
            "turn.id must remain unbound when only parent_turn_chain was provided"
        )

    async def test_run_with_no_parent_does_not_bind(self) -> None:
        """Existing call sites that don't pass parent_turn_* must not
        get spurious empty bindings."""
        import structlog
        import structlog.contextvars

        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        captured: dict[str, object] = {}

        async def capture_inside_loop(_task: str, **_kwargs: object) -> str:
            captured.update(structlog.contextvars.get_contextvars())
            return "done"

        mock_loop = MagicMock()
        mock_loop.process_direct = capture_inside_loop

        structlog.contextvars.clear_contextvars()
        try:
            with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
                await mgr._run("t1", "task", "label", "cli", "user1", None)
        finally:
            structlog.contextvars.clear_contextvars()

        for key in ("turn.id", "turn.chain", "turn.root_id"):
            assert key not in captured, f"unexpected binding {key} when no parent passed"

    async def test_spawn_forwards_parent_chain_to_run(self) -> None:
        """``SubagentManager.spawn`` must forward ``parent_turn_*`` to
        ``_run`` (which the spawner's runner adapter calls). This is
        the seam stage-3 stitches together — if it's missing, the
        ``SpawnTool`` reads contextvars for nothing.
        """
        mgr = _make_manager()
        captured: dict[str, object] = {}

        async def fake_run(*args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        with patch.object(mgr, "_run", new=fake_run):
            await mgr.spawn(
                task="work",
                parent_turn_chain="root:parent",
                parent_turn_id="parent",
            )
            await asyncio.sleep(0.01)

        assert captured.get("parent_turn_chain") == "root:parent"
        assert captured.get("parent_turn_id") == "parent"


# ---------------------------------------------------------------------------
# _announce_single
# ---------------------------------------------------------------------------


class TestAnnounceSingle:
    async def test_announce_content_structure(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)
        await mgr._announce_single(
            "label", "the task", "the result", None, "completed", "cli", "u1", "cli:u1"
        )

        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert "label" in msg.content
        assert "the task" in msg.content
        assert "the result" in msg.content
        assert "completed" in msg.content
        assert msg.session_key_override == "cli:u1"

    async def test_announce_chat_id_format(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)
        await mgr._announce_single("l", "t", "r", None, "completed", "slack", "C123", None)

        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert msg.chat_id == "slack:C123"


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatch:
    async def test_batch_announces_only_when_all_complete(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="result")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            # Spawn 3 subagents in same batch
            await mgr.spawn(task="t1", label="a", batch="b1")
            await mgr.spawn(task="t2", label="b", batch="b1")
            await mgr.spawn(task="t3", label="c", batch="b1")

            # Let them all complete
            await asyncio.sleep(0.1)

        # Should be exactly one announcement (the batch), not 3
        assert bus.publish_inbound.call_count == 1
        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert "Batch 'b1' complete" in msg.content
        assert "3 succeeded" in msg.content

    async def test_non_batch_announces_individually(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="result")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr.spawn(task="t1", label="a")
            await mgr.spawn(task="t2", label="b")
            await asyncio.sleep(0.1)

        assert bus.publish_inbound.call_count == 2

    async def test_batch_with_failure(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        call_count = 0

        async def mock_process(task: str, **_kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("boom")
            return "ok"

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(side_effect=mock_process)

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr.spawn(task="t1", label="good1", batch="b2")
            await mgr.spawn(task="t2", label="bad", batch="b2")
            await mgr.spawn(task="t3", label="good2", batch="b2")
            await asyncio.sleep(0.1)

        assert bus.publish_inbound.call_count == 1
        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert "2 succeeded" in msg.content
        assert "1 failed" in msg.content


# ---------------------------------------------------------------------------
# get_running_count
# ---------------------------------------------------------------------------


class TestGetRunningCount:
    def test_zero_initially(self) -> None:
        mgr = _make_manager()
        assert mgr.get_running_count() == 0

    async def test_count_during_run(self) -> None:
        mgr = _make_manager()
        event = asyncio.Event()

        async def slow_run(*args: object, **kwargs: object) -> None:
            await event.wait()

        with patch.object(mgr, "_run", new=slow_run):
            await mgr.spawn(task="slow")
            assert mgr.get_running_count() == 1
            event.set()
            await asyncio.sleep(0.01)

        assert mgr.get_running_count() == 0


# ---------------------------------------------------------------------------
# cancel_by_session
# ---------------------------------------------------------------------------


class TestCancelBySession:
    async def test_cancels_only_matching_session(self) -> None:
        mgr = _make_manager()
        event = asyncio.Event()

        async def slow_run(*args: object, **kwargs: object) -> None:
            await event.wait()

        with patch.object(mgr, "_run", new=slow_run):
            await mgr.spawn(task="a", session_key="sess-a")
            await mgr.spawn(task="b", session_key="sess-b")
            assert mgr.get_running_count() == 2

            cancelled = await mgr.cancel_by_session("sess-a")
            await asyncio.sleep(0.01)

            assert cancelled == 1
            assert mgr.get_running_count() == 1
            # sess-b is still running
            event.set()
            await asyncio.sleep(0.01)

        assert mgr.get_running_count() == 0

    async def test_returns_zero_for_unknown_session(self) -> None:
        mgr = _make_manager()
        event = asyncio.Event()

        async def slow_run(*args: object, **kwargs: object) -> None:
            await event.wait()

        with patch.object(mgr, "_run", new=slow_run):
            await mgr.spawn(task="a", session_key="sess-a")
            cancelled = await mgr.cancel_by_session("nope")
            assert cancelled == 0
            assert mgr.get_running_count() == 1
            event.set()
            await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# AsyncioSpawner concurrency cap
# ---------------------------------------------------------------------------


class TestAsyncioSpawnerConcurrency:
    async def test_uncapped_spawner_runs_all_concurrently(self) -> None:
        """Default (max_concurrent=None) preserves pre-cap behavior —
        every spawned runner starts immediately."""
        from exoclaw_subagent.spawner import AsyncioSpawner

        running = 0
        peak = 0
        gate = asyncio.Event()

        async def runner(**kwargs: object) -> None:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await gate.wait()
            running -= 1

        spawner = AsyncioSpawner(runner)
        handles = []
        for i in range(10):
            h = await spawner.start(
                task_id=f"t{i}",
                task=f"task {i}",
                label=f"t{i}",
                origin_channel="cli",
                origin_chat_id="chat",
                session_key=None,
                batch=None,
                skills=None,
                model=None,
            )
            handles.append(h)

        # Let all tasks enter the runner body.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert peak == 10
        gate.set()
        for h in handles:
            await h.wait()

    async def test_max_concurrent_caps_running_count(self) -> None:
        """With max_concurrent=3, no more than 3 runners execute concurrently
        even when 10 are spawned."""
        from exoclaw_subagent.spawner import AsyncioSpawner

        running = 0
        peak = 0
        gate = asyncio.Event()

        async def runner(**kwargs: object) -> None:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await gate.wait()
            running -= 1

        spawner = AsyncioSpawner(runner, max_concurrent=3)
        handles = []
        for i in range(10):
            h = await spawner.start(
                task_id=f"t{i}",
                task=f"task {i}",
                label=f"t{i}",
                origin_channel="cli",
                origin_chat_id="chat",
                session_key=None,
                batch=None,
                skills=None,
                model=None,
            )
            handles.append(h)

        # Let the first batch enter the runner body — semaphore holds the rest.
        for _ in range(5):
            await asyncio.sleep(0)
        assert peak == 3
        gate.set()
        for h in handles:
            await h.wait()
        assert peak == 3  # capped throughout, not just at start

    async def test_rejects_invalid_max_concurrent(self) -> None:
        """Zero or negative caps are nonsense — reject at construction time
        rather than blocking all subagents forever on Semaphore(0)."""
        import pytest
        from exoclaw_subagent.spawner import AsyncioSpawner

        async def noop(**kwargs: object) -> None:
            return None

        with pytest.raises(ValueError, match=">= 1 or None"):
            AsyncioSpawner(noop, max_concurrent=0)
        with pytest.raises(ValueError, match=">= 1 or None"):
            AsyncioSpawner(noop, max_concurrent=-3)

    async def test_start_returns_immediately_even_when_cap_hit(self) -> None:
        """``start()`` always returns a handle fast. The wait-for-a-slot
        happens inside the task body, not inside ``start()``."""
        from exoclaw_subagent.spawner import AsyncioSpawner

        gate = asyncio.Event()

        async def runner(**kwargs: object) -> None:
            await gate.wait()

        spawner = AsyncioSpawner(runner, max_concurrent=1)
        # First spawn takes the only slot and blocks.
        h1 = await spawner.start(
            task_id="t1",
            task="first",
            label="t1",
            origin_channel="cli",
            origin_chat_id="chat",
            session_key=None,
            batch=None,
            skills=None,
            model=None,
        )
        # Second spawn must still return its handle without blocking.
        h2 = await asyncio.wait_for(
            spawner.start(
                task_id="t2",
                task="second",
                label="t2",
                origin_channel="cli",
                origin_chat_id="chat",
                session_key=None,
                batch=None,
                skills=None,
                model=None,
            ),
            timeout=0.5,
        )
        assert h2.done() is False
        gate.set()
        await h1.wait()
        await h2.wait()
