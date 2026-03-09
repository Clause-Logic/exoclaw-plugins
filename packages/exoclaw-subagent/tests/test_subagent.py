"""Tests for exoclaw-subagent package."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        from exoclaw_tools_spawn.tool import SpawnManager
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
        assert "do something" in result or "do somethin" in result  # truncated at 30

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
            await mgr._run("t1", "do task", "do task", "cli", "user1", "cli:user1")

        mock_loop.process_direct.assert_called_once_with("do task")

    async def test_run_announces_result(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)

        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="task completed")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "do task", "label", "cli", "user1", "cli:user1")

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
            await mgr._run("t1", "do task", "label", "cli", "user1", None)

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
            await mgr._run("t1", "task", "label", "cli", "user1", None)

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

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop) as MockLoop:
            await mgr._run("t1", "task", "label", "cli", "user1", None)

        _, kwargs = MockLoop.call_args
        assert kwargs["model"] == "claude-opus"

    async def test_run_session_key_none_in_announcement(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)
        mock_loop = MagicMock()
        mock_loop.process_direct = AsyncMock(return_value="done")

        with patch("exoclaw_subagent.manager.AgentLoop", return_value=mock_loop):
            await mgr._run("t1", "task", "label", "telegram", "chat99", None)

        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert msg.chat_id == "telegram:chat99"
        assert msg.session_key_override is None


# ---------------------------------------------------------------------------
# _announce
# ---------------------------------------------------------------------------


class TestAnnounce:
    async def test_announce_content_structure(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)
        await mgr._announce("label", "the task", "the result", "completed", "cli", "u1", "cli:u1")

        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert "label" in msg.content
        assert "the task" in msg.content
        assert "the result" in msg.content
        assert "completed" in msg.content
        assert msg.session_key_override == "cli:u1"

    async def test_announce_chat_id_format(self) -> None:
        bus = _make_bus()
        mgr = _make_manager(bus=bus)
        await mgr._announce("l", "t", "r", "completed", "slack", "C123", None)

        msg: InboundMessage = bus.publish_inbound.call_args[0][0]
        assert msg.chat_id == "slack:C123"


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
