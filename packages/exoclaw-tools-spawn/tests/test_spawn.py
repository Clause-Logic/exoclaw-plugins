"""Tests for exoclaw-tools-spawn package."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from exoclaw_tools_spawn.tool import SpawnManager, SpawnTool


# ---------------------------------------------------------------------------
# SpawnManager protocol
# ---------------------------------------------------------------------------


class TestSpawnManagerProtocol:
    def test_concrete_class_satisfies_protocol(self) -> None:
        class MyManager:
            async def spawn(
                self,
                task: str,
                label: str | None = None,
                origin_channel: str = "cli",
                origin_chat_id: str = "direct",
                session_key: str | None = None,
                search: bool = False,
            ) -> str:
                return "done"

        assert isinstance(MyManager(), SpawnManager)

    def test_missing_spawn_fails_protocol(self) -> None:
        class BadManager:
            pass

        assert not isinstance(BadManager(), SpawnManager)


# ---------------------------------------------------------------------------
# SpawnTool
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> AsyncMock:
    m = AsyncMock(spec=SpawnManager)
    m.spawn = AsyncMock(return_value="Subagent [test] started (id: abc12345).")
    return m


@pytest.fixture
def tool(manager: AsyncMock) -> SpawnTool:
    t = SpawnTool(manager=manager)
    t.set_context("cli", "user1", session_key="cli:user1")
    return t


class TestSpawnToolProperties:
    def test_name(self, tool: SpawnTool) -> None:
        assert tool.name == "spawn"

    def test_description(self, tool: SpawnTool) -> None:
        assert "subagent" in tool.description.lower()

    def test_parameters_schema(self, tool: SpawnTool) -> None:
        p = tool.parameters
        assert p["type"] == "object"
        assert "task" in p["properties"]
        assert p["required"] == ["task"]


class TestSpawnToolSetContext:
    def test_set_context_explicit_session_key(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager)
        t.set_context("telegram", "chat123", session_key="tg:chat123")
        assert t._origin_channel == "telegram"
        assert t._origin_chat_id == "chat123"
        assert t._session_key == "tg:chat123"

    def test_set_context_auto_session_key(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager)
        t.set_context("discord", "guild:chan")
        assert t._session_key == "discord:guild:chan"

    def test_defaults(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager)
        assert t._origin_channel == "cli"
        assert t._origin_chat_id == "direct"
        assert t._session_key == "cli:direct"


class TestSpawnToolExecute:
    async def test_basic_spawn(self, tool: SpawnTool, manager: AsyncMock) -> None:
        result = await tool.execute(task="do something")
        assert "started" in result
        manager.spawn.assert_called_once_with(
            task="do something",
            label=None,
            origin_channel="cli",
            origin_chat_id="user1",
            session_key="cli:user1",
            search=False,
        )

    async def test_spawn_with_label(self, tool: SpawnTool, manager: AsyncMock) -> None:
        await tool.execute(task="do something", label="my task")
        manager.spawn.assert_called_once_with(
            task="do something",
            label="my task",
            origin_channel="cli",
            origin_chat_id="user1",
            session_key="cli:user1",
            search=False,
        )

    async def test_spawn_with_search(self, tool: SpawnTool, manager: AsyncMock) -> None:
        await tool.execute(task="research topic", search=True)
        call_kwargs = manager.spawn.call_args.kwargs
        assert call_kwargs["search"] is True

    async def test_spawn_returns_manager_response(self, tool: SpawnTool, manager: AsyncMock) -> None:
        manager.spawn.return_value = "Subagent [my task] started (id: xyz)."
        result = await tool.execute(task="do it")
        assert result == "Subagent [my task] started (id: xyz)."

    async def test_kwargs_ignored(self, tool: SpawnTool, manager: AsyncMock) -> None:
        result = await tool.execute(task="do it", unknown_param="ignored")
        assert "started" in result
