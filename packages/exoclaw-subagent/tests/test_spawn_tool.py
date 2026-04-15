"""Tests for the SpawnTool + SpawnManager protocol.

Lived in ``exoclaw-tools-spawn`` until the merge into ``exoclaw-subagent``
in 0.9.0. Imports below use the canonical ``exoclaw_subagent`` location;
the legacy ``from exoclaw_tools_spawn.tool import …`` path still works
via a shim for one release cycle and is exercised by a separate test in
the ``exoclaw-tools-spawn`` package.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from exoclaw_subagent import SpawnManager, SpawnTool

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
                batch: str | None = None,
                skills: list[str] | None = None,
                model: str | None = None,
            ) -> str:
                return "done"

            def get_status(self) -> dict:
                return {}

            def list_results(self, limit: int = 20) -> list[dict[str, str]]:
                return []

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
        assert "action" in p["properties"]


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
            batch=None,
            skills=None,
            model=None,
            parent_turn_chain=None,
            parent_turn_id=None,
        )

    async def test_spawn_with_label(self, tool: SpawnTool, manager: AsyncMock) -> None:
        await tool.execute(task="do something", label="my task")
        manager.spawn.assert_called_once_with(
            task="do something",
            label="my task",
            origin_channel="cli",
            origin_chat_id="user1",
            session_key="cli:user1",
            batch=None,
            skills=None,
            model=None,
            parent_turn_chain=None,
            parent_turn_id=None,
        )

    async def test_spawn_inherits_parent_skills(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager)
        t.set_context("cli", "user1", session_key="cli:user1", skills=["research"])
        await t.execute(task="do research")
        manager.spawn.assert_called_once_with(
            task="do research",
            label=None,
            origin_channel="cli",
            origin_chat_id="user1",
            session_key="cli:user1",
            batch=None,
            skills=["research"],
            model=None,
            parent_turn_chain=None,
            parent_turn_id=None,
        )

    async def test_spawn_explicit_skills_override(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager)
        t.set_context("cli", "user1", session_key="cli:user1", skills=["research"])
        await t.execute(task="do other", skills=["other-skill"])
        manager.spawn.assert_called_once_with(
            task="do other",
            label=None,
            origin_channel="cli",
            origin_chat_id="user1",
            session_key="cli:user1",
            batch=None,
            skills=["other-skill"],
            model=None,
            parent_turn_chain=None,
            parent_turn_id=None,
        )

    async def test_spawn_passes_model(self, tool: SpawnTool, manager: AsyncMock) -> None:
        await tool.execute(task="cheap task", model="claude-haiku-4-5")
        _, kwargs = manager.spawn.call_args
        assert kwargs["model"] == "claude-haiku-4-5"

    async def test_spawn_returns_manager_response(
        self, tool: SpawnTool, manager: AsyncMock
    ) -> None:
        manager.spawn.return_value = "Subagent [my task] started (id: xyz)."
        result = await tool.execute(task="do it")
        assert result == "Subagent [my task] started (id: xyz)."

    async def test_kwargs_ignored(self, tool: SpawnTool, manager: AsyncMock) -> None:
        result = await tool.execute(task="do it", unknown_param="ignored")
        assert "started" in result


class TestSpawnToolModelAllowlist:
    def test_schema_has_plain_model_when_no_allowlist(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager)
        model_schema = t.parameters["properties"]["model"]
        assert model_schema["type"] == "string"
        assert "enum" not in model_schema

    def test_schema_advertises_enum_when_allowlist_set(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager, allowed_models=["haiku", "nano"])
        model_schema = t.parameters["properties"]["model"]
        assert model_schema["enum"] == ["haiku", "nano"]
        assert "haiku" in model_schema["description"]
        assert "nano" in model_schema["description"]

    async def test_no_allowlist_accepts_any_model(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager)
        t.set_context("cli", "user1", session_key="cli:user1")
        await t.execute(task="anything", model="some/expensive-model")
        _, kwargs = manager.spawn.call_args
        assert kwargs["model"] == "some/expensive-model"

    async def test_allowlist_accepts_listed_model(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager, allowed_models=["haiku", "nano"])
        t.set_context("cli", "user1", session_key="cli:user1")
        result = await t.execute(task="cheap", model="haiku")
        assert "started" in result
        _, kwargs = manager.spawn.call_args
        assert kwargs["model"] == "haiku"

    async def test_allowlist_rejects_unlisted_model(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager, allowed_models=["haiku", "nano"])
        t.set_context("cli", "user1", session_key="cli:user1")
        result = await t.execute(task="try", model="opus")
        assert "Error" in result
        assert "opus" in result
        assert "haiku" in result  # allowed models listed in error
        manager.spawn.assert_not_called()

    async def test_allowlist_allows_model_none(self, manager: AsyncMock) -> None:
        t = SpawnTool(manager=manager, allowed_models=["haiku"])
        t.set_context("cli", "user1", session_key="cli:user1")
        result = await t.execute(task="default")
        assert "started" in result
        _, kwargs = manager.spawn.call_args
        assert kwargs["model"] is None


# ---------------------------------------------------------------------------
# Parent turn ancestry forwarding (stage 3)
# ---------------------------------------------------------------------------


class TestSpawnToolParentTurnPropagation:
    """``SpawnTool`` reads ``turn.chain`` / ``turn.id`` from structlog
    contextvars at call time and forwards them as ``parent_turn_chain``
    / ``parent_turn_id`` to the manager. The agent loop binds those
    contextvars for the duration of every ``_process_turn_inline``
    call (exoclaw 0.15 stage 1), so when the LLM invokes ``spawn``
    inside its tool dispatch, the parent turn is always on the stack.
    """

    async def test_execute_forwards_bound_chain(self, tool: SpawnTool, manager: AsyncMock) -> None:
        import structlog
        import structlog.contextvars

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            **{"turn.id": "p1", "turn.chain": "root:p1", "turn.root_id": "root"}
        )
        try:
            await tool.execute(task="child task")
        finally:
            structlog.contextvars.clear_contextvars()

        _, kwargs = manager.spawn.call_args
        assert kwargs["parent_turn_chain"] == "root:p1"
        assert kwargs["parent_turn_id"] == "p1"

    async def test_execute_with_no_parent_passes_none(
        self, tool: SpawnTool, manager: AsyncMock
    ) -> None:
        """When no parent turn is bound (spawn called outside an
        agent turn), both fields are ``None`` — manager.spawn should
        not see stale or partial values."""
        import structlog
        import structlog.contextvars

        structlog.contextvars.clear_contextvars()
        await tool.execute(task="child task")

        _, kwargs = manager.spawn.call_args
        assert kwargs["parent_turn_chain"] is None
        assert kwargs["parent_turn_id"] is None

    async def test_execute_with_context_forwards_bound_chain(self, manager: AsyncMock) -> None:
        import structlog
        import structlog.contextvars
        from exoclaw.agent.tools.protocol import ToolContext

        t = SpawnTool(manager=manager)
        ctx = ToolContext(session_key="cli:user1", channel="cli", chat_id="user1")

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            **{"turn.id": "p1", "turn.chain": "root:a:b:p1", "turn.root_id": "root"}
        )
        try:
            await t.execute_with_context(ctx, task="child task")
        finally:
            structlog.contextvars.clear_contextvars()

        _, kwargs = manager.spawn.call_args
        assert kwargs["parent_turn_chain"] == "root:a:b:p1"
        assert kwargs["parent_turn_id"] == "p1"
