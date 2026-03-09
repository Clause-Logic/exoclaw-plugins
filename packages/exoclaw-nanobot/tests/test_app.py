"""Tests for exoclaw_nanobot.app."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw_nanobot.app import ExoclawNanobot, create
from exoclaw_nanobot.config.schema import Config


def _make_app(
    run_cli: bool = False,
) -> ExoclawNanobot:
    config = Config()
    bus = MagicMock()
    agent_loop = MagicMock()
    agent_loop.run = AsyncMock()
    cli = MagicMock()
    cli.start = AsyncMock()
    cli.stop = AsyncMock()
    cron_service = MagicMock()
    cron_service.start = AsyncMock()
    heartbeat = MagicMock()
    heartbeat.start = AsyncMock()
    mcp_stack = AsyncExitStack()

    return ExoclawNanobot(
        config=config,
        bus=bus,
        agent_loop=agent_loop,
        cli=cli,
        cron_service=cron_service,
        heartbeat=heartbeat,
        mcp_stack=mcp_stack,
    )


class TestExoclawNanobotRun:
    async def test_starts_background_tasks_and_cli(self) -> None:
        app = _make_app()
        await app.run()

        app._cron_service.start.assert_called_once()
        app._heartbeat.start.assert_called_once()
        app._agent_loop.run.assert_called_once()
        app._cli.start.assert_called_once_with(app._bus)

    async def test_cancels_tasks_on_exit(self) -> None:
        app = _make_app()
        cancelled: list[bool] = []

        async def slow_cron() -> None:
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        app._cron_service.start = slow_cron

        async def _cli_start(_bus: object) -> None:
            await asyncio.sleep(0)  # yield so background tasks actually start

        app._cli.start = _cli_start

        await app.run()
        assert len(cancelled) == 1

    async def test_stop_delegates_to_cli(self) -> None:
        app = _make_app()
        await app.stop()
        app._cli.stop.assert_called_once()

    async def test_mcp_stack_closed_on_exit(self) -> None:
        app = _make_app()
        closed = []
        original_aclose = app._mcp_stack.aclose

        async def _track_close() -> None:
            closed.append(True)
            await original_aclose()

        app._mcp_stack.aclose = _track_close  # type: ignore[method-assign]
        await app.run()
        assert closed == [True]


class TestCreate:
    """Test the create() factory with mocked dependencies."""

    def _patch_all(self) -> dict[str, MagicMock]:
        mocks: dict[str, MagicMock] = {}

        fake_provider = MagicMock()
        fake_provider.get_default_model = MagicMock(return_value="anthropic/claude-opus-4-5")

        fake_conversation = MagicMock()
        fake_conversation.create = MagicMock(return_value=MagicMock())

        fake_bus = MagicMock()
        fake_bus.publish_outbound = AsyncMock()

        fake_loop = MagicMock()
        fake_loop.run = AsyncMock()
        fake_loop.process_direct = AsyncMock(return_value="result")

        fake_cli = MagicMock()
        fake_cli.start = AsyncMock()
        fake_cli.stop = AsyncMock()

        fake_cron = MagicMock()
        fake_cron.start = AsyncMock()

        fake_heartbeat = MagicMock()
        fake_heartbeat.start = AsyncMock()

        mocks["LiteLLMProvider"] = MagicMock(return_value=fake_provider)
        mocks["MessageBus"] = MagicMock(return_value=fake_bus)
        mocks["DefaultConversation"] = MagicMock()
        mocks["DefaultConversation"].create = MagicMock(return_value=MagicMock())
        mocks["AgentLoop"] = MagicMock(return_value=fake_loop)
        mocks["CLIChannel"] = MagicMock(return_value=fake_cli)
        mocks["CronService"] = MagicMock(return_value=fake_cron)
        mocks["CronTool"] = MagicMock(return_value=MagicMock())
        mocks["MessageTool"] = MagicMock(return_value=MagicMock())
        mocks["SpawnTool"] = MagicMock(return_value=MagicMock())
        mocks["SubagentManager"] = MagicMock(return_value=MagicMock())
        mocks["HeartbeatService"] = MagicMock(return_value=fake_heartbeat)
        mocks["ReadFileTool"] = MagicMock(return_value=MagicMock())
        mocks["WriteFileTool"] = MagicMock(return_value=MagicMock())
        mocks["EditFileTool"] = MagicMock(return_value=MagicMock())
        mocks["ListDirTool"] = MagicMock(return_value=MagicMock())
        mocks["ExecTool"] = MagicMock(return_value=MagicMock())
        mocks["WebSearchTool"] = MagicMock(return_value=MagicMock())
        mocks["WebFetchTool"] = MagicMock(return_value=MagicMock())
        mocks["connect_mcp_servers"] = AsyncMock()

        return mocks

    async def test_create_returns_app(self, tmp_path: object) -> None:
        config = Config()

        patches = [
            patch("exoclaw_nanobot.app.LiteLLMProvider", MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x")))),
            patch("exoclaw_nanobot.app.MessageBus", MagicMock(return_value=MagicMock(publish_outbound=AsyncMock()))),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch("exoclaw_nanobot.app.AgentLoop", MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock()))),
            patch("exoclaw_nanobot.app.CLIChannel", MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()))),
            patch("exoclaw_nanobot.app.CronService", MagicMock(return_value=MagicMock(start=AsyncMock()))),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.HeartbeatService", MagicMock(return_value=MagicMock(start=AsyncMock()))),
            patch("exoclaw_nanobot.app.ReadFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WriteFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.EditFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.ListDirTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.ExecTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WebSearchTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WebFetchTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.connect_mcp_servers", AsyncMock()),
        ]

        for p in patches:
            p.start()

        try:
            app = await create(config)
            assert isinstance(app, ExoclawNanobot)
        finally:
            for p in patches:
                p.stop()

    async def test_create_with_mcp_servers(self) -> None:
        config = Config()
        config.tools.mcp_servers = {
            "test": config.tools.mcp_servers.__class__.__class__  # type: ignore[dict-item]
        }
        # Use a fresh config with mcp server configured via dict
        import json as _json
        from exoclaw_nanobot.config.schema import MCPServerConfig

        config2 = Config()
        config2.tools.mcp_servers["mysrv"] = MCPServerConfig(command="npx", args=["-y", "srv"])

        fake_tool = MagicMock()
        fake_tool.name = "mcp_mysrv_tool1"

        patches: list[Any] = [
            patch("exoclaw_nanobot.app.LiteLLMProvider", MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x")))),
            patch("exoclaw_nanobot.app.MessageBus", MagicMock(return_value=MagicMock(publish_outbound=AsyncMock()))),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch("exoclaw_nanobot.app.AgentLoop", MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock()))),
            patch("exoclaw_nanobot.app.CLIChannel", MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()))),
            patch("exoclaw_nanobot.app.CronService", MagicMock(return_value=MagicMock(start=AsyncMock()))),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.HeartbeatService", MagicMock(return_value=MagicMock(start=AsyncMock()))),
            patch("exoclaw_nanobot.app.ReadFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WriteFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.EditFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.ListDirTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.ExecTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WebSearchTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WebFetchTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.connect_mcp_servers", AsyncMock()),
            patch("exoclaw.agent.tools.registry.ToolRegistry._tools", {"mcp_mysrv_tool1": fake_tool}, create=True),
        ]

        for p in patches:
            p.start()

        try:
            app = await create(config2)
            assert isinstance(app, ExoclawNanobot)
        finally:
            for p in patches:
                p.stop()

    async def test_create_loads_config_from_path(self, tmp_path: object) -> None:
        import json as _json
        from pathlib import Path as _Path

        assert isinstance(tmp_path, _Path)
        p = tmp_path / "config.json"
        p.write_text(_json.dumps({"agents": {"defaults": {"maxTokens": 1111}}}))

        patches = [
            patch("exoclaw_nanobot.app.LiteLLMProvider", MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x")))),
            patch("exoclaw_nanobot.app.MessageBus", MagicMock(return_value=MagicMock(publish_outbound=AsyncMock()))),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch("exoclaw_nanobot.app.AgentLoop", MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock()))),
            patch("exoclaw_nanobot.app.CLIChannel", MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()))),
            patch("exoclaw_nanobot.app.CronService", MagicMock(return_value=MagicMock(start=AsyncMock()))),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.HeartbeatService", MagicMock(return_value=MagicMock(start=AsyncMock()))),
            patch("exoclaw_nanobot.app.ReadFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WriteFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.EditFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.ListDirTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.ExecTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WebSearchTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WebFetchTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.connect_mcp_servers", AsyncMock()),
        ]

        for p_obj in patches:
            p_obj.start()

        try:
            app = await create(config_path=p)
            assert isinstance(app, ExoclawNanobot)
            assert app._config.agents.defaults.max_tokens == 1111
        finally:
            for p_obj in patches:
                p_obj.stop()
