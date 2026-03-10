"""Tests for exoclaw_nanobot.app."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw_nanobot.app import ExoclawNanobot, create
from exoclaw_nanobot.config.schema import Config
from exoclaw_tools_cron.types import CronJob, CronJobState, CronPayload, CronSchedule


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


class TestDispatchOutbound:
    def _make_app_with_channel(self, channel_name: str = "telegram") -> tuple[Any, Any, Any]:
        from exoclaw.bus.queue import MessageBus
        bus = MessageBus()
        fake_channel = MagicMock()
        fake_channel.name = channel_name
        fake_channel.send = AsyncMock()

        config = Config()
        agent_loop = MagicMock()
        agent_loop.run = AsyncMock()
        cron_service = MagicMock()
        cron_service.start = AsyncMock()
        heartbeat = MagicMock()
        heartbeat.start = AsyncMock()

        from contextlib import AsyncExitStack
        app = ExoclawNanobot(
            config=config,
            bus=bus,
            agent_loop=agent_loop,
            cli=None,
            cron_service=cron_service,
            heartbeat=heartbeat,
            mcp_stack=AsyncExitStack(),
            extra_channels=[fake_channel],
        )
        return app, bus, fake_channel

    async def test_regular_message_forwarded_to_channel(self) -> None:
        from exoclaw.bus.events import OutboundMessage
        app, bus, channel = self._make_app_with_channel()

        await bus.publish_outbound(OutboundMessage(
            channel="telegram", chat_id="123", content="Hello!",
        ))

        task = asyncio.create_task(app._dispatch_outbound())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        channel.send.assert_called_once()

    async def test_tool_hint_not_forwarded_to_channel(self) -> None:
        """Tool-hint progress messages must NOT be sent to channels like Telegram.

        When the agent calls read_file("...") or web_search("..."), it publishes
        a progress message with _tool_hint=True. These are internal status updates
        and should be suppressed before reaching the user's channel.
        """
        from exoclaw.bus.events import OutboundMessage
        app, bus, channel = self._make_app_with_channel()

        await bus.publish_outbound(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content='read_file("foo.txt")',
            metadata={"_progress": True, "_tool_hint": True},
        ))

        task = asyncio.create_task(app._dispatch_outbound())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        channel.send.assert_not_called()


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

    def _make_patches(self, fake_bus: Any, fake_loop: Any) -> list[Any]:
        """Build the standard patch list, injecting provided bus and loop mocks."""
        return [
            patch("exoclaw_nanobot.app.LiteLLMProvider", MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x")))),
            patch("exoclaw_nanobot.app.MessageBus", MagicMock(return_value=fake_bus)),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch("exoclaw_nanobot.app.AgentLoop", MagicMock(return_value=fake_loop)),
            patch("exoclaw_nanobot.app.CLIChannel", MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()))),
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

    async def _get_on_cron_job(self, fake_bus: Any, fake_loop: Any) -> Any:
        """Create the app and return the on_job callback wired to cron_service."""
        # Use a real CronService (not mocked) so we can capture the on_job assignment
        from exoclaw_tools_cron.service import CronService as RealCronService

        captured: dict[str, Any] = {}
        original_init = RealCronService.__init__

        class CapturingCronService(RealCronService):
            pass

        fake_cron = CapturingCronService.__new__(CapturingCronService)
        fake_cron._store = None
        fake_cron._last_mtime = 0.0
        fake_cron._timer_task = None
        fake_cron._running = False

        original_set = object.__setattr__

        class CronCapture:
            """Minimal stand-in that captures .on_job assignment."""
            def __setattr__(self, name: str, value: Any) -> None:
                if name == "on_job":
                    captured["on_job"] = value
                object.__setattr__(self, name, value)

        cron_capture = CronCapture()

        patches = self._make_patches(fake_bus, fake_loop)
        patches.append(patch("exoclaw_nanobot.app.CronService", MagicMock(return_value=cron_capture)))

        for p in patches:
            p.start()
        try:
            await create(Config())
        finally:
            for p in patches:
                p.stop()

        return captured.get("on_job")

    async def test_cron_job_calls_process_direct_not_bus(self) -> None:
        """Cron jobs must call process_direct() directly — NOT publish to the bus.

        Publishing to the bus causes tool-hint progress messages (read_file("..."),
        web_search("..."), etc.) to be routed back to the user's channel (e.g. Telegram)
        mid-run. Using process_direct() with on_progress=None suppresses all hints
        and only delivers the final response when deliver=True.
        """
        fake_bus = MagicMock()
        fake_bus.publish_inbound = AsyncMock()
        fake_bus.publish_outbound = AsyncMock()

        fake_loop = MagicMock()
        fake_loop.run = AsyncMock()
        fake_loop.process_direct = AsyncMock(return_value="Task complete.")

        on_job = await self._get_on_cron_job(fake_bus, fake_loop)
        assert on_job is not None, "on_job was never assigned to cron_service"

        job = CronJob(
            id="abc",
            name="test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            payload=CronPayload(
                message="summarise my emails",
                channel="telegram",
                to="123456",
                deliver=False,
            ),
            state=CronJobState(),
        )

        await on_job(job)

        # Must use process_direct, not the bus
        fake_loop.process_direct.assert_called_once()
        fake_bus.publish_inbound.assert_not_called()

    async def test_cron_job_deliver_true_publishes_final_response(self) -> None:
        """When deliver=True, the final response must be published as an outbound message."""
        fake_bus = MagicMock()
        fake_bus.publish_inbound = AsyncMock()
        fake_bus.publish_outbound = AsyncMock()

        fake_loop = MagicMock()
        fake_loop.run = AsyncMock()
        fake_loop.process_direct = AsyncMock(return_value="Here is your summary.")

        on_job = await self._get_on_cron_job(fake_bus, fake_loop)
        assert on_job is not None

        job = CronJob(
            id="xyz",
            name="deliver-test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            payload=CronPayload(
                message="morning digest",
                channel="telegram",
                to="999",
                deliver=True,
            ),
            state=CronJobState(),
        )

        await on_job(job)

        fake_bus.publish_outbound.assert_called_once()
        call_args = fake_bus.publish_outbound.call_args
        outbound = call_args[0][0]
        assert outbound.channel == "telegram"
        assert outbound.chat_id == "999"
        assert "summary" in outbound.content

    async def test_cron_job_deliver_false_no_outbound(self) -> None:
        """When deliver=False, no outbound message is published after the run."""
        fake_bus = MagicMock()
        fake_bus.publish_inbound = AsyncMock()
        fake_bus.publish_outbound = AsyncMock()

        fake_loop = MagicMock()
        fake_loop.run = AsyncMock()
        fake_loop.process_direct = AsyncMock(return_value="silent result")

        on_job = await self._get_on_cron_job(fake_bus, fake_loop)
        assert on_job is not None

        job = CronJob(
            id="silent",
            name="silent-job",
            schedule=CronSchedule(kind="every", every_ms=60000),
            payload=CronPayload(
                message="background task",
                channel="telegram",
                to="999",
                deliver=False,
            ),
            state=CronJobState(),
        )

        await on_job(job)

        fake_bus.publish_outbound.assert_not_called()

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
