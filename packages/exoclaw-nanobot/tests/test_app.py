"""Tests for exoclaw_nanobot.app."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from exoclaw_nanobot.app import (
    ExoclawNanobot,
    _build_configured_channels,
    _build_router,
    create,
)
from exoclaw_nanobot.config.schema import Config, RouterConfig, RouterDeployment
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


class TestBuildRouter:
    def test_returns_none_when_empty(self) -> None:
        """Empty model_list keeps the router off so provider falls back to
        the single-deployment ``litellm.acompletion`` path."""
        config = Config()
        assert config.router.model_list == []
        assert _build_router(config) is None

    def test_builds_router_from_model_list(self) -> None:
        config = Config()
        config.router = RouterConfig(
            model_list=[
                RouterDeployment(
                    model_name="group-a",
                    litellm_params={"model": "openai/gpt-5", "api_key": "k1"},
                ),
                RouterDeployment(
                    model_name="group-a",
                    litellm_params={"model": "groq/gpt-5", "api_key": "k2"},
                ),
            ],
            routing_strategy="simple-shuffle",
            num_retries=3,
            fallbacks=[{"group-a": ["group-b"]}],
        )
        fake_router_cls = MagicMock()
        with patch.dict("sys.modules", {"litellm": MagicMock(Router=fake_router_cls)}):
            result = _build_router(config)
        assert result is fake_router_cls.return_value
        assert fake_router_cls.call_args is not None
        kwargs = fake_router_cls.call_args.kwargs
        assert len(kwargs["model_list"]) == 2
        assert kwargs["model_list"][0]["model_name"] == "group-a"
        assert kwargs["model_list"][0]["litellm_params"]["model"] == "openai/gpt-5"
        assert kwargs["routing_strategy"] == "simple-shuffle"
        assert kwargs["num_retries"] == 3
        assert kwargs["fallbacks"] == [{"group-a": ["group-b"]}]

    def test_optional_fields_omitted_when_unset(self) -> None:
        """Leave ``num_retries`` / ``timeout`` / etc. to litellm defaults
        when the config doesn't override them — don't pass ``None``."""
        config = Config()
        config.router = RouterConfig(
            model_list=[RouterDeployment(model_name="g", litellm_params={"model": "openai/gpt-5"})],
        )
        fake_router_cls = MagicMock()
        with patch.dict("sys.modules", {"litellm": MagicMock(Router=fake_router_cls)}):
            _build_router(config)
        assert fake_router_cls.call_args is not None
        kwargs = fake_router_cls.call_args.kwargs
        assert "num_retries" not in kwargs
        assert "timeout" not in kwargs
        assert "cooldown_time" not in kwargs
        assert "allowed_fails" not in kwargs


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

        await bus.publish_outbound(
            OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="Hello!",
            )
        )

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

        await bus.publish_outbound(
            OutboundMessage(
                channel="telegram",
                chat_id="123",
                content='read_file("foo.txt")',
                metadata={"_progress": True, "_tool_hint": True},
            )
        )

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
            patch(
                "exoclaw_nanobot.app.LiteLLMProvider",
                MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x"))),
            ),
            patch(
                "exoclaw_nanobot.app.MessageBus",
                MagicMock(return_value=MagicMock(publish_outbound=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch(
                "exoclaw_nanobot.app.AgentLoop",
                MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CLIChannel",
                MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CronService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch(
                "exoclaw_nanobot.app.HeartbeatService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
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

    async def test_create_with_caller_supplied_provider(self, tmp_path: Path) -> None:
        """When a host passes ``provider=``, nanobot must use it and
        skip building ``LiteLLMProvider`` entirely — otherwise the
        whole point of the seam (letting hosts swap the LLM client
        without nanobot knowing about every provider) is lost.

        Sets ``workspace`` to ``tmp_path`` so ``create()`` doesn't
        ``mkdir ~/.nanobot/workspace`` and touch the runner's home
        directory (Copilot review on #61)."""
        config = Config()
        config.agents.defaults.workspace = str(tmp_path)

        custom_provider = MagicMock()
        custom_provider.get_default_model = MagicMock(return_value="custom-model")

        litellm_ctor = MagicMock(
            return_value=MagicMock(get_default_model=MagicMock(return_value="x"))
        )

        patches = [
            patch("exoclaw_nanobot.app.LiteLLMProvider", litellm_ctor),
            patch(
                "exoclaw_nanobot.app.MessageBus",
                MagicMock(return_value=MagicMock(publish_outbound=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch(
                "exoclaw_nanobot.app.AgentLoop",
                MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CLIChannel",
                MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CronService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch(
                "exoclaw_nanobot.app.HeartbeatService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
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
            app = await create(config, provider=custom_provider)
            assert isinstance(app, ExoclawNanobot)
            litellm_ctor.assert_not_called()
        finally:
            for p in patches:
                p.stop()

    async def test_create_with_mcp_servers(self) -> None:
        config = Config()
        config.tools.mcp_servers = {  # type: ignore[assignment]
            "test": config.tools.mcp_servers.__class__.__class__
        }
        # Use a fresh config with mcp server configured via dict
        from exoclaw_nanobot.config.schema import MCPServerConfig

        config2 = Config()
        config2.tools.mcp_servers["mysrv"] = MCPServerConfig(command="npx", args=["-y", "srv"])

        fake_tool = MagicMock()
        fake_tool.name = "mcp_mysrv_tool1"

        patches: list[Any] = [
            patch(
                "exoclaw_nanobot.app.LiteLLMProvider",
                MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x"))),
            ),
            patch(
                "exoclaw_nanobot.app.MessageBus",
                MagicMock(return_value=MagicMock(publish_outbound=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch(
                "exoclaw_nanobot.app.AgentLoop",
                MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CLIChannel",
                MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CronService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch(
                "exoclaw_nanobot.app.HeartbeatService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.ReadFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WriteFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.EditFileTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.ListDirTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.ExecTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WebSearchTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.WebFetchTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.connect_mcp_servers", AsyncMock()),
            patch(
                "exoclaw.agent.tools.registry.ToolRegistry._tools",
                {"mcp_mysrv_tool1": fake_tool},
                create=True,
            ),
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
            patch(
                "exoclaw_nanobot.app.LiteLLMProvider",
                MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x"))),
            ),
            patch("exoclaw_nanobot.app.MessageBus", MagicMock(return_value=fake_bus)),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch("exoclaw_nanobot.app.AgentLoop", MagicMock(return_value=fake_loop)),
            patch(
                "exoclaw_nanobot.app.CLIChannel",
                MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch(
                "exoclaw_nanobot.app.HeartbeatService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
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

        class CapturingCronService(RealCronService):
            pass

        fake_cron = CapturingCronService.__new__(CapturingCronService)
        fake_cron._store = None
        fake_cron._last_mtime = 0.0
        fake_cron._timer_task = None
        fake_cron._running = False

        class CronCapture:
            """Minimal stand-in that captures .on_job assignment."""

            def __setattr__(self, name: str, value: Any) -> None:
                if name == "on_job":
                    captured["on_job"] = value
                object.__setattr__(self, name, value)

        cron_capture = CronCapture()

        patches = self._make_patches(fake_bus, fake_loop)
        patches.append(
            patch("exoclaw_nanobot.app.CronService", MagicMock(return_value=cron_capture))
        )

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

    async def test_cron_job_passes_skills_to_process_direct(self) -> None:
        """When a cron job has skills configured, they must be forwarded to the agent.

        Without this, the agent running the cron job doesn't have the skill
        loaded in its context — it just sees the prompt text but has no skill
        instructions, so it behaves completely differently from a manual run.
        """
        fake_bus = MagicMock()
        fake_bus.publish_inbound = AsyncMock()
        fake_bus.publish_outbound = AsyncMock()

        fake_loop = MagicMock()
        fake_loop.run = AsyncMock()
        fake_loop.process_direct = AsyncMock(return_value="Done.")

        on_job = await self._get_on_cron_job(fake_bus, fake_loop)
        assert on_job is not None

        job = CronJob(
            id="skill-test",
            name="note-research",
            schedule=CronSchedule(kind="cron", expr="0 * * * *"),
            payload=CronPayload(
                message="Run note-research skill: pick a topic and create a zettel",
                channel="telegram",
                to="123",
                deliver=True,
                skills=["note-research"],
            ),
            state=CronJobState(),
        )

        await on_job(job)

        fake_loop.process_direct.assert_called_once()
        call_kwargs = fake_loop.process_direct.call_args
        # The skills must be passed through so the agent loop can load them
        # into the system prompt. The exact parameter name may be `skills` or
        # `skill_names` — either way it must appear in the call.
        all_args = {**dict(zip(["content"], call_kwargs.args)), **(call_kwargs.kwargs or {})}
        skills_value = all_args.get("skills") or all_args.get("skill_names")
        assert skills_value == ["note-research"], (
            f"Expected skills=['note-research'] to be passed to process_direct, "
            f"got call args: {call_kwargs}"
        )

    async def test_cron_job_stateless_uses_unique_session_key(self) -> None:
        """When stateless=True, each cron run must use a unique session key.

        Otherwise history accumulates across runs in the same session
        (e.g. 'cron:<job_id>'), causing the agent to see stale context
        from previous executions instead of starting fresh each time.
        """
        fake_bus = MagicMock()
        fake_bus.publish_inbound = AsyncMock()
        fake_bus.publish_outbound = AsyncMock()

        fake_loop = MagicMock()
        fake_loop.run = AsyncMock()
        fake_loop.process_direct = AsyncMock(return_value="Done.")

        on_job = await self._get_on_cron_job(fake_bus, fake_loop)
        assert on_job is not None

        job = CronJob(
            id="stateless-test",
            name="stateless-job",
            schedule=CronSchedule(kind="every", every_ms=3600000),
            payload=CronPayload(
                message="do something fresh",
                channel="telegram",
                to="123",
                deliver=False,
                stateless=True,
            ),
            state=CronJobState(),
        )

        # Run twice
        await on_job(job)
        await on_job(job)

        assert fake_loop.process_direct.call_count == 2

        first_call = fake_loop.process_direct.call_args_list[0]
        second_call = fake_loop.process_direct.call_args_list[1]

        first_session = (
            first_call.kwargs.get("session_key") or first_call.args[1]
            if len(first_call.args) > 1
            else first_call.kwargs.get("session_key")
        )
        second_session = (
            second_call.kwargs.get("session_key") or second_call.args[1]
            if len(second_call.args) > 1
            else second_call.kwargs.get("session_key")
        )

        # Session keys must differ between runs so no history accumulates
        assert first_session != second_session, (
            f"Stateless cron job used the same session key for both runs: {first_session!r}. "
            f"Expected unique keys per run to prevent history accumulation."
        )
        # And neither should be the static 'cron:<id>' key
        assert first_session != "cron:stateless-test", (
            "Stateless cron job used the static session key 'cron:stateless-test' — "
            "this causes history to accumulate across runs."
        )

    async def test_cron_job_model_override_passed_to_process_direct(self) -> None:
        """When payload.model is set, the cron dispatch must forward it to
        process_direct so the override reaches the provider call.
        """
        fake_bus = MagicMock()
        fake_bus.publish_inbound = AsyncMock()
        fake_bus.publish_outbound = AsyncMock()

        fake_loop = MagicMock()
        fake_loop.run = AsyncMock()
        fake_loop.process_direct = AsyncMock(return_value="Done.")

        on_job = await self._get_on_cron_job(fake_bus, fake_loop)
        assert on_job is not None

        job = CronJob(
            id="model-test",
            name="model-job",
            schedule=CronSchedule(kind="every", every_ms=3600000),
            payload=CronPayload(
                message="summarize my day",
                channel="telegram",
                to="123",
                deliver=False,
                model="openrouter/google/gemma-4-26b-a4b-it",
            ),
            state=CronJobState(),
        )

        await on_job(job)

        fake_loop.process_direct.assert_called_once()
        call = fake_loop.process_direct.call_args
        assert call.kwargs.get("model") == "openrouter/google/gemma-4-26b-a4b-it", (
            f"Expected model=openrouter/google/gemma-4-26b-a4b-it to reach process_direct, "
            f"got call args: {call}"
        )

    async def test_cron_job_without_model_passes_none(self) -> None:
        """When payload.model is unset, process_direct receives model=None
        so the loop falls back to its default.
        """
        fake_bus = MagicMock()
        fake_bus.publish_inbound = AsyncMock()
        fake_bus.publish_outbound = AsyncMock()

        fake_loop = MagicMock()
        fake_loop.run = AsyncMock()
        fake_loop.process_direct = AsyncMock(return_value="Done.")

        on_job = await self._get_on_cron_job(fake_bus, fake_loop)
        assert on_job is not None

        job = CronJob(
            id="default-model-test",
            name="default-model-job",
            schedule=CronSchedule(kind="every", every_ms=3600000),
            payload=CronPayload(
                message="use the default",
                channel="telegram",
                to="123",
                deliver=False,
            ),
            state=CronJobState(),
        )

        await on_job(job)

        fake_loop.process_direct.assert_called_once()
        call = fake_loop.process_direct.call_args
        assert call.kwargs.get("model") is None

    async def test_create_loads_config_from_path(self, tmp_path: object) -> None:
        import json as _json
        from pathlib import Path as _Path

        assert isinstance(tmp_path, _Path)
        p = tmp_path / "config.json"
        p.write_text(_json.dumps({"agents": {"defaults": {"maxTokens": 1111}}}))

        patches = [
            patch(
                "exoclaw_nanobot.app.LiteLLMProvider",
                MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x"))),
            ),
            patch(
                "exoclaw_nanobot.app.MessageBus",
                MagicMock(return_value=MagicMock(publish_outbound=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch(
                "exoclaw_nanobot.app.AgentLoop",
                MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CLIChannel",
                MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CronService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch(
                "exoclaw_nanobot.app.HeartbeatService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
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

    async def test_spawn_tool_receives_no_allowlist_by_default(self) -> None:
        config = Config()
        spawn_tool_mock = MagicMock(return_value=MagicMock())

        patches = [
            patch(
                "exoclaw_nanobot.app.LiteLLMProvider",
                MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x"))),
            ),
            patch(
                "exoclaw_nanobot.app.MessageBus",
                MagicMock(return_value=MagicMock(publish_outbound=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch(
                "exoclaw_nanobot.app.AgentLoop",
                MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CLIChannel",
                MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CronService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", spawn_tool_mock),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch(
                "exoclaw_nanobot.app.HeartbeatService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
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
            await create(config)
            spawn_tool_mock.assert_called_once()
            _, kwargs = spawn_tool_mock.call_args
            assert kwargs["allowed_models"] is None
        finally:
            for p_obj in patches:
                p_obj.stop()

    async def test_spawn_tool_receives_allowlist_when_configured(self) -> None:
        config = Config()
        config.agents.subagent_allowed_models = ["claude-haiku-4-5", "gpt-5-nano"]
        spawn_tool_mock = MagicMock(return_value=MagicMock())

        patches = [
            patch(
                "exoclaw_nanobot.app.LiteLLMProvider",
                MagicMock(return_value=MagicMock(get_default_model=MagicMock(return_value="x"))),
            ),
            patch(
                "exoclaw_nanobot.app.MessageBus",
                MagicMock(return_value=MagicMock(publish_outbound=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.DefaultConversation"),
            patch(
                "exoclaw_nanobot.app.AgentLoop",
                MagicMock(return_value=MagicMock(run=AsyncMock(), process_direct=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CLIChannel",
                MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock())),
            ),
            patch(
                "exoclaw_nanobot.app.CronService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
            patch("exoclaw_nanobot.app.CronTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.MessageTool", MagicMock(return_value=MagicMock())),
            patch("exoclaw_nanobot.app.SpawnTool", spawn_tool_mock),
            patch("exoclaw_nanobot.app.SubagentManager", MagicMock(return_value=MagicMock())),
            patch(
                "exoclaw_nanobot.app.HeartbeatService",
                MagicMock(return_value=MagicMock(start=AsyncMock())),
            ),
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
            await create(config)
            spawn_tool_mock.assert_called_once()
            _, kwargs = spawn_tool_mock.call_args
            assert kwargs["allowed_models"] == ["claude-haiku-4-5", "gpt-5-nano"]
        finally:
            for p_obj in patches:
                p_obj.stop()


class TestBuildConfiguredChannels:
    """`_build_configured_channels` reads `config.channels.<name>.enabled`
    and instantiates the matching channel class via lazy import. Each
    optional channel package is opt-in via `[project.optional-dependencies]`,
    so a missing package must produce a clear "install this extra" error
    rather than a cryptic ModuleNotFoundError mid-startup."""

    def test_returns_empty_when_nothing_enabled(self) -> None:
        """Default config has every optional channel disabled — returns
        an empty list so callers can pass it through unchanged."""
        config = Config()
        assert _build_configured_channels(config) == []

    def test_builds_enabled_channel_with_section_config(self) -> None:
        config = Config()
        config.channels.slack.enabled = True
        config.channels.slack.bot_token = "xoxb-test"
        config.channels.slack.allow_from = ["U123"]

        fake_cls = MagicMock()
        fake_module = MagicMock(SlackChannel=fake_cls)
        with patch.dict("sys.modules", {"exoclaw_channel_slack.channel": fake_module}):
            result = _build_configured_channels(config)

        assert len(result) == 1
        fake_cls.assert_called_once()
        passed_config = fake_cls.call_args.kwargs["config"]
        assert passed_config["enabled"] is True
        assert passed_config["bot_token"] == "xoxb-test"
        assert passed_config["allow_from"] == ["U123"]

    def test_missing_package_raises_pointing_to_extra(self) -> None:
        """Enabled channel + uninstalled package = RuntimeError that names
        the right `pip install 'exoclaw-nanobot[<extra>]'` invocation. The
        original ModuleNotFoundError chains as __cause__ for debuggability."""
        config = Config()
        config.channels.telegram.enabled = True

        # Simulate the real failure mode: the channel package itself isn't
        # installed, so `import exoclaw_channel_telegram.channel` raises
        # with `e.name == "exoclaw_channel_telegram"` (the topmost missing
        # parent).
        err = ModuleNotFoundError("no module")
        err.name = "exoclaw_channel_telegram"
        with patch("importlib.import_module", side_effect=err):
            with pytest.raises(RuntimeError) as exc_info:
                _build_configured_channels(config)

        msg = str(exc_info.value)
        assert "channels.telegram.enabled is true" in msg
        assert "pip install 'exoclaw-nanobot[telegram]'" in msg
        assert exc_info.value.__cause__ is err

    def test_transitive_dep_missing_propagates_original_error(self) -> None:
        """If the channel package IS installed but one of its own deps
        isn't (e.g. ``slack_sdk`` failed to install), the original
        ModuleNotFoundError must propagate — re-framing it as "install the
        slack extra" would mislead the user into reinstalling something
        that's already there. Only re-frame when the missing module is
        the channel package itself."""
        config = Config()
        config.channels.slack.enabled = True

        err = ModuleNotFoundError("No module named 'slack_sdk'")
        err.name = "slack_sdk"
        with patch("importlib.import_module", side_effect=err):
            with pytest.raises(ModuleNotFoundError) as exc_info:
                _build_configured_channels(config)

        assert exc_info.value is err

    def test_builds_multiple_channels_in_declared_order(self) -> None:
        """When several channels are enabled, the order matches the
        `_OPTIONAL_CHANNELS` declaration so outbound routing and startup
        logs are deterministic across runs."""
        config = Config()
        config.channels.slack.enabled = True
        config.channels.discord.enabled = True
        config.channels.matrix.enabled = True

        fake_slack = MagicMock(name="SlackChannel")
        fake_discord = MagicMock(name="DiscordChannel")
        fake_matrix = MagicMock(name="MatrixChannel")
        modules = {
            "exoclaw_channel_slack.channel": MagicMock(SlackChannel=fake_slack),
            "exoclaw_channel_discord.channel": MagicMock(DiscordChannel=fake_discord),
            "exoclaw_channel_matrix.channel": MagicMock(MatrixChannel=fake_matrix),
        }
        with patch.dict("sys.modules", modules):
            result = _build_configured_channels(config)

        assert len(result) == 3
        # Order matches `_OPTIONAL_CHANNELS` (slack before discord before matrix)
        assert result[0] is fake_slack.return_value
        assert result[1] is fake_discord.return_value
        assert result[2] is fake_matrix.return_value

    def test_disabled_channel_is_skipped_even_if_package_present(self) -> None:
        """`enabled: false` skips construction entirely — we don't even
        try to import the module. This keeps cold-start fast for users
        who installed `[all-channels]` but only enable a few."""
        config = Config()
        config.channels.slack.enabled = False

        with patch("importlib.import_module") as fake_import:
            result = _build_configured_channels(config)

        assert result == []
        fake_import.assert_not_called()
