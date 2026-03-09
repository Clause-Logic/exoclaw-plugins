"""ExoclawNanobot — wires all exoclaw-plugins into a running agent."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from loguru import logger

from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.bus.queue import MessageBus
from exoclaw_channel_cli.channel import CLIChannel
from exoclaw_channel_heartbeat.service import HeartbeatService
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_subagent.manager import SubagentManager
from exoclaw_tools_cron.service import CronService
from exoclaw_tools_cron.tool import CronTool
from exoclaw_tools_message.tool import MessageTool
from exoclaw_tools_mcp.config import MCPServerConfig as MCPConfig
from exoclaw_tools_mcp.tool import connect_mcp_servers
from exoclaw_tools_spawn.tool import SpawnTool
from exoclaw_tools_workspace.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from exoclaw_tools_workspace.shell import ExecTool
from exoclaw_tools_workspace.web import WebFetchTool, WebSearchTool

from exoclaw_nanobot.config.loader import load_config
from exoclaw_nanobot.config.schema import Config


class ExoclawNanobot:
    """A fully wired exoclaw agent ready to run."""

    def __init__(
        self,
        config: Config,
        bus: Any,
        agent_loop: Any,
        cli: Any,
        cron_service: Any,
        heartbeat: Any,
        mcp_stack: AsyncExitStack,
    ) -> None:
        self._config = config
        self._bus = bus
        self._agent_loop = agent_loop
        self._cli = cli
        self._cron_service = cron_service
        self._heartbeat = heartbeat
        self._mcp_stack = mcp_stack

    async def run(self) -> None:
        """Start all background services and run the CLI REPL until exit."""
        tasks: list[asyncio.Task[None]] = []
        try:
            tasks.append(asyncio.create_task(self._cron_service.start()))
            tasks.append(asyncio.create_task(self._heartbeat.start()))
            tasks.append(asyncio.create_task(self._agent_loop.run()))
            await self._cli.start(self._bus)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._mcp_stack.aclose()

    async def stop(self) -> None:
        """Signal the CLI to exit."""
        await self._cli.stop()


async def create(
    config: Config | None = None,
    *,
    config_path: Path | None = None,
) -> ExoclawNanobot:
    """
    Create a fully wired ExoclawNanobot.

    Loads config, builds provider, bus, conversation, all tools (workspace,
    cron, message, spawn, MCP), subagent manager, agent loop, CLI channel,
    and heartbeat service.

    Usage::

        import asyncio
        from exoclaw_nanobot import create

        asyncio.run(main())

        async def main():
            bot = await create()
            await bot.run()
    """
    if config is None:
        config = load_config(config_path)

    workspace = config.workspace_path
    workspace.mkdir(parents=True, exist_ok=True)

    model = config.agents.defaults.model
    prov = config.get_provider(model)
    provider = LiteLLMProvider(
        api_key=prov.api_key or None if prov else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=prov.extra_headers if prov else None,
    )

    bus = MessageBus()

    conversation = DefaultConversation.create(
        workspace=workspace,
        provider=provider,
        model=model,
        memory_window=config.agents.defaults.memory_window,
    )

    # Workspace tools
    allowed_dir = workspace if config.tools.restrict_to_workspace else None
    tools: list[Any] = [
        ReadFileTool(workspace=workspace, allowed_dir=allowed_dir),
        WriteFileTool(workspace=workspace, allowed_dir=allowed_dir),
        EditFileTool(workspace=workspace, allowed_dir=allowed_dir),
        ListDirTool(workspace=workspace, allowed_dir=allowed_dir),
        ExecTool(
            timeout=config.tools.exec.timeout,
            working_dir=str(workspace),
            restrict_to_workspace=config.tools.restrict_to_workspace,
            path_append=config.tools.exec.path_append,
        ),
        WebSearchTool(
            api_key=config.tools.web.search.api_key,
            max_results=config.tools.web.search.max_results,
            proxy=config.tools.web.proxy,
        ),
        WebFetchTool(proxy=config.tools.web.proxy),
    ]

    # Cron
    cron_service = CronService(store_path=workspace / "cron.json")
    tools.append(CronTool(cron_service=cron_service))

    # Message
    tools.append(
        MessageTool(
            send_callback=bus.publish_outbound,
            suppress_patterns=config.channels.suppress_patterns,
        )
    )

    # Subagent + spawn
    subagent_mgr = SubagentManager(
        provider=provider,
        bus=bus,
        conversation_factory=lambda: DefaultConversation.create(
            workspace=workspace,
            provider=provider,
            model=model,
        ),
        model=config.agents.defaults.search_model or model,
        max_iterations=config.agents.defaults.max_tool_iterations,
    )
    tools.append(SpawnTool(manager=subagent_mgr))

    # MCP servers
    mcp_stack = AsyncExitStack()
    if config.tools.mcp_servers:
        mcp_cfgs = {
            name: MCPConfig(
                type=srv.type,
                command=srv.command or None,
                args=list(srv.args),
                env=dict(srv.env) or None,
                url=srv.url or None,
                headers=dict(srv.headers) or None,
                tool_timeout=srv.tool_timeout,
            )
            for name, srv in config.tools.mcp_servers.items()
        }
        mcp_registry = ToolRegistry()
        await connect_mcp_servers(mcp_cfgs, mcp_registry, mcp_stack)
        tools.extend(mcp_registry._tools.values())
        logger.info("MCP: {} tools registered", len(mcp_registry._tools))

    # Agent loop
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=conversation,
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        tools=tools,
    )

    # CLI channel
    cli = CLIChannel(history_dir=workspace / "history")

    # Heartbeat
    heartbeat = HeartbeatService(
        workspace=workspace,
        provider=provider,
        model=model,
        on_execute=lambda task: agent_loop.process_direct(task),
        interval_s=config.gateway.heartbeat.interval_s,
        enabled=config.gateway.heartbeat.enabled,
    )

    return ExoclawNanobot(
        config=config,
        bus=bus,
        agent_loop=agent_loop,
        cli=cli,
        cron_service=cron_service,
        heartbeat=heartbeat,
        mcp_stack=mcp_stack,
    )
