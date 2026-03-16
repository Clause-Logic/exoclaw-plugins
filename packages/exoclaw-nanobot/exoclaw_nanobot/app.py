"""ExoclawNanobot — wires all exoclaw-plugins into a running agent."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.bus.events import OutboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw_channel_cli.channel import CLIChannel
from exoclaw_channel_heartbeat.service import HeartbeatService
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_subagent.manager import SubagentManager
from exoclaw_tools_cron.service import CronService
from exoclaw_tools_cron.tool import CronTool
from exoclaw_tools_cron.types import CronJob
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
        extra_channels: list[Any] | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._agent_loop = agent_loop
        self._cli = cli
        self._cron_service = cron_service
        self._heartbeat = heartbeat
        self._mcp_stack = mcp_stack
        self._extra_channels: list[Any] = extra_channels or []
        self._stop_event: asyncio.Event = asyncio.Event()

    async def run(self) -> None:
        """Start all background services and channels, then run until stopped.

        If a CLI channel is configured it drives the lifetime (interactive mode).
        If only extra_channels are present (gateway mode) the process runs until
        the OS delivers SIGINT/SIGTERM or :meth:`stop` is called.
        """
        tasks: list[asyncio.Task[None]] = []
        channel_tasks: list[asyncio.Task[None]] = []
        try:
            tasks.append(asyncio.create_task(self._cron_service.start()))
            tasks.append(asyncio.create_task(self._heartbeat.start()))
            tasks.append(asyncio.create_task(self._agent_loop.run()))
            if self._extra_channels:
                tasks.append(asyncio.create_task(self._dispatch_outbound()))
            for ch in self._extra_channels:
                t = asyncio.create_task(ch.start(self._bus))
                tasks.append(t)
                channel_tasks.append(t)

            if self._cli is not None:
                # Interactive: block until the user exits the REPL.
                await self._cli.start(self._bus)
            else:
                # Gateway: block until stop() is called or a channel dies.
                # Only watch channel tasks — infrastructure tasks (cron, heartbeat,
                # agent_loop) may complete normally and must not trigger shutdown.
                watch = [asyncio.create_task(self._stop_event.wait()), *channel_tasks]
                done, _ = await asyncio.wait(
                    watch,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in done:
                    if not t.cancelled():
                        exc = t.exception()
                        if exc is not None:
                            logger.error("Channel task failed, triggering shutdown: {!r}", exc)
        finally:
            for ch in self._extra_channels:
                try:
                    await ch.stop()
                except Exception as e:
                    logger.warning("Error stopping channel {}: {}", ch, e)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._mcp_stack.aclose()

    async def _dispatch_outbound(self) -> None:
        """Route outbound bus messages to the matching extra_channel."""
        channel_map: dict[str, Any] = {ch.name: ch for ch in self._extra_channels}
        while True:
            try:
                msg = await asyncio.wait_for(self._bus.consume_outbound(), timeout=1.0)
                if msg.metadata and msg.metadata.get("_tool_hint"):
                    continue
                ch = channel_map.get(msg.channel)
                if ch is not None:
                    try:
                        await ch.send(msg)
                    except Exception as e:
                        logger.error("Error sending to {}: {}", msg.channel, e)
                else:
                    logger.warning("No channel for outbound message to: {}", msg.channel)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Signal the agent to shut down."""
        self._stop_event.set()
        if self._cli is not None:
            await self._cli.stop()


async def create(
    config: Config | None = None,
    *,
    config_path: Path | None = None,
    extra_channels: list[Any] | None = None,
    extra_tools: list[Any] | None = None,
    enable_cli: bool = True,
    on_pre_context: Callable[[str, str, str, str], Awaitable[str]] | None = None,
    on_pre_tool: Callable[[str, dict[str, Any], str], Awaitable[str | None]] | None = None,
    on_post_turn: Callable[[list[dict[str, Any]], str, str, str], Awaitable[None]] | None = None,
    on_max_iterations: Callable[[str, str, str], Awaitable[None]] | None = None,
) -> ExoclawNanobot:
    """
    Create a fully wired ExoclawNanobot.

    Loads config, builds provider, bus, conversation, all tools (workspace,
    cron, message, spawn, MCP), subagent manager, agent loop, CLI channel,
    and heartbeat service.

    Args:
        extra_channels: Additional Channel implementations started alongside the
            agent (e.g. Telegram, IPC).  Each must implement
            ``start(bus)``, ``stop()``, and ``send(msg)``.
        enable_cli: Set to ``False`` to skip the interactive CLI (gateway mode).
        on_pre_context: Called before each turn with ``(content, ctx)``; return
            extra markdown to inject into the system prompt, or ``None``.
        on_pre_tool: Called before each tool with ``(tool_name, args, ctx)``;
            return a rejection reason string to block the call, or ``None``.
        on_post_turn: Called after each turn with ``(messages, ctx)``.
        on_max_iterations: Called when the tool-call limit is reached with ``(ctx,)``.

    Usage (gateway mode)::

        import asyncio
        from exoclaw_nanobot import create

        async def main():
            bot = await create(enable_cli=False, extra_channels=[telegram, ipc])
            await bot.run()

        asyncio.run(main())
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
        skill_packages=config.skills.packages or None,
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
        model=model,
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

    if extra_tools:
        tools.extend(extra_tools)

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
        on_pre_context=on_pre_context,
        on_pre_tool=on_pre_tool,
        on_post_turn=on_post_turn,
        on_max_iterations=on_max_iterations,
    )

    # Wire cron jobs to run silently via process_direct.
    # Using process_direct with on_progress=None suppresses all tool-hint progress
    # messages (e.g. read_file("...")) that would otherwise be sent to the user's
    # channel mid-run. The final response is only delivered when deliver=True.
    async def _on_cron_job(job: CronJob) -> str | None:
        if job.payload.kind == "agent_turn":
            channel = job.payload.channel or "cli"
            chat_id = job.payload.to or "direct"
            response = await agent_loop.process_direct(
                job.payload.message,
                session_key=f"cron:{job.id}",
                channel=channel,
                chat_id=chat_id,
                on_progress=None,
            )
            if job.payload.deliver and response:
                await bus.publish_outbound(OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=response,
                ))
        return None

    cron_service.on_job = _on_cron_job

    # CLI channel (optional)
    cli = CLIChannel(history_dir=workspace / "history") if enable_cli else None

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
        extra_channels=extra_channels,
    )
