"""ExoclawNanobot — wires all exoclaw-plugins into a running agent."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable

import structlog
from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.bus.events import OutboundMessage
from exoclaw.bus.queue import MessageBus
from exoclaw.providers.protocol import LLMProvider
from exoclaw.utils import create_isolated_task
from exoclaw_channel_cli.channel import CLIChannel
from exoclaw_channel_heartbeat.service import HeartbeatService
from exoclaw_conversation import LoadSkillTool
from exoclaw_conversation.context import ContextBuilder
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_conversation.memory import MemoryStore
from exoclaw_conversation.session.manager import SessionManager
from exoclaw_conversation.summarizing_policy import SummarizingConsolidationPolicy
from exoclaw_executor_dbos import (
    DBOSBatchStore,
    DBOSExecutor,
    DBOSSubagentSpawner,
    set_loop_context,
)
from exoclaw_loop_detection import LoopDetectionConfig, LoopDetectionPolicy
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_subagent import SpawnTool, SubagentManager
from exoclaw_tools_cron.service import CronService, LocalCronBackend
from exoclaw_tools_cron.tool import CronTool
from exoclaw_tools_cron.types import CronJob
from exoclaw_tools_mcp.config import MCPServerConfig as MCPConfig
from exoclaw_tools_mcp.tool import connect_mcp_servers
from exoclaw_tools_message.tool import MessageTool
from exoclaw_tools_workspace.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from exoclaw_tools_workspace.shell import ExecTool
from exoclaw_tools_workspace.web import WebFetchTool, WebSearchTool

from exoclaw_conversation import LoadSkillTool

from exoclaw_nanobot.config.loader import load_config
from exoclaw_nanobot.config.schema import Config


def _build_router(config: Config) -> Any | None:
    """Construct a ``litellm.Router`` from config, or return ``None`` when
    unconfigured. Kept as a module-level helper so the provider wiring in
    ``create()`` stays a one-liner and so the router construction is
    independently testable without spinning up the whole bot.

    ``litellm`` itself is already loaded at import time via
    ``LiteLLMProvider``; we import it inline here only to keep the symbol
    local to this helper so tests can patch ``sys.modules['litellm']`` in
    isolation. The real skip-when-unconfigured win is that we don't build
    a ``Router`` instance (which instantiates per-deployment clients,
    cooldown trackers, etc.) on bots that don't use one.
    """
    rc = config.router
    if not rc.model_list:
        return None
    import litellm

    kwargs: dict[str, Any] = {
        "model_list": [d.model_dump() for d in rc.model_list],
        "routing_strategy": rc.routing_strategy,
    }
    if rc.fallbacks:
        kwargs["fallbacks"] = rc.fallbacks
    if rc.num_retries is not None:
        kwargs["num_retries"] = rc.num_retries
    if rc.timeout is not None:
        kwargs["timeout"] = rc.timeout
    if rc.cooldown_time is not None:
        kwargs["cooldown_time"] = rc.cooldown_time
    if rc.allowed_fails is not None:
        kwargs["allowed_fails"] = rc.allowed_fails
    router = litellm.Router(**kwargs)
    groups = sorted({d.model_name for d in rc.model_list})
    logger.info(
        "litellm_router_built",
        **{
            "router.deployment.count": len(rc.model_list),
            "router.group.count": len(groups),
            "router.groups": ",".join(groups),
            "router.strategy": rc.routing_strategy,
        },
    )
    return router


logger = structlog.get_logger()


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
            tasks.append(create_isolated_task(self._cron_service.start()))
            tasks.append(create_isolated_task(self._heartbeat.start()))
            tasks.append(create_isolated_task(self._agent_loop.run()))
            if self._extra_channels:
                tasks.append(create_isolated_task(self._dispatch_outbound()))
            for ch in self._extra_channels:
                t = create_isolated_task(ch.start(self._bus))
                tasks.append(t)
                channel_tasks.append(t)

            if self._cli is not None:
                # Interactive: block until the user exits the REPL.
                await self._cli.start(self._bus)
            else:
                # Gateway: block until stop() is called or a channel dies.
                # Only watch channel tasks — infrastructure tasks (cron, heartbeat,
                # agent_loop) may complete normally and must not trigger shutdown.
                watch = [create_isolated_task(self._stop_event.wait()), *channel_tasks]
                done, _ = await asyncio.wait(
                    watch,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in done:
                    if not t.cancelled():
                        exc = t.exception()
                        if exc is not None:
                            logger.error("channel_task_failed", **{"error.repr": repr(exc)})
        finally:
            for ch in self._extra_channels:
                try:
                    await ch.stop()
                except Exception as e:
                    logger.warning(
                        "channel_stop_failed",
                        **{"channel.name": getattr(ch, "name", str(ch))},
                        error=e,
                    )
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
                        logger.error(
                            "outbound_send_failed", **{"channel.name": msg.channel}, error=e
                        )
                else:
                    logger.warning("outbound_no_channel", **{"channel.name": msg.channel})
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
    cli_channel: Any | None = None,
    provider: LLMProvider | None = None,
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
    # Caller-provided provider wins — ``provider=`` was added so hosts
    # that want a different LLM client (e.g. a direct-httpx streaming
    # provider for memory reasons) can wire it themselves without
    # nanobot needing to know about every provider implementation. When
    # ``None``, fall back to the built-in LiteLLM path so existing
    # callers aren't affected.
    if provider is None:
        prov = config.get_provider(model)
        router = _build_router(config)
        provider = LiteLLMProvider(
            api_key=prov.api_key or None if prov else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=prov.extra_headers if prov else None,
            model_max_concurrent={
                name: cfg.max_concurrent for name, cfg in config.agents.models.items()
            },
            model_extra_body={
                name: cfg.extra_body for name, cfg in config.agents.models.items() if cfg.extra_body
            },
            router=router,
        )

    bus = MessageBus()

    # streaming_history=True drops the per-session unconsolidated tail
    # from RAM — the unconsolidated history lives only on disk, read
    # on demand by ``read_history``. Required for multi-tenant openclaw:
    # without it, N concurrent sessions × per-session message-list size
    # blows the cgroup as session length grows. See
    # docs/memory-model.md Step C.
    history_store = SessionManager(workspace, streaming_history=True)
    memory_store = MemoryStore(workspace, provider, model, history=history_store)
    consolidation_policy = SummarizingConsolidationPolicy(memory=memory_store)
    conversation = DefaultConversation(
        history=history_store,
        memory=memory_store,
        prompt=ContextBuilder(
            workspace, memory=memory_store, skill_packages=config.skills.packages or None
        ),
        memory_window=config.agents.defaults.memory_window,
        consolidation_policy=consolidation_policy,
    )

    # Load skill tool — lets the agent activate skills on demand
    prompt = conversation.prompt
    load_skill_tool = LoadSkillTool(
        skills=prompt.skills,  # type: ignore[attr-defined]
        active_tools=prompt._active_optional_tools,  # type: ignore[attr-defined]
    )

    # Workspace tools
    allowed_dir = workspace if config.tools.restrict_to_workspace else None
    tools: list[Any] = [
        load_skill_tool,
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
    tools.append(CronTool(backend=LocalCronBackend(cron_service)))

    # Message
    tools.append(
        MessageTool(
            send_callback=bus.publish_outbound,
            suppress_patterns=config.channels.suppress_patterns,
        )
    )

    # Subagent + spawn
    _skill_pkgs = config.skills.packages or None
    # Pass the tools list by reference — it's mutated in-place below (MCP, extra_tools),
    # so subagents will have the full tool set when they actually run.
    spawner_factory: Any = DBOSSubagentSpawner
    if config.agents.subagent_max_concurrent is not None:
        spawner_factory = partial(
            DBOSSubagentSpawner,
            max_concurrent=config.agents.subagent_max_concurrent,
        )

    # Subagents need their own iteration policy — a fresh one per spawn so
    # the loop-detection history stays isolated from the parent and from
    # siblings. Without this, child loops fall back to the static
    # ``max_iterations`` cap (default 40) even though loop detection is on.
    _ld = config.agents.defaults.loop_detection

    def _build_subagent_policy() -> LoopDetectionPolicy:
        return LoopDetectionPolicy(
            LoopDetectionConfig(
                history_size=_ld.history_size,
                warning_threshold=_ld.warning_threshold,
                critical_threshold=_ld.critical_threshold,
                global_circuit_breaker=_ld.global_circuit_breaker,
                detect_repeat=_ld.detect_repeat,
                detect_ping_pong=_ld.detect_ping_pong,
            )
        )

    subagent_iteration_policy_factory: Callable[[], LoopDetectionPolicy] | None = (
        _build_subagent_policy if _ld.enabled else None
    )

    subagent_mgr = SubagentManager(
        provider=provider,
        bus=bus,
        conversation_factory=lambda: DefaultConversation.create(
            workspace=workspace,
            provider=provider,
            model=model,
            skill_packages=_skill_pkgs,
        ),
        tools=tools,
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        workspace=workspace,
        # Route subagents through DBOS child workflows so concurrent spawns
        # can't race into the parent's step journal (see 2026-04-13 Feed
        # curator incident).
        spawner_factory=spawner_factory,
        # Persist batch lifecycle to disk via DBOS steps so a restart
        # during a multi-subagent batch doesn't orphan completions (see
        # 2026-04-23 feed-digest-retry incident — three recovered
        # subagents completed into an empty in-memory ``_batches`` dict
        # and the batch announcement never fired).
        batch_store=DBOSBatchStore(workspace=workspace),
        iteration_policy_factory=subagent_iteration_policy_factory,
    )
    spawn_tool = SpawnTool(
        manager=subagent_mgr,
        allowed_models=config.agents.subagent_allowed_models or None,
    )
    tools.append(spawn_tool)

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
        logger.info("mcp_tools_registered", **{"tool.count": len(mcp_registry._tools)})

    if extra_tools:
        tools.extend(extra_tools)

    # Iteration policy (loop detection)
    ld = config.agents.defaults.loop_detection
    iteration_policy: LoopDetectionPolicy | None = None
    if ld.enabled:
        iteration_policy = LoopDetectionPolicy(
            LoopDetectionConfig(
                history_size=ld.history_size,
                warning_threshold=ld.warning_threshold,
                critical_threshold=ld.critical_threshold,
                global_circuit_breaker=ld.global_circuit_breaker,
                detect_repeat=ld.detect_repeat,
                detect_ping_pong=ld.detect_ping_pong,
            )
        )

    # Hook: record tool calls into the iteration policy for pattern detection,
    # and reset history at the start of each turn so prior sessions don't bleed.
    on_tool_calls = None
    if iteration_policy is not None:
        _policy = iteration_policy
        _orig_pre_context = on_pre_context

        async def _reset_then_pre_context(
            content: str, session_key: str, channel: str, chat_id: str
        ) -> str:
            _policy.reset()
            if _orig_pre_context:
                return await _orig_pre_context(content, session_key, channel, chat_id)
            return ""

        on_pre_context = _reset_then_pre_context

        async def _record_tool_calls(tool_calls: list[Any]) -> None:
            for tc in tool_calls:
                _policy.record(tc.name, tc.arguments)

        on_tool_calls = _record_tool_calls

    # Context overflow recovery — compact messages when context window is exceeded
    async def _on_context_overflow(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        from exoclaw_conversation.context import drop_oldest_half

        compacted = drop_oldest_half(messages)
        if len(compacted) < len(messages):
            logger.info(
                "context_overflow_compacted",
                **{"message.count.before": len(messages), "message.count.after": len(compacted)},
            )
            return compacted
        return None

    # Durable executor — every LLM call and tool execution is checkpointed
    executor = DBOSExecutor()

    # Agent loop
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=conversation,
        # ty can't structurally prove DBOSExecutor satisfies the
        # ``exoclaw.executor.Executor`` Protocol — every method
        # name and signature matches but the runtime check passes
        # via ``@runtime_checkable``. Cast to silence; tested at
        # runtime end-to-end.
        executor=executor,  # type: ignore[invalid-argument-type]
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        tools=tools,
        iteration_policy=iteration_policy,
        on_pre_context=on_pre_context,
        on_pre_tool=on_pre_tool,
        on_post_turn=on_post_turn,
        on_max_iterations=on_max_iterations,
        on_tool_calls=on_tool_calls,
        on_context_overflow=_on_context_overflow,
    )

    # Wire cron jobs to run silently via process_direct.
    # Using process_direct with on_progress=None suppresses all tool-hint progress
    # messages (e.g. read_file("...")) that would otherwise be sent to the user's
    # channel mid-run. The final response is only delivered when deliver=True.
    async def _on_cron_job(job: CronJob) -> str | None:
        if job.payload.kind == "agent_turn":
            channel = job.payload.channel or "cli"
            chat_id = job.payload.to or "direct"
            if job.payload.stateless:
                sid = f"cron:{job.id}:{uuid.uuid4().hex[:8]}"
            else:
                sid = f"cron:{job.id}"
            cron_skills = job.payload.skills or None
            spawn_tool.set_context(channel, chat_id, session_key=sid, skills=cron_skills)
            try:
                response = await agent_loop.process_direct(
                    job.payload.message,
                    session_key=sid,
                    channel=channel,
                    chat_id=chat_id,
                    on_progress=None,
                    skills=cron_skills,
                    model=job.payload.model,
                )
            finally:
                spawn_tool.set_context(channel, chat_id, session_key=sid, skills=None)
            if job.payload.deliver and response:
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=channel,
                        chat_id=chat_id,
                        content=response,
                    )
                )
        return None

    cron_service.on_job = _on_cron_job

    # CLI channel (optional)
    if not enable_cli:
        cli = None
    elif cli_channel is not None:
        cli = cli_channel
    else:
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

    # DBOS durable execution — caller owns DBOS lifecycle. We only wire the
    # agent loop reference so replayed workflows can find it. The caller is
    # responsible for constructing DBOS() and calling DBOS.launch() after
    # create() returns but before running the bot.
    set_loop_context(agent_loop)

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
