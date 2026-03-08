"""
Nanobot — the composition root.

Usage (Python API):

    import asyncio
    from nanobot import Nanobot
    from nanobot_telegram import TelegramChannel

    app = Nanobot(
        provider=MyProvider(),
        channels=[TelegramChannel(...)],
        workspace="~/.nanobot/workspace",
    )
    asyncio.run(app.run())

Usage (CLI):

    nanobot run bot.py

where bot.py contains a module-level Nanobot instance named `app`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.conversation import Conversation
from nanobot.agent.tools.protocol import Tool
from nanobot.bus.protocol import Bus
from nanobot.channels.manager import ChannelManager
from nanobot.channels.protocol import Channel

if TYPE_CHECKING:
    from nanobot.providers.protocol import LLMProvider


class Nanobot:
    """
    Wires together all nanobot components and runs the event loop.

    The only required arguments are a provider and a list of channels.
    Everything else has a sensible default.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        channels: list[Channel] | None = None,
        tools: list[Tool] | None = None,
        conversation: Conversation | None = None,
        bus: Bus | None = None,
        workspace: str | Path = "~/.nanobot/workspace",
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        max_iterations: int = 40,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        send_progress: bool = True,
        send_tool_hints: bool = False,
    ):
        self.provider = provider
        self.channels = channels or []
        self.tools = tools or []
        self.conversation = conversation
        self.bus = bus  # None = use default MessageBus
        self.workspace = Path(workspace).expanduser()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.send_progress = send_progress
        self.send_tool_hints = send_tool_hints

    def _build(self):
        """Instantiate all internal components. Called once at run time."""
        from nanobot.agent.conversation import DefaultConversation
        from nanobot.agent.loop import AgentLoop
        from nanobot.config.schema import ChannelsConfig

        if self.bus is not None:
            bus = self.bus
        else:
            from nanobot.bus.queue import MessageBus
            bus = MessageBus()

        model = self.model or self.provider.get_default_model()
        conversation = self.conversation or DefaultConversation(
            workspace=self.workspace,
            provider=self.provider,
            model=model,
            memory_window=self.memory_window,
        )

        # Temporary shim: AgentLoop still reads send_progress/send_tool_hints
        # from ChannelsConfig. Will be refactored away with AgentLoop.
        channels_cfg = ChannelsConfig(
            send_progress=self.send_progress,
            send_tool_hints=self.send_tool_hints,
        )

        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            workspace=self.workspace,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_iterations=self.max_iterations,
            reasoning_effort=self.reasoning_effort,
            tools=self.tools,
            conversation=conversation,
            channels_config=channels_cfg,
        )

        channel_manager = ChannelManager(self.channels, bus)

        return bus, agent, channel_manager

    async def run(self) -> None:
        """Start all components and run until interrupted."""
        from nanobot.utils.helpers import sync_workspace_templates

        self.workspace.mkdir(parents=True, exist_ok=True)
        sync_workspace_templates(self.workspace)

        bus, agent, channel_manager = self._build()

        logger.info("Nanobot starting (workspace={})", self.workspace)

        try:
            await asyncio.gather(
                agent.run(),
                channel_manager.start_all(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Shutting down...")
        finally:
            await agent.close_mcp()
            agent.stop()
            await channel_manager.stop_all()
