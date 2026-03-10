"""Subagent manager — nested AgentLoop execution."""

from __future__ import annotations

import asyncio
import uuid
from typing import Callable

from loguru import logger

from exoclaw.agent.conversation import Conversation
from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.protocol import Tool
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.protocol import Bus
from exoclaw.providers.protocol import LLMProvider


class SubagentManager:
    """
    Concrete implementation of the SpawnManager protocol.

    Spawns background subagents by nesting a fresh AgentLoop via
    process_direct — no bespoke loop needed. Results are announced
    back to the main agent as system InboundMessages on the bus.

    Compatible with exoclaw-tools-spawn's SpawnManager protocol.
    """

    def __init__(
        self,
        provider: LLMProvider,
        bus: Bus,
        conversation_factory: Callable[[], Conversation],
        tools: list[Tool] | None = None,
        model: str | None = None,
        max_iterations: int = 15,
    ):
        self._provider = provider
        self._bus = bus
        self._conversation_factory = conversation_factory
        self._tools = tools or []
        self._model = model
        self._max_iterations = max_iterations
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
    ) -> str:
        """Spawn a background subagent. Returns immediately."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or (task[:30] + ("..." if len(task) > 30 else ""))

        bg_task = asyncio.create_task(
            self._run(task_id, task, display_label, origin_channel, origin_chat_id, session_key)
        )
        self._running_tasks[task_id] = bg_task

        def _cleanup(_: asyncio.Task[None]) -> None:
            self._running_tasks.pop(task_id, None)

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run(
        self,
        task_id: str,
        task: str,
        label: str,
        origin_channel: str,
        origin_chat_id: str,
        session_key: str | None,
    ) -> None:
        """Execute the subagent and announce the result."""
        logger.info("Subagent [{}] starting: {}", task_id, label)
        status = "completed"

        try:
            loop = AgentLoop(
                bus=self._bus,
                provider=self._provider,
                conversation=self._conversation_factory(),
                model=self._model,
                max_iterations=self._max_iterations,
                tools=self._tools,
            )
            result = await loop.process_direct(task)
        except Exception as e:
            result = f"Error: {e}"
            status = "failed"
            logger.error("Subagent [{}] failed: {}", task_id, e)

        logger.info("Subagent [{}] {}", task_id, status)
        await self._announce(label, task, result, status, origin_channel, origin_chat_id, session_key)

    async def _announce(
        self,
        label: str,
        task: str,
        result: str,
        status: str,
        origin_channel: str,
        origin_chat_id: str,
        session_key: str | None,
    ) -> None:
        """Publish the subagent result back to the main agent via the bus."""
        content = (
            f"[Subagent '{label}' {status}]\n\n"
            f"Task: {task}\n\n"
            f"Result:\n{result}\n\n"
            "Summarize this naturally for the user. Keep it brief (1-2 sentences). "
            "Do not mention technical details like 'subagent' or task IDs."
        )
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin_channel}:{origin_chat_id}",
            content=content,
            session_key_override=session_key,
        )
        await self._bus.publish_inbound(msg)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
