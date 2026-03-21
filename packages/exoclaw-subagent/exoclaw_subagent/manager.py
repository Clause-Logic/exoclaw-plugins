"""Subagent manager — nested AgentLoop execution."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Callable

import structlog
from exoclaw.agent.conversation import Conversation
from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.protocol import Tool
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.protocol import Bus
from exoclaw.providers.protocol import LLMProvider

logger = structlog.get_logger()


def _safe_filename(label: str) -> str:
    """Convert a label to a safe filename."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in label).strip("-")


class SubagentManager:
    """
    Concrete implementation of the SpawnManager protocol.

    Spawns background subagents by nesting a fresh AgentLoop via
    process_direct — no bespoke loop needed. Results are written to
    disk and announced back to the main agent as system InboundMessages
    on the bus with a file path reference (not inline content).

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
        workspace: Path | None = None,
    ):
        self._provider = provider
        self._bus = bus
        self._conversation_factory = conversation_factory
        self._tools = tools or []
        self._model = model
        self._max_iterations = max_iterations
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._results_dir: Path | None = None
        if workspace is not None:
            self._results_dir = workspace / "subagents"
            self._results_dir.mkdir(parents=True, exist_ok=True)

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

        logger.info("subagent_spawned", id=task_id, label=display_label)
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
        logger.info("subagent_starting", id=task_id, label=label)
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
            logger.error("subagent_failed", id=task_id, error=e)

        logger.info("subagent_done", id=task_id, status=status)

        # Write result to disk so it survives compaction (skip for errors)
        result_path = None
        if status == "completed":
            result_path = self._write_result(task_id, label, task, result, status)

        await self._announce(
            label, task, result, result_path, status, origin_channel, origin_chat_id, session_key
        )

    def _write_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        status: str,
    ) -> str | None:
        """Write subagent result to disk. Returns the file path, or None if no workspace."""
        if self._results_dir is None:
            return None
        filename = f"{_safe_filename(label)}-{task_id}.md"
        path = self._results_dir / filename
        content = f"# Subagent: {label}\n\n**Status:** {status}\n\n## Task\n\n{task}\n\n## Result\n\n{result}\n"
        path.write_text(content, encoding="utf-8")
        logger.info("subagent_result_written", path=str(path))
        return str(path)

    async def _announce(
        self,
        label: str,
        task: str,
        result: str,
        result_path: str | None,
        status: str,
        origin_channel: str,
        origin_chat_id: str,
        session_key: str | None,
    ) -> None:
        """Publish the subagent result back to the main agent via the bus."""
        if result_path:
            content = (
                f"[Subagent '{label}' {status}]\n\n"
                f"Task: {task}\n\n"
                f"Result saved to: {result_path}\n\n"
                "Read the file to see the full result, then summarize for the user."
            )
        else:
            # No workspace or error — include result inline
            content = (
                f"[Subagent '{label}' {status}]\n\n"
                f"Task: {task}\n\n"
                f"Result:\n{result}\n\n"
                "Summarize this naturally for the user."
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
