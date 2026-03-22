"""Subagent manager — nested AgentLoop execution."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
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


@dataclass
class _BatchState:
    """Tracks completion of a batch of subagents."""

    total: int = 0
    completed: int = 0
    results: list[dict[str, str]] = field(default_factory=list)
    origin_channel: str = "cli"
    origin_chat_id: str = "direct"
    session_key: str | None = None


class SubagentManager:
    """
    Concrete implementation of the SpawnManager protocol.

    Spawns background subagents by nesting a fresh AgentLoop via
    process_direct — no bespoke loop needed. Results are written to
    disk and announced back to the main agent as system InboundMessages
    on the bus with a file path reference (not inline content).

    When ``batch`` is set on spawn, individual completions are silent.
    A single announcement fires when all subagents in the batch complete.

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
        self._batches: dict[str, _BatchState] = {}
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
        batch: str | None = None,
    ) -> str:
        """Spawn a background subagent. Returns immediately."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or (task[:30] + ("..." if len(task) > 30 else ""))

        if batch is not None:
            state = self._batches.setdefault(batch, _BatchState())
            state.total += 1
            state.origin_channel = origin_channel
            state.origin_chat_id = origin_chat_id
            state.session_key = session_key

        bg_task = asyncio.create_task(
            self._run(
                task_id, task, display_label, origin_channel, origin_chat_id, session_key, batch
            )
        )
        self._running_tasks[task_id] = bg_task

        def _cleanup(_: asyncio.Task[None]) -> None:
            self._running_tasks.pop(task_id, None)

        bg_task.add_done_callback(_cleanup)

        logger.info("subagent_spawned", id=task_id, label=display_label, batch=batch)
        return f"Subagent [{display_label}] started (id: {task_id})."

    async def _run(
        self,
        task_id: str,
        task: str,
        label: str,
        origin_channel: str,
        origin_chat_id: str,
        session_key: str | None,
        batch: str | None,
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
                tools=[t for t in self._tools if t.name != "spawn"],
            )
            result = await loop.process_direct(task)
        except Exception as e:
            result = f"Error: {e}"
            status = "failed"
            logger.error("subagent_failed", id=task_id, error=e)

        logger.info("subagent_done", id=task_id, status=status, batch=batch)

        # Write result to disk so it survives compaction
        result_path = self._write_result(task_id, label, task, result, status)

        if batch is not None:
            await self._record_batch_completion(batch, label, status, result_path, result)
        else:
            await self._announce_single(
                label, task, result, result_path, status,
                origin_channel, origin_chat_id, session_key,
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
        content = (
            f"# Subagent: {label}\n\n"
            f"**Status:** {status}\n\n"
            f"## Task\n\n{task}\n\n"
            f"## Result\n\n{result}\n"
        )
        path.write_text(content, encoding="utf-8")
        logger.info("subagent_result_written", path=str(path))
        return str(path)

    async def _record_batch_completion(
        self,
        batch: str,
        label: str,
        status: str,
        result_path: str | None,
        result: str,
    ) -> None:
        """Record a batch member's completion. Announce when all are done."""
        state = self._batches.get(batch)
        if state is None:
            return

        state.completed += 1
        state.results.append({
            "label": label,
            "status": status,
            "path": result_path or "(no file)",
        })
        logger.info(
            "batch_progress", batch=batch,
            completed=state.completed, total=state.total,
        )

        if state.completed >= state.total:
            await self._announce_batch(batch, state)
            del self._batches[batch]

    async def _announce_batch(self, batch: str, state: _BatchState) -> None:
        """Announce that all subagents in a batch have completed."""
        completed = [r for r in state.results if r["status"] == "completed"]
        failed = [r for r in state.results if r["status"] != "completed"]

        lines = [f"[Batch '{batch}' complete — {len(completed)} succeeded, {len(failed)} failed]\n"]

        if completed:
            lines.append("Results:")
            for r in completed:
                lines.append(f"- **{r['label']}**: {r['path']}")

        if failed:
            lines.append("\nFailed:")
            for r in failed:
                lines.append(f"- **{r['label']}**: {r['status']}")

        lines.append(
            "\nRead each result file with read_file, then synthesize "
            "the findings and respond to the user."
        )

        content = "\n".join(lines)
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{state.origin_channel}:{state.origin_chat_id}",
            content=content,
            session_key_override=state.session_key,
            metadata={"session_key": state.session_key} if state.session_key else {},
        )
        logger.info("batch_announced", batch=batch, results=len(state.results))
        await self._bus.publish_inbound(msg)

    async def _announce_single(
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
        """Publish a single (non-batched) subagent result back to the main agent."""
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
            metadata={"session_key": session_key} if session_key else {},
        )
        await self._bus.publish_inbound(msg)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel running subagents. Returns count cancelled."""
        cancelled = 0
        for task_id, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled
