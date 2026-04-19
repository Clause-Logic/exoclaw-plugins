"""Subagent manager — nested AgentLoop execution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog
import structlog.contextvars
from exoclaw.agent.conversation import Conversation
from exoclaw.agent.loop import AgentLoop
from exoclaw.agent.tools.protocol import Tool
from exoclaw.bus.events import InboundMessage
from exoclaw.bus.protocol import Bus
from exoclaw.providers.protocol import LLMProvider

from .spawner import AsyncioSpawner, SpawnerFactory, SubagentHandle

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
        spawner_factory: SpawnerFactory | None = None,
    ):
        self._provider = provider
        self._bus = bus
        self._conversation_factory = conversation_factory
        self._tools = tools or []
        self._model = model
        self._max_iterations = max_iterations
        self._handles: dict[str, SubagentHandle] = {}
        self._sessions: dict[str, str | None] = {}
        self._results_dir: Path | None = None
        self._batches: dict[str, _BatchState] = {}
        if workspace is not None:
            self._results_dir = workspace / "subagents"
            self._results_dir.mkdir(parents=True, exist_ok=True)

        # Runner adapter re-resolves self._run at call time so tests that
        # patch ``_run`` on the instance are still picked up by the spawner.
        async def _runner(**kwargs: Any) -> None:
            tid = kwargs["task_id"]
            try:
                await self._run(**kwargs)
            finally:
                self._handles.pop(tid, None)
                self._sessions.pop(tid, None)

        factory = spawner_factory or AsyncioSpawner
        self._spawner = factory(_runner)

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        batch: str | None = None,
        skills: list[str] | None = None,
        model: str | None = None,
        parent_turn_chain: str | None = None,
        parent_turn_id: str | None = None,
    ) -> str:
        """Spawn a background subagent. Returns immediately.

        ``model`` overrides the manager-wide default model for this spawn
        only — useful for routing a single task at a cheaper model while
        the main agent keeps its configured default.

        ``parent_turn_chain`` and ``parent_turn_id`` carry the parent
        turn's trace ancestry through to the child; ``_run`` rebinds
        them into structlog contextvars before the child agent loop
        starts so the child's own ``_process_turn_inline`` extends the
        chain instead of starting a fresh root. Durable spawners pass
        these as workflow arguments so the ancestry survives replay.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or (task[:30] + ("..." if len(task) > 30 else ""))

        if batch is not None:
            state = self._batches.setdefault(batch, _BatchState())
            state.total += 1
            state.origin_channel = origin_channel
            state.origin_chat_id = origin_chat_id
            state.session_key = session_key

        handle = await self._spawner.start(
            task_id=task_id,
            task=task,
            label=display_label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            session_key=session_key,
            batch=batch,
            skills=skills,
            model=model,
            parent_turn_chain=parent_turn_chain,
            parent_turn_id=parent_turn_id,
        )
        self._handles[task_id] = handle
        self._sessions[task_id] = session_key

        logger.info(
            "subagent_spawned",
            **{
                "subagent.id": task_id,
                "subagent.label": display_label,
                "batch.id": batch,
                "parent_turn.id": parent_turn_id,
                "parent_turn.chain": parent_turn_chain,
            },
        )
        return f"Subagent [{display_label}] started (id: {task_id})."

    async def _run(
        self,
        task_id: str,
        task: str,
        label: str,
        origin_channel: str,
        origin_chat_id: str,
        session_key: str | None,
        batch: str | None = None,
        skills: list[str] | None = None,
        model: str | None = None,
        parent_turn_chain: str | None = None,
        parent_turn_id: str | None = None,
    ) -> None:
        """Execute the subagent and announce the result.

        If the parent passed a turn ancestry, rebind it into the local
        structlog contextvars before the child agent loop starts. The
        child's ``_process_turn_inline`` reads these contextvars when
        deciding its own ``turn.root_id`` / ``turn.parent_id`` /
        ``turn.chain``, so the binding here is what makes the trace
        ancestry extend across the spawn boundary instead of resetting.

        Wrapped in try/finally so the binding is unwound even on
        crash; structlog contextvars are per-asyncio-Task and cleaning
        up matters when the same worker reuses a task for the next
        subagent.
        """
        status = "completed"
        effective_model = model if model is not None else self._model

        _bound_keys: list[str] = []
        if parent_turn_chain is not None or parent_turn_id is not None:
            bind_payload: dict[str, str] = {}
            if parent_turn_chain is not None:
                bind_payload["turn.chain"] = parent_turn_chain
                _bound_keys.append("turn.chain")
                # Derive ``turn.root_id`` from the chain independently
                # of whether ``turn.id`` was provided. The chain is a
                # ``root:child:…``-joined string so the first segment
                # is the root; if a caller only gave us a chain (e.g.
                # a custom spawner that routed through a channel that
                # only carries the chain field), we still want log
                # lines emitted before the child mints its own
                # ``turn.id`` to be queryable by ``turn.root_id``.
                bind_payload["turn.root_id"] = parent_turn_chain.split(":", 1)[0]
                _bound_keys.append("turn.root_id")
            if parent_turn_id is not None:
                bind_payload["turn.id"] = parent_turn_id
                _bound_keys.append("turn.id")
            structlog.contextvars.bind_contextvars(**bind_payload)

        try:
            try:
                loop = AgentLoop(
                    bus=self._bus,
                    provider=self._provider,
                    conversation=self._conversation_factory(),
                    model=effective_model,
                    max_iterations=self._max_iterations,
                    tools=[t for t in self._tools if t.name != "spawn"],
                )
                kwargs: dict = {}
                if skills is not None:
                    kwargs["skills"] = skills
                # Isolate the child's on-disk conversation. Without an
                # explicit session_key, every child falls through to
                # process_direct's ``cli:direct`` default and
                # reads/writes the same JSONL as every sibling, so
                # build_prompt loads the tail of prior subagents' turns
                # as "history" and the child mimics whatever pattern
                # was in that shared tail.
                parent_session = session_key or f"{origin_channel}:{origin_chat_id}"
                child_session = f"subagent:{parent_session}:{task_id}"
                result = await loop.process_direct(
                    task,
                    session_key=child_session,
                    channel="subagent",
                    chat_id=task_id,
                    **kwargs,
                )
            except Exception as e:
                result = f"Error: {e}"
                status = "failed"
                logger.error("subagent_failed", **{"subagent.id": task_id}, error=e)

            logger.info(
                "subagent_done",
                **{"subagent.id": task_id, "subagent.status": status, "batch.id": batch},
            )

            # Write result to disk so it survives compaction
            result_path = self._write_result(task_id, label, task, result, status)

            if batch is not None:
                await self._record_batch_completion(batch, label, status, result_path, result)
            else:
                await self._announce_single(
                    label,
                    task,
                    result,
                    result_path,
                    status,
                    origin_channel,
                    origin_chat_id,
                    session_key,
                )
        finally:
            # Unbind only after the announcement and disk write so all
            # of the subagent's own log lines carry the parent ancestry.
            if _bound_keys:
                structlog.contextvars.unbind_contextvars(*_bound_keys)

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
        logger.info("subagent_result_written", **{"file.path": str(path)})
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
        state.results.append(
            {
                "label": label,
                "status": status,
                "path": result_path or "(no file)",
            }
        )
        logger.info(
            "batch_progress",
            **{"batch.id": batch, "batch.completed": state.completed, "batch.total": state.total},
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
        logger.info("batch_announced", **{"batch.id": batch, "batch.results": len(state.results)})
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
        return sum(1 for h in self._handles.values() if not h.done())

    def get_status(self) -> dict:
        """Return status of all subagents: running, batches, and completed results on disk."""
        running = [{"id": h.id, "done": h.done()} for h in self._handles.values()]

        batches = {}
        for batch_id, state in self._batches.items():
            batches[batch_id] = {
                "total": state.total,
                "completed": state.completed,
                "results": state.results,
            }

        completed = []
        if self._results_dir and self._results_dir.exists():
            for f in sorted(
                self._results_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                completed.append({"path": str(f), "name": f.stem})

        return {"running": running, "batches": batches, "completed": completed}

    def list_results(self, limit: int = 20) -> list[dict[str, str]]:
        """List completed subagent result files from disk."""
        if not self._results_dir or not self._results_dir.exists():
            return []
        results = []
        for f in sorted(
            self._results_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            if len(results) >= limit:
                break
            results.append({"path": str(f), "name": f.stem})
        return results

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel running subagents spawned from ``session_key``.

        Returns the number of subagents actually cancelled.
        """
        cancelled = 0
        for task_id, handle in list(self._handles.items()):
            if self._sessions.get(task_id) != session_key:
                continue
            if not handle.done():
                await handle.cancel()
                cancelled += 1
        return cancelled
