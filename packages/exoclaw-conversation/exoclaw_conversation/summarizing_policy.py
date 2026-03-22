"""Default ConsolidationPolicy that adds per-session summary on top of MemoryStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .protocols import MemoryBackend
    from .session.manager import Session

logger = structlog.get_logger()


class SummarizingConsolidationPolicy:
    """Consolidation policy that preserves a per-session summary.

    Delegates the actual consolidation to a ``MemoryBackend`` (same LLM call,
    same MEMORY.md + HISTORY.md writes). After consolidation succeeds, stores
    the history entry as ``session.metadata["summary"]`` so it survives
    history truncation and can be injected into the next turn's context.

    This gives the agent "what was I doing" continuity after compaction —
    without an extra LLM call.
    """

    def __init__(self, memory: MemoryBackend) -> None:
        self._memory = memory

    async def should_consolidate(self, session: Session, *, memory_window: int) -> bool:
        """Trigger when unconsolidated messages reach memory_window."""
        unconsolidated = len(session.messages) - session.last_consolidated
        return unconsolidated >= memory_window

    async def consolidate(
        self,
        session: Session,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Run standard consolidation, then capture the summary."""
        # Read HISTORY.md length before consolidation so we can detect the new entry
        history_before = self._read_history()

        success = await self._memory.consolidate(
            session,
            archive_all=archive_all,
            memory_window=memory_window,
        )

        if success:
            # Extract the new history entry (appended after the old content)
            history_after = self._read_history()
            new_entry = history_after[len(history_before) :].strip()
            if new_entry:
                session.metadata["summary"] = new_entry
                logger.info("session_summary_stored", chars=len(new_entry))

        return success

    def _read_history(self) -> str:
        """Read the current HISTORY.md content."""
        from pathlib import Path

        if hasattr(self._memory, "history_file"):
            path = getattr(self._memory, "history_file")
            if isinstance(path, Path) and path.exists():
                return path.read_text(encoding="utf-8")
        return ""
