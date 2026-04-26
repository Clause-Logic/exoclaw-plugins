"""Internal protocols for DefaultConversation sub-components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .session.manager import Session


@runtime_checkable
class HistoryStore(Protocol):
    """Protocol for session history persistence."""

    def get_or_create(self, key: str) -> "Session": ...
    def save(self, session: "Session") -> None: ...
    def invalidate(self, key: str) -> None: ...
    def list_sessions(self) -> list[dict[str, Any]]: ...

    def save_append(self, session: "Session", new_messages: list[dict[str, Any]]) -> None:
        """Append new messages to disk. Falls back to full save()."""
        self.save(session)

    def save_metadata(self, session: "Session") -> None:
        """Update metadata without rewriting messages. Falls back to full save()."""
        self.save(session)

    def load_range(self, key: str, start: int, end: int) -> list[dict[str, Any]]:
        """Load a range of messages from disk by index. Returns empty list by default."""
        return []

    def read_history(self, key: str, max_messages: int | None = None) -> list[dict[str, Any]]:
        """Return the unconsolidated tail for LLM input, applying orphan repair.

        Default implementation reads from ``get_or_create(key).get_history()`` —
        which materializes ``session.messages`` into RAM. Streaming-aware
        backends override this to read the tail directly from disk / DB on
        each call so the unconsolidated history isn't held between turns.
        ``max_messages=None`` lets the backend return the full unconsolidated
        tail (callers that don't want a window cap pass ``None``).
        """
        session = self.get_or_create(key)
        return session.get_history(max_messages=max_messages)


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for long-term memory storage and consolidation."""

    def get_memory_context(self) -> str: ...
    async def consolidate(
        self,
        session: "Session",
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool: ...

    async def consolidate_messages(
        self,
        session: "Session",
        *,
        old_messages: list[dict[str, Any]],
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate pre-loaded messages. Falls back to consolidate()."""
        return await self.consolidate(session, archive_all=archive_all, memory_window=memory_window)


@runtime_checkable
class ConsolidationPolicy(Protocol):
    """Pluggable consolidation strategy.

    Controls *when* and *how* old messages are consolidated. The default
    behaviour (no policy) delegates to MemoryBackend.consolidate() which
    runs a single LLM call to produce MEMORY.md + HISTORY.md updates.

    Implement this protocol to add custom behaviour at consolidation
    boundaries — per-session summaries, task-tracker sync, daily notes,
    multi-stage summarization, or anything else.
    """

    async def should_consolidate(
        self,
        session: "Session",
        *,
        memory_window: int,
    ) -> bool:
        """Return True when consolidation should run."""
        ...

    async def consolidate(
        self,
        session: "Session",
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Run consolidation. Returns True on success."""
        ...


@runtime_checkable
class PromptBuilder(Protocol):
    """Protocol for assembling the LLM message list."""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        *,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        extra_context: str | None = None,
        turn_context: list[str] | None = None,
        isolated: bool = False,
    ) -> list[dict[str, Any]]: ...

    def get_active_optional_tools(self) -> set[str]:
        """Return optional tool names activated by the current turn's skills.

        Optional hook — implementations that don't need skill-scoped tools
        can omit this method; the default returns an empty set.
        """
        return set()
