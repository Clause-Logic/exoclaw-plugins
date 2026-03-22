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
    ) -> list[dict[str, Any]]: ...

    def get_active_optional_tools(self) -> set[str]:
        """Return optional tool names activated by the current turn's skills.

        Optional hook — implementations that don't need skill-scoped tools
        can omit this method; the default returns an empty set.
        """
        return set()
