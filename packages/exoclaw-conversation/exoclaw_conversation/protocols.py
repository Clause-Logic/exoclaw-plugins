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
class PromptBuilder(Protocol):
    """Protocol for assembling the LLM message list."""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        *,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        extra_context: str | None = None,
    ) -> list[dict[str, Any]]: ...
