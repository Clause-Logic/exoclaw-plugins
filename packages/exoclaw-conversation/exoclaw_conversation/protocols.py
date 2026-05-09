"""Internal protocols for DefaultConversation sub-components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .session.manager import Session


@runtime_checkable
class SessionReader(Protocol):
    """Read-only, streaming view of a session's append-only message log.

    All access is lazy — implementations stream from disk and never
    materialize the full log in RAM. Restartable: ``stream()`` may be
    called multiple times.

    A ``SessionReader`` is policy-facing. The policy uses it to inspect
    the log without coupling to ``HistoryStore`` internals or holding
    the messages in Python memory.
    """

    @property
    def key(self) -> str:
        """The session key this reader is bound to."""
        ...

    async def count(self) -> int:
        """Total messages currently in the log. Cheap — backed by the
        store's index, not a full scan."""
        ...

    def stream(
        self,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream messages in [start, end) one at a time. ``end=None``
        streams to the current tail. Restartable — call again to re-read.
        """
        ...

    async def at(self, index: int) -> dict[str, Any] | None:
        """Random-access read of one message. Returns ``None`` if out of
        range. For peek/lookahead — not for bulk reads."""
        ...


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

    def reader(self, key: str) -> SessionReader:
        """Return a streaming reader for the session's append-only log.

        Default implementation wraps ``load_range`` / ``read_history`` —
        correct but not memory-efficient. Backends should override with
        a true streaming impl (e.g. line-by-line JSONL read, DB cursor).
        """
        from ._reader import _DefaultSessionReader

        return _DefaultSessionReader(self, key)

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
    """Protocol for long-term memory storage and summarization.

    The backend's job is to produce two artifacts from a list of
    messages: a long-term memory document (e.g. ``MEMORY.md``) and a
    grep-searchable history log entry (e.g. ``HISTORY.md``). It does
    not own session state — boundaries, summaries, and view assembly
    belong to ``ConsolidationPolicy``.
    """

    def get_memory_context(self) -> str:
        """Return text to inject into the system prompt as long-term
        memory context. Empty string when no memory has been accumulated."""
        ...

    async def summarize(
        self,
        messages: list[dict[str, Any]],
        *,
        archive_all: bool = False,
    ) -> str | None:
        """Summarize ``messages`` and persist the result to long-term
        memory + history artifacts. Returns the new history-log entry
        text on success (the policy uses it as its rolling preamble),
        or ``None`` on failure.

        Pure with respect to session state — this method must not
        read or mutate any session/policy state. The caller (the
        consolidation policy) owns boundary advancement and sidecar
        persistence.
        """
        ...


@runtime_checkable
class ConsolidationPolicy(Protocol):
    """Pluggable consolidation strategy.

    A policy owns the *view* the LLM sees: it transforms the append-only
    message log into the message list sent to the model. It may drop,
    replace, prepend (e.g. with a summary), or truncate messages — and
    it persists its own state in a sidecar next to the session file.
    The session log itself is append-only and never mutated by the
    policy.

    ``transform`` is the read seam: ``DefaultConversation`` calls it
    every turn to materialize the LLM input from a ``SessionReader``.
    ``on_turn_complete`` is the write seam: called once per turn so
    the policy can run any deferred work (token-estimate maintenance,
    background summarization) and persist its sidecar.

    Policies receive no ``Session`` handle. They are constructed with
    whatever state-store they need; the only runtime input is a
    streaming reader over the session's append-only log.
    """

    def transform(
        self,
        reader: SessionReader,
        *,
        budget: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Transform a streaming view of the session log into the
        message list to send to the LLM.

        May drop, replace, prepend, or truncate messages. May call
        ``reader.stream()`` multiple times, peek with ``at()``, check
        size with ``count()``. Must not hold the full log in memory.

        If ``budget`` is given, the emitted stream should aim to fit
        within ``budget`` tokens (best-effort, for overflow recovery).
        Without ``budget``, normal consolidation rules apply.

        Default is a passthrough — no transformation.
        """

        async def _passthrough() -> AsyncIterator[dict[str, Any]]:
            async for m in reader.stream():
                yield m

        return _passthrough()

    async def on_turn_complete(self, reader: SessionReader) -> None:
        """Notify the policy a turn finished. Lets it run background
        work (chunk-token-estimate updates, deferred summarization)
        and persist its sidecar.

        Default is a no-op."""
        return None


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
