"""DefaultConversation — file-backed implementation of the Conversation protocol."""

from __future__ import annotations

import asyncio
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from .protocols import ConsolidationPolicy, HistoryStore, MemoryBackend, PromptBuilder
from .session.manager import Session

logger = structlog.get_logger()

if TYPE_CHECKING:
    from exoclaw.providers.protocol import LLMProvider

# Injected before each user message at call time; stripped before persisting.
_RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

_TOOL_RESULT_MAX_CHARS = 500


class DefaultConversation:
    """
    File-backed conversation state manager.

    Implements the exoclaw Conversation protocol without inheriting from any
    exoclaw class.

    Accepts HistoryStore, MemoryBackend, and PromptBuilder as constructor
    arguments so each layer can be replaced independently. Use
    DefaultConversation.create() for the standard file-backed setup.

    - build_prompt: builds messages via PromptBuilder, triggers background
      consolidation when unconsolidated history exceeds memory_window.
    - record: saves new turn messages (stripping runtime context, truncating
      large tool results) into the JSONL session file.
    - clear: archives the current session to memory and resets to fresh state.
    - list_sessions: lists all sessions from the sessions directory.
    """

    def __init__(
        self,
        history: HistoryStore,
        memory: MemoryBackend,
        prompt: PromptBuilder,
        memory_window: int = 100,
        consolidation_policy: ConsolidationPolicy | None = None,
    ):
        self.history = history
        self.memory = memory
        self.prompt = prompt
        self.memory_window = memory_window
        self._consolidation_policy = consolidation_policy

        self._consolidating: set[str] = set()
        self._consolidation_tasks: set[asyncio.Task[Any]] = set()
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    @classmethod
    def create(
        cls,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        memory_window: int = 100,
        skill_packages: list[str] | None = None,
        consolidation_policy: ConsolidationPolicy | None = None,
    ) -> DefaultConversation:
        """Construct with the standard file-backed implementations."""
        from .context import ContextBuilder
        from .memory import MemoryStore
        from .session.manager import SessionManager

        memory = MemoryStore(workspace, provider, model)
        return cls(
            history=SessionManager(workspace),
            memory=memory,
            prompt=ContextBuilder(workspace, memory=memory, skill_packages=skill_packages),
            memory_window=memory_window,
            consolidation_policy=consolidation_policy,
        )

    async def build_prompt(
        self,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        turn_context: list[str] | None = None,
        **kwargs: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Return the full messages list to send to the LLM."""
        skills: list[str] | None = kwargs.get("skills")
        session = self.history.get_or_create(session_id)

        # Trigger background consolidation when policy says so (or default: history is long)
        should = await self._should_consolidate(session)
        if should and session_id not in self._consolidating:
            self._consolidating.add(session_id)
            lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())

            async def _consolidate_and_unlock() -> None:
                try:
                    async with lock:
                        success = await self._consolidate_memory(session)
                        if success:
                            self.history.save_metadata(session)
                finally:
                    self._consolidating.discard(session_id)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        history = session.get_history(max_messages=self.memory_window)

        extra_context: str | None = None
        if plugin_context:
            extra_context = "\n\n".join(plugin_context)

        # Inject per-session summary from consolidation policy (if present)
        effective_turn_context = list(turn_context or [])
        summary = session.metadata.get("summary")
        if summary:
            effective_turn_context.insert(0, f"## Previous Session Summary\n{summary}")

        return self.prompt.build_messages(
            history=history,
            current_message=message,
            skill_names=skills,
            media=media,
            channel=channel,
            chat_id=chat_id,
            extra_context=extra_context,
            turn_context=effective_turn_context or None,
        )

    async def record(
        self,
        session_id: str,
        new_messages: list[dict[str, Any]],
    ) -> None:
        """Persist the messages produced during one turn."""
        session = self.history.get_or_create(session_id)
        prepared = self._prepare_turn(session, new_messages)
        self.history.save_append(session, prepared)

    async def clear(self, session_id: str) -> bool:
        """Archive current session to memory and start fresh. Returns True on success."""
        session = self.history.get_or_create(session_id)
        lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())
        self._consolidating.add(session_id)
        try:
            async with lock:
                # session.messages contains only unconsolidated messages
                snapshot = list(session.messages)
                if snapshot:
                    temp = Session(key=session_id)
                    temp.messages = list(snapshot)
                    success = await self._consolidate_memory(temp, archive_all=True)
                    if not success:
                        return False
        except Exception:
            logger.exception("session_clear_failed", session_id=session_id)
            return False
        finally:
            self._consolidating.discard(session_id)

        session.clear()
        self.history.save(session)
        self.history.invalidate(session_id)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all known sessions."""
        return self.history.list_sessions()

    def active_tools(self) -> set[str]:
        """Return optional tool names activated by the current turn's skills."""
        return self.prompt.get_active_optional_tools()

    async def _should_consolidate(self, session: Session) -> bool:
        """Check whether consolidation should run."""
        if self._consolidation_policy is not None:
            return await self._consolidation_policy.should_consolidate(
                session, memory_window=self.memory_window
            )
        unconsolidated = session.total_messages - session.last_consolidated
        return unconsolidated >= self.memory_window

    async def _consolidate_memory(self, session: Session, archive_all: bool = False) -> bool:
        """Delegate to ConsolidationPolicy if present, otherwise MemoryBackend.

        Loads the consolidation range from disk so we don't need the full
        message history in RAM.
        """
        if self._consolidation_policy is not None:
            return await self._consolidation_policy.consolidate(
                session,
                archive_all=archive_all,
                memory_window=self.memory_window,
            )

        # Load only the messages that need consolidating from disk
        if archive_all:
            old_messages = self._load_consolidation_range(session, archive_all=True)
        else:
            old_messages = self._load_consolidation_range(session, archive_all=False)

        if not old_messages:
            return True

        return await self.memory.consolidate_messages(
            session,
            old_messages=old_messages,
            archive_all=archive_all,
            memory_window=self.memory_window,
        )

    def _load_consolidation_range(
        self, session: Session, *, archive_all: bool
    ) -> list[dict[str, Any]]:
        """Load the message range to consolidate from disk."""
        if archive_all:
            loaded = self.history.load_range(session.key, 0, session.total_messages)
            return loaded or list(session.messages)

        keep_count = self.memory_window // 2
        if session.total_messages <= keep_count:
            return []
        end = session.total_messages - keep_count
        start = session.last_consolidated
        if start >= end:
            return []
        loaded = self.history.load_range(session.key, start, end)
        return loaded or session.messages[: end - start]

    def _prepare_turn(
        self, session: Session, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare turn messages, truncating large tool results.

        Appends to session.messages (in-memory view) and returns the
        list of prepared entries for disk persistence.
        """
        from datetime import datetime

        prepared: list[dict[str, Any]] = []

        for m in messages:
            entry = dict(m)
            role = entry.get("role")
            content = entry.get("content")

            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context

            if (
                role == "tool"
                and isinstance(content, str)
                and len(content) > _TOOL_RESULT_MAX_CHARS
            ):
                entry["content"] = content[:_TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"

            elif role == "user":
                if isinstance(content, str) and content.startswith(_RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if (
                            c.get("type") == "text"
                            and isinstance(c.get("text"), str)
                            and c["text"].startswith(_RUNTIME_CONTEXT_TAG)
                        ):
                            continue
                        if c.get("type") == "image_url" and c.get("image_url", {}).get(
                            "url", ""
                        ).startswith("data:image/"):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered

            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
            session.total_messages += 1
            prepared.append(entry)

        session.updated_at = datetime.now()
        return prepared
