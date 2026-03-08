"""DefaultConversation — file-backed implementation of the Conversation protocol."""

from __future__ import annotations

import asyncio
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from .context import ContextBuilder
from .memory import MemoryStore
from .session.manager import Session, SessionManager

if TYPE_CHECKING:
    from exoclaw.providers.protocol import LLMProvider

# Tag used to strip runtime context when persisting user messages
_RUNTIME_CONTEXT_TAG = ContextBuilder._RUNTIME_CONTEXT_TAG

_TOOL_RESULT_MAX_CHARS = 500


class DefaultConversation:
    """
    File-backed conversation state manager.

    Implements the exoclaw Conversation protocol without inheriting from any
    exoclaw class.

    - build_prompt: builds messages via ContextBuilder, triggers background
      consolidation when unconsolidated history exceeds memory_window.
    - record: saves new turn messages (stripping runtime context, truncating
      large tool results) into the JSONL session file.
    - clear: archives the current session to memory and resets to fresh state.
    - list_sessions: lists all sessions from the sessions directory.
    """

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        memory_window: int = 100,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.memory_window = memory_window

        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)

        self._consolidating: set[str] = set()
        self._consolidation_tasks: set[asyncio.Task] = set()
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
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
    ) -> list[dict[str, Any]]:
        """Return the full messages list to send to the LLM."""
        session = self.sessions.get_or_create(session_id)

        # Trigger background consolidation when history is long
        unconsolidated = len(session.messages) - session.last_consolidated
        if unconsolidated >= self.memory_window and session_id not in self._consolidating:
            self._consolidating.add(session_id)
            lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())

            async def _consolidate_and_unlock() -> None:
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session_id)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        history = session.get_history(max_messages=self.memory_window)

        # plugin_context is a list of extra context strings from pre_context hooks
        extra_context: str | None = None
        if plugin_context:
            extra_context = "\n\n".join(plugin_context)

        return self.context.build_messages(
            history=history,
            current_message=message,
            media=media,
            channel=channel,
            chat_id=chat_id,
            extra_context=extra_context,
        )

    async def record(
        self,
        session_id: str,
        new_messages: list[dict[str, Any]],
    ) -> None:
        """Persist the messages produced during one turn."""
        session = self.sessions.get_or_create(session_id)
        self._save_turn(session, new_messages)
        self.sessions.save(session)

    async def clear(self, session_id: str) -> bool:
        """Archive current session to memory and start fresh. Returns True on success."""
        session = self.sessions.get_or_create(session_id)
        lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())
        self._consolidating.add(session_id)
        try:
            async with lock:
                snapshot = session.messages[session.last_consolidated:]
                if snapshot:
                    temp = Session(key=session_id)
                    temp.messages = list(snapshot)
                    success = await self._consolidate_memory(temp, archive_all=True)
                    if not success:
                        return False
        except Exception:
            logger.exception("clear() archival failed for {}", session_id)
            return False
        finally:
            self._consolidating.discard(session_id)

        session.clear()
        self.sessions.save(session)
        self.sessions.invalidate(session_id)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all known sessions."""
        return self.sessions.list_sessions()

    async def _consolidate_memory(
        self, session: Session, archive_all: bool = False
    ) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session,
            self.provider,
            self.model,
            archive_all=archive_all,
            memory_window=self.memory_window,
        )

    def _save_turn(self, session: Session, messages: list[dict[str, Any]]) -> None:
        """Save turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages:
            entry = dict(m)
            role = entry.get("role")
            content = entry.get("content")

            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context

            if role == "tool" and isinstance(content, str) and len(content) > _TOOL_RESULT_MAX_CHARS:
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
                        if (
                            c.get("type") == "image_url"
                            and c.get("image_url", {}).get("url", "").startswith("data:image/")
                        ):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered

            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()
