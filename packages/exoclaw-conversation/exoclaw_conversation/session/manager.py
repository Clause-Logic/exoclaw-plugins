"""Session management for conversation history."""

import json
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from ..helpers import ensure_dir, safe_filename

logger = structlog.get_logger()


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Only the tail of the message history (unconsolidated messages) is kept
    in RAM.  Older messages remain on disk and are never loaded unless
    explicitly requested (e.g. for consolidation).

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Absolute index: messages already consolidated to files
    _total_messages: int = 0  # Explicit total; 0 means "derive from messages"
    _messages_offset: int = 0  # Absolute index of first entry in self.messages

    @property
    def total_messages(self) -> int:
        """Total messages on disk (including consolidated).

        Falls back to offset + len(messages) when not explicitly set.
        """
        if self._total_messages > 0:
            return self._total_messages
        return self._messages_offset + len(self.messages)

    @total_messages.setter
    def total_messages(self, value: int) -> None:
        self._total_messages = value

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        self.messages.append(msg)
        self._total_messages = self.total_messages + 1
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a user turn.

        When loaded from disk, self.messages starts at _messages_offset so
        we compute the relative skip from last_consolidated.
        """
        relative_consolidated = max(0, self.last_consolidated - self._messages_offset)
        unconsolidated = self.messages[relative_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid orphaned tool_result blocks
        for i, m in enumerate(sliced):
            if m.get("role") == "user":
                sliced = sliced[i:]
                break

        out: list[dict[str, Any]] = []
        for m in sliced:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self._total_messages = 0
        self._messages_offset = 0
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.

    Only the unconsolidated tail of each session is loaded into RAM.
    Saves append new messages to the JSONL file rather than rewriting it.
    Sessions are NOT cached — each get_or_create() loads from disk.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        # WeakValueDict: sessions stay cached while any caller holds a reference,
        # then get GC'd automatically — no unbounded growth.
        self._cache: weakref.WeakValueDictionary[str, Session] = weakref.WeakValueDictionary()

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Uses a WeakValueDictionary so sessions stay cached while any
        caller holds a reference, then get GC'd automatically.
        Only the unconsolidated tail of messages is loaded from disk.
        """
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk.

        Only keeps unconsolidated messages (after last_consolidated) in RAM.
        """
        path = self._get_session_path(key)

        if not path.exists():
            return None

        try:
            metadata: dict[str, Any] = {}
            created_at = None
            last_consolidated = 0
            total_messages = 0

            # First pass: read metadata line, count messages, collect tail
            # We use a deque as a sliding window to avoid loading all messages.
            # After reading metadata we know last_consolidated, but we must
            # read through the file to count total and collect the tail.
            all_msg_lines: deque[str] = deque()

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Peek to check for metadata line
                    if total_messages == 0 and '"_type"' in line:
                        data = json.loads(line)
                        if data.get("_type") == "metadata":
                            metadata = data.get("metadata", {})
                            created_at = (
                                datetime.fromisoformat(data["created_at"])
                                if data.get("created_at")
                                else None
                            )
                            last_consolidated = data.get("last_consolidated", 0)
                            continue

                    total_messages += 1
                    all_msg_lines.append(line)

            # Only parse the unconsolidated tail
            unconsolidated_lines = list(all_msg_lines)[last_consolidated:]
            messages = [json.loads(line) for line in unconsolidated_lines]

            session = Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
            session._total_messages = total_messages
            session._messages_offset = last_consolidated
            return session
        except Exception as e:
            logger.warning("session_load_failed", session_key=key, error=e)
            return None

    def load_range(self, key: str, start: int, end: int) -> list[dict[str, Any]]:
        """Load a range of messages from disk by index.

        Useful for consolidation which needs to read messages in
        [last_consolidated : -keep_count] without holding them all in RAM.
        """
        path = self._get_session_path(key)
        if not path.exists():
            return []

        messages: list[dict[str, Any]] = []
        idx = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '"_type"' in line:
                    try:
                        data = json.loads(line)
                        if data.get("_type") == "metadata":
                            continue
                    except json.JSONDecodeError:
                        pass
                if idx >= end:
                    break
                if idx >= start:
                    messages.append(json.loads(line))
                idx += 1
        return messages

    def save(self, session: Session) -> None:
        """Rewrite the full session to disk.

        Used after clear() or consolidation — operations that change
        metadata or restructure the file.  For normal turn recording,
        prefer save_append() which only writes new messages.
        """
        path = self._get_session_path(session.key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def save_append(self, session: Session, new_messages: list[dict[str, Any]]) -> None:
        """Append new messages to the JSONL file and update the metadata line.

        Much cheaper than save() for normal turn recording — avoids
        rewriting the entire file.
        """
        path = self._get_session_path(session.key)

        # Update metadata line (first line) by rewriting just that line
        # Read existing content, replace first line, append new messages
        if path.exists():
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []

        metadata_line = {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "last_consolidated": session.last_consolidated,
        }

        # Replace or insert metadata as first line
        meta_str = json.dumps(metadata_line, ensure_ascii=False) + "\n"
        if lines and "_type" in lines[0]:
            lines[0] = meta_str
        else:
            lines.insert(0, meta_str)

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
            for msg in new_messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def save_metadata(self, session: Session) -> None:
        """Update only the metadata line (first line) of the JSONL file.

        Used after consolidation updates last_consolidated without
        changing messages.
        """
        path = self._get_session_path(session.key)
        if not path.exists():
            return

        metadata_line = {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "last_consolidated": session.last_consolidated,
        }

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        meta_str = json.dumps(metadata_line, ensure_ascii=False) + "\n"
        if lines and "_type" in lines[0]:
            lines[0] = meta_str
        else:
            lines.insert(0, meta_str)

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def invalidate(self, key: str) -> None:
        """Remove a session from the weak cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append(
                                {
                                    "key": key,
                                    "created_at": data.get("created_at"),
                                    "updated_at": data.get("updated_at"),
                                    "path": str(path),
                                }
                            )
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
