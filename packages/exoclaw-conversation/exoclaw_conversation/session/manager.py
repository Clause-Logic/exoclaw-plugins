"""Session management for conversation history."""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from exoclaw._compat import IS_MICROPYTHON, Path, WeakValueDictionary, get_logger

if TYPE_CHECKING:
    from typing import AsyncIterator

    from ..protocols import SessionReader  # noqa: F401

from ..helpers import ensure_dir, safe_filename

logger = get_logger()

# Note on ``WeakValueDictionary`` import above: on CPython this is
# ``weakref.WeakValueDictionary``; on MicroPython it's a plain ``dict``
# subclass (MP doesn't ship ``weakref``). The original use case is the
# in-flight session-locks dict — entries are short-lived and bounded
# by the active-session count, so a small leak under MP is acceptable.


def _repair_and_project(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair orphan tool references and project to LLM-input shape.

    Two cleanups, in order:

    * Drop ``tool`` messages whose ``tool_call_id`` has no matching
      ``assistant.tool_calls[].id`` earlier in the list, and strip
      ``tool_calls`` entries with no matching ``tool`` response.
    * Project each entry to the minimal LLM-input dict — keep
      ``role``/``content`` plus the tool-pair fields, drop
      ``timestamp`` and any other persistence-only metadata.

    Pure transform — no slicing, no leading-non-user peel. Callers
    own those concerns. Used both by ``_normalize_history`` (full
    in-memory path) and by ``DefaultConversation.build_prompt``
    (policy-streamed path, which preserves the rolling-summary
    preamble that the leading-non-user peel would otherwise drop).
    """
    declared_ids: set[str] = set()
    responded_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if tid := tc.get("id"):
                    declared_ids.add(tid)
        elif m.get("role") == "tool":
            if tid := m.get("tool_call_id"):
                responded_ids.add(tid)
    valid_ids = declared_ids & responded_ids

    if declared_ids != valid_ids or responded_ids != valid_ids:
        repaired: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                if m.get("tool_call_id") in valid_ids:
                    repaired.append(m)
            elif role == "assistant" and m.get("tool_calls"):
                kept = [tc for tc in m["tool_calls"] if tc.get("id") in valid_ids]
                if kept:
                    # Avoid ``{**m, ...}`` — MicroPython 1.27
                    # doesn't support PEP 448 dict-unpacking in
                    # dict literals. Plain copy + assign works on
                    # both runtimes.
                    merged = dict(m)
                    merged["tool_calls"] = kept
                    repaired.append(merged)
                elif m.get("content"):
                    repaired.append({k: v for k, v in m.items() if k != "tool_calls"})
            else:
                repaired.append(m)
        messages = repaired

    out: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
        for k in ("tool_calls", "tool_call_id", "name"):
            if k in m:
                entry[k] = m[k]
        out.append(entry)
    return out


def _normalize_history(
    messages: list[dict[str, Any]], max_messages: int | None = 500
) -> list[dict[str, Any]]:
    """Apply LLM-input cleanup to a slice of messages.

    Slice to ``max_messages``, drop leading non-user messages, then
    delegate to ``_repair_and_project`` for orphan repair + dict
    projection. Used by ``Session.get_history`` (in-memory path) and
    the streaming ``SessionManager.read_history`` (on-disk path) —
    both of which feed the LLM a tail-only view where leading
    non-user messages are mid-conversation noise.
    """
    sliced = messages[-max_messages:] if max_messages else list(messages)

    for i, m in enumerate(sliced):
        if m.get("role") == "user":
            sliced = sliced[i:]
            break

    return _repair_and_project(sliced)


if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass, field

    @dataclass
    class Session:
        """A conversation session — handle to an append-only message log.

        ``messages`` mirrors the on-disk JSONL log when
        ``streaming_history=False`` (the default). Streaming-aware
        ``HistoryStore`` backends keep ``messages`` empty and serve
        reads from disk via ``HistoryStore.reader``. The consolidation
        policy owns boundary state in its own per-session sidecar
        (``_consolidation_state.py``) and is not represented here.
        """

        key: str  # channel:chat_id
        messages: list[dict[str, Any]] = field(default_factory=list)
        created_at: datetime = field(default_factory=datetime.now)
        updated_at: datetime = field(default_factory=datetime.now)
        metadata: dict[str, Any] = field(default_factory=dict)
        _total_messages: int = 0  # Explicit total; 0 = derive from messages

        @property
        def total_messages(self) -> int:
            if self._total_messages > 0:
                return self._total_messages
            return len(self.messages)

        @total_messages.setter
        def total_messages(self, value: int) -> None:
            self._total_messages = value

        def add_message(self, role: str, content: str, **kwargs: Any) -> None:
            new_total = self.total_messages + 1
            # Avoid ``{**kwargs}`` in dict literals — MicroPython
            # 1.27 doesn't support PEP 448 dict-unpacking. Build
            # the dict and update with kwargs instead.
            msg: dict[str, Any] = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
            msg.update(kwargs)
            self.messages.append(msg)
            self._total_messages = new_total
            self.updated_at = datetime.now()

        def get_history(self, max_messages: int | None = 500) -> list[dict[str, Any]]:
            return _normalize_history(self.messages, max_messages=max_messages)

        def clear(self) -> None:
            self.messages = []
            self._total_messages = 0
            self.updated_at = datetime.now()

else:  # pragma: no cover (cpython)

    class Session:
        """MicroPython fallback — plain class with hand-written
        ``__init__``. Same shape as the CPython ``@dataclass`` branch
        above; MP strips ``name: type`` annotations at compile time
        so the runtime decorator can't introspect fields. See
        ``exoclaw/_compat.py`` for the same pattern in core."""

        def __init__(
            self,
            key: str,
            messages: list[dict[str, Any]] | None = None,
            created_at: datetime | None = None,
            updated_at: datetime | None = None,
            metadata: dict[str, Any] | None = None,
            _total_messages: int = 0,
        ) -> None:
            self.key = key
            self.messages = messages if messages is not None else []
            self.created_at = created_at if created_at is not None else datetime.now()
            self.updated_at = updated_at if updated_at is not None else datetime.now()
            self.metadata = metadata if metadata is not None else {}
            self._total_messages = _total_messages

        @property
        def total_messages(self) -> int:
            if self._total_messages > 0:
                return self._total_messages
            return len(self.messages)

        @total_messages.setter
        def total_messages(self, value: int) -> None:
            self._total_messages = value

        def add_message(self, role: str, content: str, **kwargs: Any) -> None:
            new_total = self.total_messages + 1
            msg = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
            for k, v in kwargs.items():
                msg[k] = v
            self.messages.append(msg)
            self._total_messages = new_total
            self.updated_at = datetime.now()

        def get_history(self, max_messages: int | None = 500) -> list[dict[str, Any]]:
            return _normalize_history(self.messages, max_messages=max_messages)

        def clear(self) -> None:
            self.messages = []
            self._total_messages = 0
            self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.

    Only the unconsolidated tail of each session is loaded into RAM.
    Saves append new messages to the JSONL file rather than rewriting it.
    Sessions are weakly cached: get_or_create() returns an existing
    in-memory Session while any caller holds a reference; once
    garbage-collected, the next call reloads from disk.
    """

    def __init__(self, workspace: Path, *, streaming_history: bool = False):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        # WeakValueDict: sessions stay cached while any caller holds a reference,
        # then get GC'd automatically — no unbounded growth.
        # ``WeakValueDictionary`` on CPython, plain ``dict`` on MP
        # (see ``exoclaw._compat`` for the why). CPython entries
        # GC-evict when no caller holds a reference; on MP the
        # entries persist for the process lifetime, but the
        # bound — one per active session — is tiny on a
        # microcontroller's typical workload.
        self._cache: WeakValueDictionary[str, Session] = WeakValueDictionary()
        # streaming_history=True: ``_load`` does not populate ``session.messages``.
        # The unconsolidated tail lives only on disk and ``read_history`` reads
        # it on demand. Cuts the per-session RAM floor — the headline win for
        # multi-tenant deployments where N concurrent sessions each holding
        # their tail blows the cgroup. See docs/memory-model.md Step C.
        self.streaming_history = streaming_history

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

        Reads the metadata header, then either materializes the full
        message log into ``session.messages`` (default) or just counts
        lines for the streaming path. The ``last_consolidated`` field
        in legacy headers is ignored here — the
        ``ConsolidationPolicy`` reads it directly from the JSONL on
        first use to seed its sidecar (see
        ``_consolidation_state.load_state``).
        """
        path = self._get_session_path(key)

        if not path.exists():
            return None

        try:
            metadata: dict[str, Any] = {}
            created_at = None
            total_messages = 0
            buffered_lines: list[str] = []

            with open(str(path)) as f:
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
                            continue

                    # Streaming-aware backends keep ``messages`` empty —
                    # callers go through ``reader()`` for on-demand disk
                    # reads. Non-streaming keeps the full log in RAM as
                    # an in-memory cache.
                    if not self.streaming_history:
                        buffered_lines.append(line)
                    total_messages += 1

            messages = [json.loads(line) for line in buffered_lines]

            session = Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
            )
            session._total_messages = total_messages
            return session
        except Exception as e:
            logger.warning("session_load_failed", **{"session.key": key}, error=e)
            return None

    def read_history(self, key: str, max_messages: int | None = None) -> list[dict[str, Any]]:
        """Read the message log from disk and apply LLM-input cleanup.

        Convenience wrapper retained for tests / external callers that
        still want a synchronous list. The new prompt path goes through
        ``reader()`` and the consolidation policy's ``transform``.
        """
        if self.streaming_history:
            tail = self.load_range(key, 0, 1 << 30)
        else:
            session = self.get_or_create(key)
            tail = list(session.messages)
        return _normalize_history(tail, max_messages=max_messages)

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
        with open(str(path)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '"_type"' in line:
                    try:
                        data = json.loads(line)
                        if data.get("_type") == "metadata":
                            continue
                    except (ValueError, json.JSONDecodeError):
                        # MicroPython's ``json.loads`` raises plain
                        # ``ValueError`` (no ``JSONDecodeError`` is
                        # exposed); CPython's ``JSONDecodeError`` is
                        # a ``ValueError`` subclass. Catch the union.
                        pass
                if idx >= end:
                    break
                if idx >= start:
                    messages.append(json.loads(line))
                idx += 1
        return messages

    def reader(self, key: str) -> "SessionReader":  # noqa: F821
        """Return a streaming reader backed by the on-disk JSONL log.

        Streams messages line-by-line via ``load_range`` — never holds
        the full log in RAM. ``count`` reads the cached/persisted
        ``total_messages`` rather than scanning the file.
        """
        return _JsonlSessionReader(self, key)

    def save(self, session: Session) -> None:
        """Rewrite the full session to disk.

        Used after ``clear()`` or any operation that restructures the
        file. For normal turn recording, prefer ``save_append`` which
        only writes new messages.

        Under ``streaming_history`` the in-memory ``session.messages``
        list is empty, so we re-read from disk before rewriting to
        preserve the persisted log. Empty-after-clear() sessions skip
        the re-read because ``total_messages`` is 0.
        """
        path = self._get_session_path(session.key)

        if self.streaming_history and session.total_messages > 0:
            messages = self.load_range(session.key, 0, session.total_messages)
        else:
            messages = list(session.messages)

        with open(str(path), "w") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
            }
            f.write(json.dumps(metadata_line) + "\n")
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    def save_append(self, session: Session, new_messages: list[dict[str, Any]]) -> None:
        """Append new messages to the JSONL file.

        O(new_messages) — does not read or rewrite existing content.
        Creates the file with a metadata header if it doesn't exist.
        """
        path = self._get_session_path(session.key)

        if not path.exists():
            with open(str(path), "w") as f:
                meta = {
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                }
                f.write(json.dumps(meta) + "\n")
                for msg in new_messages:
                    f.write(json.dumps(msg) + "\n")
        else:
            with open(str(path), "a") as f:
                for msg in new_messages:
                    f.write(json.dumps(msg) + "\n")

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
                with open(str(path)) as f:
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


class _JsonlSessionReader:
    """Streaming ``SessionReader`` backed by ``SessionManager``'s JSONL log.

    Streams messages line-by-line from disk — never holds the full log
    in RAM. ``stream`` is restartable: each call reopens the file.
    ``count`` reuses ``SessionManager``'s cached session metadata when
    available, scanning the file only as a fallback.
    """

    def __init__(self, manager: "SessionManager", key: str) -> None:
        self._manager = manager
        self._key = key

    @property
    def key(self) -> str:
        return self._key

    async def count(self) -> int:
        # Cached path: the session is usually live in the WeakValueDict
        # because the caller just built the prompt from it.
        cached = self._manager._cache.get(self._key)
        if cached is not None:
            return cached.total_messages
        # Cold path: scan the file (skip the metadata line).
        path = self._manager._get_session_path(self._key)
        if not path.exists():
            return 0
        total = 0
        with open(str(path)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if total == 0 and '"_type"' in line:
                    try:
                        if json.loads(line).get("_type") == "metadata":
                            continue
                    except (ValueError, json.JSONDecodeError):
                        pass
                total += 1
        return total

    def stream(
        self,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> "AsyncIterator[dict[str, Any]]":  # noqa: F821
        manager = self._manager
        key = self._key

        async def _gen():  # type: ignore[no-untyped-def]
            path = manager._get_session_path(key)
            if not path.exists():
                return
            stop = end if end is not None else (1 << 30)
            idx = 0
            with open(str(path)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if '"_type"' in line:
                        try:
                            if json.loads(line).get("_type") == "metadata":
                                continue
                        except (ValueError, json.JSONDecodeError):
                            pass
                    if idx >= stop:
                        break
                    if idx >= start:
                        yield json.loads(line)
                    idx += 1

        return _gen()

    async def at(self, index: int) -> "dict[str, Any] | None":
        async for msg in self.stream(start=index, end=index + 1):
            return msg
        return None
