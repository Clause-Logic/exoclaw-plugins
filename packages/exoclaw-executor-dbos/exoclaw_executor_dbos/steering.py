"""Durable mailbox for user messages that arrive during an active turn.

``exoclaw`` owns the policy for *when* a new user message can be injected
(between model calls and tool calls).  This module owns the executor-side
transport: append incoming content to a per-session mailbox before returning
control to the channel, then drain it through a DBOS step when the core loop
reaches one of those safe boundaries.

The split is intentional.  An inbound channel callback starts a DBOS workflow
under the channel event's id; its first step performs the fsync-backed append.
The destructive drain runs in the active turn workflow as a DBOS step; on
recovery DBOS returns the recorded messages again, allowing the core's
journaled conversation appends to finish instead of losing an already-drained
message.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from dbos import DBOS
from exoclaw.bus.events import InboundMessage

# Both channel callbacks and DBOS workflow steps run in this process.  A
# process-wide lock makes the rename-and-drain operation atomic with respect to
# a concurrent append, without relying on an asyncio event loop being shared
# by the queue manager and recovered workflows.
_INBOX_LOCK = threading.Lock()


def _inbox_path(root: Path, session_id: str) -> Path:
    """Return an opaque, single-file mailbox path for ``session_id``."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return root / f"{digest}.jsonl"


def _append_message(root: Path, session_id: str, content: str) -> None:
    """Append one message and fsync it before acknowledging the channel."""
    payload = (json.dumps({"content": content}, ensure_ascii=False) + "\n").encode("utf-8")
    path = _inbox_path(root, session_id)
    root.mkdir(parents=True, exist_ok=True)

    with _INBOX_LOCK:
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            # A single record write while holding the process-wide lock keeps
            # records intact even for unusually large channel messages.
            written = 0
            while written < len(payload):
                written += os.write(fd, payload[written:])
            os.fsync(fd)
        finally:
            os.close(fd)


def _drain_messages(root: Path, session_id: str) -> list[str]:
    """Atomically take all currently pending messages for a session.

    The temporary ``.draining`` file survives a crash that occurs while this
    function is executing.  A retry drains it before newer messages appended
    to the fresh live mailbox, preserving arrival order.
    """
    path = _inbox_path(root, session_id)
    draining_path = path.with_suffix(".draining")

    with _INBOX_LOCK:
        files: list[Path] = []
        if draining_path.exists():
            files.append(draining_path)
        if path.exists():
            if files:
                # A prior drain was interrupted after the rename.  Keep its
                # older records and consume the newer live mailbox separately
                # rather than replacing ``.draining`` and losing the old one.
                files.append(path)
            else:
                os.replace(path, draining_path)
                files.append(draining_path)

        pending: list[str] = []
        for candidate in files:
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                continue
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = record.get("content")
                if isinstance(content, str):
                    pending.append(content)
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        return pending


@DBOS.step()
async def _drain_steering_step(root: str, session_id: str) -> list[str]:
    """Destructively drain one mailbox, journaled by the active workflow."""
    return _drain_messages(Path(root), session_id)


@DBOS.step()
async def _append_steering_step(root: str, session_id: str, content: str) -> None:
    """Persist one inbound message as a DBOS-journaled filesystem write."""
    _append_message(Path(root), session_id, content)


class SteeringInbox:
    """Per-session durable inbox backed by files under a persistent workspace."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def store(self, msg: InboundMessage) -> None:
        """Persist a message as a step in its inbound DBOS workflow."""
        await _append_steering_step(str(self._root), msg.session_key, msg.content)

    async def drain(self, session_id: str) -> list[str]:
        """Return and consume pending content through a DBOS-journaled step."""
        return await _drain_steering_step(str(self._root), session_id)
