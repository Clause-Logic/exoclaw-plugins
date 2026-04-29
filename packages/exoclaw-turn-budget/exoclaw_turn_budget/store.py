"""Pluggable storage for budget tracker state.

The default ``InMemoryBudgetStore`` mirrors the original (non-durable)
behavior — fine for chip deploys and tests. ``FileBudgetStore`` writes
JSON to a configured path with atomic rename so a kill mid-write can't
corrupt the counter. A DBOS-backed store ships separately in
``exoclaw-executor-dbos`` for deployments that already use DBOS for
durable agent-loop state.

Tracker integration: pass ``store=`` to ``DailyBudgetTracker`` to opt in.
The tracker calls ``load()`` once at construction and ``save(state)`` on
every recorded chat() and on day-boundary auto-reset. ``clear()`` is
exposed for ops use (e.g., a "reset my daily budget" admin command).

**Security note** — when using ``FileBudgetStore`` with an agent that has
filesystem tools (read/write/edit/exec), put the state path *outside*
the agent's workspace. Both the workspace tools' ``allowed_dir`` check
and ``ExecTool``'s ``restrict_to_workspace=True`` block paths outside
the workspace boundary; a state file inside workspace would let a
prompt-injected or self-correcting agent edit its own quota.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover (runtime)
    # ``Protocol`` and ``runtime_checkable`` aren't on MicroPython's
    # ``typing`` shim. Type-checking-only import keeps both runtimes
    # happy: ty/mypy see the protocol; MP never tries to load it.
    from pathlib import Path
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class BudgetStateStore(Protocol):
        """Read / update / reset durable storage for a budget tracker.

        State is an opaque ``dict[str, object]`` — the tracker owns the
        schema and the store just persists what it's given.
        """

        def load(self) -> "dict[str, object] | None":
            """Return the persisted state, or ``None`` for a fresh start."""
            ...

        def save(self, state: "dict[str, object]") -> None:
            """Persist *state*. Implementations should be atomic where
            relevant — a kill mid-write must not leave the store in a
            partially-updated state that would mis-attribute tokens.
            """
            ...

        def clear(self) -> None:
            """Forget any persisted state. Next ``load()`` returns ``None``."""
            ...


class InMemoryBudgetStore:
    """Non-durable store — state lives in process memory only.

    Default for ``DailyBudgetTracker`` so chip deploys and tests don't
    pull in the filesystem at all. A container restart starts the daily
    counter back at zero, which on a long-running server is the wrong
    answer; pair with ``FileBudgetStore`` for those.
    """

    def __init__(self) -> None:
        self._state: "dict[str, object] | None" = None

    def load(self) -> "dict[str, object] | None":
        return self._state

    def save(self, state: "dict[str, object]") -> None:
        self._state = state

    def clear(self) -> None:
        self._state = None


class FileBudgetStore:
    """JSON file store with atomic write.

    Writes go to a sibling ``.tmp`` file then ``replace()`` onto the
    target — so a SIGKILL in the middle of writing leaves the previous
    state intact rather than producing a half-written JSON document
    that ``load()`` would have to reject (and start the counter over).

    The state schema is the tracker's concern — this class just
    serializes / deserializes. Bad JSON returns ``None`` (fresh start)
    rather than raising, since the failure mode of "we lost the counter
    once" is recoverable; raising would crash the agent loop on every
    chat() call until the file is hand-fixed.
    """

    def __init__(self, path: "str | Path") -> None:
        # ``str | Path`` accepted to avoid forcing every caller through
        # ``exoclaw._compat.Path`` import. Coerce here.
        from exoclaw._compat import Path as _Path

        self._path: "_Path" = path if isinstance(path, _Path) else _Path(str(path))

    def load(self) -> "dict[str, object] | None":
        if not self._path.exists():
            return None
        try:
            text = self._path.read_text()
        except OSError:
            return None
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def save(self, state: "dict[str, object]") -> None:
        # ``json.dumps`` without ``indent`` / ``ensure_ascii`` — both
        # kwargs are dropped on MicroPython 1.27. Ops debugging on a
        # one-line file is fine; this state is small (~120 bytes).
        text = json.dumps(state)
        parent = self._path.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        # Atomic write — temp + rename. ``os.rename`` is the cross-runtime
        # atomic-on-POSIX primitive; the ``Path.replace`` method isn't on
        # the MP-compat ``Path`` shim so we drop to ``os.rename`` instead.
        tmp = parent / (self._path.name + ".tmp")
        tmp.write_text(text)
        os.rename(str(tmp), str(self._path))

    def clear(self) -> None:
        try:
            self._path.unlink()
        except (OSError, FileNotFoundError):
            return
