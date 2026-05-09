"""Default ``SessionReader`` implementation.

Wraps a ``HistoryStore`` that doesn't yet provide a native streaming
reader. Backends with on-disk storage (``SessionManager``) override
``HistoryStore.reader`` with a true line-by-line streaming impl that
never holds the full log in RAM. This default is a correctness
fallback — it materializes via ``load_range`` so it works for any
store but loses the memory-budget guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from .protocols import HistoryStore


class _DefaultSessionReader:
    """Generic streaming reader over a ``HistoryStore``.

    Restartable: ``stream()`` re-reads on each call. ``count()`` and
    ``at()`` go through the store directly so backends that maintain
    a cheap index can answer them without scanning the full log.
    """

    def __init__(self, store: "HistoryStore", key: str) -> None:
        self._store = store
        self._key = key

    @property
    def key(self) -> str:
        return self._key

    async def count(self) -> int:
        # Fall back to scanning load_range from 0 to a large index;
        # backends with a cheaper index path should override this
        # entire reader rather than relying on the default.
        msgs = self._store.load_range(self._key, 0, 1 << 30)
        return len(msgs)

    def stream(
        self,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async def _gen() -> AsyncIterator[dict[str, Any]]:
            stop = end if end is not None else (1 << 30)
            for msg in self._store.load_range(self._key, start, stop):
                yield msg

        return _gen()

    async def at(self, index: int) -> dict[str, Any] | None:
        msgs = self._store.load_range(self._key, index, index + 1)
        return msgs[0] if msgs else None
