"""Default ConsolidationPolicy: rolling summary + sidecar-backed tail pointer.

Implements the policy-as-transform contract from ``protocols.py``. The
policy owns its own state in a per-session JSON sidecar
(``_consolidation_state.py``). The session log itself stays
append-only — this policy never mutates message data.

Two operating surfaces:

* ``transform(reader)`` (read seam): emit the rolling summary as a
  synthetic system preamble, then stream the unconsolidated tail
  (everything past ``state.summarized_through``).
* ``on_turn_complete(reader)`` (background seam): if the
  unconsolidated tail has crossed ``memory_window``, summarize one
  chunk. Long sessions catch up over multiple turns rather than
  freezing one turn for a multi-chunk batch.

Reactive overflow recovery hangs off ``Conversation.recover_from_overflow``
(in ``conversation.py``), which calls this policy's
``recover_from_overflow`` to advance the sidecar by summarizing the
next chunk. The agent loop catches ``ContextWindowExceededError``,
asks the conversation to recover, and retries with the smaller view.
There is no preemptive routing — overflow recovery is reactive only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from exoclaw._compat import get_logger

from . import _consolidation_state as state_io

if TYPE_CHECKING:
    from exoclaw._compat import Path

    from .protocols import MemoryBackend, SessionReader

logger = get_logger()


# Async chunk-summarizer callable — defaults to MemoryBackend.summarize
# but can be overridden for tests / Temporal-activity boundaries where
# the LLM call must run via a different transport.
#
# Subscribing ``Callable[[list[dict[...]], bool], Awaitable[...]]`` at
# module level evaluates ``list[dict[...]]`` eagerly, which MicroPython
# 1.27 rejects with ``TypeError: 'type' object isn't subscriptable``
# (built-in ``list``/``dict`` aren't subscriptable in MP). Gate the
# real alias on ``TYPE_CHECKING`` so type checkers see it; at runtime
# ``ChunkSummarizer`` is just ``object`` and stringified annotations
# resolve via the type-checker side. Same pattern as core's
# ``PriorSource`` in ``exoclaw/executor.py``.
if TYPE_CHECKING:
    ChunkSummarizer = Callable[[list[dict[str, Any]], bool], Awaitable["str | None"]]
else:
    ChunkSummarizer = object


class SummarizingConsolidationPolicy:
    """Rolling-summary policy backed by a per-session JSON sidecar.

    Constructor parameters:

    * ``memory``: ``MemoryBackend`` used to summarize chunks and to
      provide the long-term memory preamble for the system prompt.
    * ``state_dir``: directory where per-session sidecars live. By
      convention this is the same directory as the session JSONL files
      so sidecars sit beside their sessions and the migration shim in
      ``_consolidation_state.load_state`` can find legacy
      ``last_consolidated`` headers.
    * ``memory_window``: messages of unconsolidated tail before
      ``on_turn_complete`` triggers a periodic consolidation pass.
      Also the chunk size used by the recovery cascade.
    * ``summarize_chunk``: optional override for how a chunk gets
      summarized. Defaults to ``memory.summarize``. Overridden by
      callers (e.g. Temporal workflows) that need to route the LLM
      call through a different transport than the in-process backend.
    """

    def __init__(
        self,
        memory: "MemoryBackend",
        state_dir: "Path",
        *,
        memory_window: int = 50,
        summarize_chunk: ChunkSummarizer | None = None,
    ) -> None:
        self._memory = memory
        self._state_dir = state_dir
        self._memory_window = memory_window
        self._summarize_chunk: ChunkSummarizer = summarize_chunk or self._default_summarize_chunk

    async def _default_summarize_chunk(
        self, chunk: list[dict[str, Any]], archive_all: bool
    ) -> str | None:
        return await self._memory.summarize(chunk, archive_all=archive_all)

    # ─── ConsolidationPolicy protocol surface ───

    def transform(
        self,
        reader: "SessionReader",
        *,
        budget: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Materialize the LLM-input view from the append-only log.

        Emits ``[summary preamble?, tail...]`` where the tail is the
        segment of the log past ``state.summarized_through``.

        ``budget`` is accepted for ``ConsolidationPolicy`` protocol
        compatibility but ignored — preemptive routing was dropped in
        favor of reactive-only overflow recovery via
        ``recover_from_overflow``.
        """

        del budget  # protocol-level kwarg; reactive-only design ignores it

        async def _gen() -> AsyncIterator[dict[str, Any]]:
            state = state_io.load_state(self._state_dir, reader.key)

            if state.summary:
                yield {
                    "role": "system",
                    "content": _format_summary_preamble(state.summary),
                }

            async for msg in reader.stream(start=state.summarized_through):
                yield msg

        return _gen()

    async def on_turn_complete(self, reader: "SessionReader") -> None:
        """Periodic-consolidation hook. Runs at most one summarize pass
        per call; long sessions catch up over multiple turns rather
        than freezing one turn for a multi-chunk batch."""
        state = state_io.load_state(self._state_dir, reader.key)
        total = await reader.count()
        unconsolidated = total - state.summarized_through

        if unconsolidated < self._memory_window:
            return

        await self._summarize_one_chunk(reader, state, total)
        state_io.save_state(self._state_dir, reader.key, state)

    async def recover_from_overflow(self, reader: "SessionReader") -> bool:
        """Reactive overflow-recovery seam. Summarizes one chunk and
        advances the sidecar pointer.

        Called by ``DefaultConversation.recover_from_overflow`` when
        the agent loop catches ``ContextWindowExceededError``. Returns
        ``True`` if the pointer advanced (caller re-materializes the
        view and retries the LLM call); ``False`` if there's nothing
        left to summarize.

        Only summarizes one chunk per call — the agent loop's
        ``max_recovery_attempts`` cap drives multiple invocations if
        a single chunk isn't enough. That keeps each recovery attempt
        bounded in latency and lets the loop give up cleanly.
        """
        state = state_io.load_state(self._state_dir, reader.key)
        total = await reader.count()
        advanced = await self._summarize_one_chunk(reader, state, total)
        if advanced:
            state_io.save_state(self._state_dir, reader.key, state)
        return advanced

    # ─── Internal: chunk summarization + boundary repair ───

    async def _summarize_one_chunk(
        self,
        reader: "SessionReader",
        state: state_io.ConsolidationState,
        total: int,
    ) -> bool:
        """Summarize the next ``memory_window``-sized chunk past the
        current pointer. Returns True if the pointer advanced."""
        chunk_end_target = min(total, state.summarized_through + self._memory_window)
        if chunk_end_target <= state.summarized_through:
            return False

        chunk: list[dict[str, Any]] = []
        async for msg in reader.stream(start=state.summarized_through, end=chunk_end_target):
            chunk.append(msg)
        if not chunk:
            return False

        history_entry = await self._summarize_chunk(chunk, False)
        if history_entry is None:
            logger.warning(
                "consolidation_chunk_summarize_failed",
                **{"session.key": reader.key, "chunk.size": len(chunk)},
            )
            return False

        # Advance past tool_use/tool_result pairs to avoid orphaning
        # a tool_result whose tool_call_id lives in the archived
        # region — providers like MiniMax reject the next request
        # otherwise. Walk forward from the proposed boundary,
        # peeking at neighboring messages via ``reader.at`` (cheap,
        # bounded — at most a handful of messages near the cut).
        new_boundary = await self._repair_boundary(reader, chunk_end_target, total)
        state.summarized_through = new_boundary
        state.summary = _merge_summary(state.summary, history_entry)
        logger.info(
            "consolidation_chunk_summarized",
            **{
                "session.key": reader.key,
                "chunk.size": len(chunk),
                "summarized_through": state.summarized_through,
            },
        )
        return True

    async def _repair_boundary(
        self,
        reader: "SessionReader",
        boundary: int,
        total: int,
    ) -> int:
        """Walk ``boundary`` forward past any tool_use/tool_result pair
        it would split. Bounded by ``total`` so a runaway chain can't
        push past the end of the log."""
        cur_idx = boundary
        while cur_idx < total:
            curr = await reader.at(cur_idx)
            if curr is None:
                break
            if curr.get("role") == "tool":
                cur_idx += 1
                continue
            prev = await reader.at(cur_idx - 1) if cur_idx > 0 else None
            if prev is not None and prev.get("role") == "assistant" and prev.get("tool_calls"):
                cur_idx += 1
                continue
            break
        return cur_idx


def _format_summary_preamble(summary: str) -> str:
    """Wrap the rolling summary in a stable header so the model treats
    it as carryover context rather than fresh user text."""
    return f"## Previous Session Summary\n{summary}"


def _merge_summary(existing: str, new_entry: str) -> str:
    """Append the new history entry to the rolling summary. Keeps the
    summary append-style — older context isn't rewritten, just
    extended. Callers that want to bound summary growth should override
    by subclassing or by constructing the policy with a different
    summarizer."""
    if not existing:
        return new_entry.strip()
    return f"{existing.strip()}\n\n{new_entry.strip()}"
