"""Default ConsolidationPolicy: rolling summary + sidecar-backed tail pointer.

Implements the policy-as-transform contract from ``protocols.py``. The
policy owns its own state in a per-session JSON sidecar
(``_consolidation_state.py``). The session log itself stays
append-only — this policy never mutates message data.

Two operating modes:

* **Normal** (``transform(reader)``, no budget): emit the rolling
  summary as a synthetic system preamble, then stream the
  unconsolidated tail (everything past ``state.summarized_through``).
* **Budget** (``transform(reader, budget=N)``): if the estimated
  unconsolidated tail exceeds ``N`` tokens, run the OpenClaw-style
  cascade — summarize the next chunk via the ``MemoryBackend``,
  advance the sidecar pointer, repeat until the tail fits or the
  attempt cap is reached. Then emit. Used for synchronous overflow
  recovery before each LLM call.

``on_turn_complete`` runs the periodic background consolidation:
update the running token estimate; if the unconsolidated tail has
crossed the consolidation threshold, summarize one chunk.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from exoclaw._compat import get_logger

from . import _consolidation_state as state_io

if TYPE_CHECKING:
    from exoclaw._compat import Path

    from .protocols import MemoryBackend, SessionReader

logger = get_logger()


# Default chars-per-token estimator. Cheap, runs without a tokenizer
# library so the policy works on MicroPython.
def _default_token_estimator(message: dict[str, Any]) -> int:
    try:
        return max(1, len(json.dumps(message)) // 4)
    except (TypeError, ValueError):
        # Fallback for non-JSON-serializable content blocks.
        text = str(message.get("content", ""))
        return max(1, len(text) // 4)


# Type alias for the token-estimator callable.
TokenEstimator = Callable[[dict[str, Any]], int]
# Async chunk-summarizer callable — defaults to MemoryBackend.summarize
# but can be overridden for tests / Temporal-activity boundaries where
# the LLM call must run via a different transport.
ChunkSummarizer = Callable[[list[dict[str, Any]], bool], Awaitable["str | None"]]


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
      Also the chunk size used by the budget-mode cascade.
    * ``target_context_ratio``: ``transform(budget=...)`` callers
      typically pass ``int(model.context_window * target_context_ratio)``;
      the ratio itself isn't read by the policy, it's documented here
      as the recommended default.
    * ``max_overflow_attempts``: cap on the number of summarize-and-
      advance iterations the budget cascade will run before giving up.
      Returning a still-too-big stream is an honest signal to the
      caller; better than looping forever.
    * ``token_estimator``: callable that estimates a single message's
      token count. Default is a chars/4 heuristic; pass a real
      tokenizer when one is available.
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
        target_context_ratio: float = 0.75,
        max_overflow_attempts: int = 3,
        token_estimator: TokenEstimator | None = None,
        summarize_chunk: ChunkSummarizer | None = None,
    ) -> None:
        self._memory = memory
        self._state_dir = state_dir
        self._memory_window = memory_window
        self._target_context_ratio = target_context_ratio
        self._max_overflow_attempts = max_overflow_attempts
        self._estimate = token_estimator or _default_token_estimator
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

        Without ``budget``: emits ``[summary preamble?, tail...]``
        where the tail is the segment of the log past
        ``state.summarized_through``.

        With ``budget``: if the running tail estimate exceeds
        ``budget``, runs the cascade (summarize next chunk, advance
        pointer, persist sidecar) up to ``max_overflow_attempts``
        times. Then emits as above.
        """

        async def _gen() -> AsyncIterator[dict[str, Any]]:
            state = state_io.load_state(self._state_dir, reader.key)

            if budget is not None:
                state = await self._apply_budget(reader, state, budget)

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
            # Below threshold — just refresh the running estimate so
            # budget-mode decisions stay accurate. Cheap because it
            # streams from disk and only sums a bounded window.
            await self._refresh_estimate(reader, state, total)
            state_io.save_state(self._state_dir, reader.key, state)
            return

        await self._summarize_one_chunk(reader, state, total)
        state_io.save_state(self._state_dir, reader.key, state)

    # ─── Internal: cascade + helpers ───

    async def _apply_budget(
        self,
        reader: "SessionReader",
        state: state_io.ConsolidationState,
        budget: int,
    ) -> state_io.ConsolidationState:
        """Summarize chunks until the estimated tail fits ``budget``,
        capped by ``max_overflow_attempts``. Persists the sidecar
        after each successful chunk so partial progress survives a
        crash mid-cascade."""
        attempts = 0
        total = await reader.count()
        # Refresh the estimate from disk before deciding — the cached
        # value may be stale if writes have happened since the last
        # ``on_turn_complete``.
        await self._refresh_estimate(reader, state, total)

        while state.unconsolidated_token_estimate > budget:
            if attempts >= self._max_overflow_attempts:
                logger.warning(
                    "consolidation_budget_giveup",
                    **{
                        "session.key": reader.key,
                        "budget.tokens": budget,
                        "tail.estimate": state.unconsolidated_token_estimate,
                        "attempts": attempts,
                    },
                )
                break
            attempts += 1
            advanced = await self._summarize_one_chunk(reader, state, total)
            state_io.save_state(self._state_dir, reader.key, state)
            if not advanced:
                # Nothing left to summarize but we still don't fit —
                # the unconsolidated tail past the boundary is
                # legitimately oversized. Caller will see the too-big
                # stream and surface it.
                break
        return state

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
        async for msg in reader.stream(
            start=state.summarized_through, end=chunk_end_target
        ):
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
        await self._refresh_estimate(reader, state, total)
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
            if (
                prev is not None
                and prev.get("role") == "assistant"
                and prev.get("tool_calls")
            ):
                cur_idx += 1
                continue
            break
        return cur_idx

    async def _refresh_estimate(
        self,
        reader: "SessionReader",
        state: state_io.ConsolidationState,
        total: int,
    ) -> None:
        """Recompute ``state.unconsolidated_token_estimate`` by streaming
        the post-pointer tail. Bounded memory (sum is one int, messages
        released as iterated)."""
        running = 0
        async for msg in reader.stream(start=state.summarized_through, end=total):
            running += self._estimate(msg)
        state.unconsolidated_token_estimate = running


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
