"""Direct tests for ``SummarizingConsolidationPolicy`` budget mode +
the sidecar I/O helpers + the default ``SessionReader`` fallback.

The end-to-end test in
``exoclaw-nanobot/tests/test_phase_persistence_integration.py``
exercises the policy through ``AgentLoop`` + DBOS + a real provider,
which covers the ``on_turn_complete`` happy path. These tests cover
the spots that path doesn't reach: the budget-mode cascade
(preemptive overflow recovery), the ``max_overflow_attempts`` cap,
the "nothing left to summarize" break, the migration shim, and the
default ``HistoryStore.reader`` fallback that ships for backends
which don't override it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from exoclaw_conversation import _consolidation_state as ss
from exoclaw_conversation._reader import _DefaultSessionReader
from exoclaw_conversation.summarizing_policy import SummarizingConsolidationPolicy


class _ListReader:
    """Minimal in-memory ``SessionReader`` for unit tests. Mirrors
    the production reader's contract on a Python list."""

    def __init__(self, key: str, source: list[dict[str, Any]]) -> None:
        self._key = key
        self._source = source

    @property
    def key(self) -> str:
        return self._key

    async def count(self) -> int:
        return len(self._source)

    def stream(self, *, start: int = 0, end: int | None = None):
        async def _gen():
            stop = end if end is not None else len(self._source)
            for msg in self._source[start:stop]:
                yield msg

        return _gen()

    async def at(self, index: int):
        if 0 <= index < len(self._source):
            return self._source[index]
        return None


def _summarize_succeeds(label: str = "[chunk]") -> Any:
    """Build a memory backend whose ``summarize`` always returns a
    synthetic history entry — used to drive the cascade past a
    chunk boundary."""
    memory = MagicMock(spec=["get_memory_context", "summarize"])
    memory.get_memory_context = MagicMock(return_value="")
    memory.summarize = AsyncMock(return_value=label)
    return memory


def _summarize_fails() -> Any:
    """Backend whose ``summarize`` returns ``None`` — simulates LLM
    refusal / provider error mid-cascade."""
    memory = MagicMock(spec=["get_memory_context", "summarize"])
    memory.get_memory_context = MagicMock(return_value="")
    memory.summarize = AsyncMock(return_value=None)
    return memory


@pytest.mark.asyncio
class TestBudgetModeCascade:
    async def test_under_budget_is_noop(self, tmp_path: Path) -> None:
        """When the estimated tail already fits, the cascade does
        nothing — no summarize calls, no sidecar advance."""
        memory = _summarize_succeeds()
        policy = SummarizingConsolidationPolicy(
            memory=memory, state_dir=tmp_path, memory_window=4
        )
        # 2 small messages — under any reasonable budget.
        reader = _ListReader("ut:fits", [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        out = [m async for m in policy.transform(reader, budget=10_000)]
        assert [m["role"] for m in out] == ["user", "assistant"]
        memory.summarize.assert_not_called()
        sidecar = ss.load_state(tmp_path, "ut:fits")
        assert sidecar.summarized_through == 0

    async def test_cascade_advances_until_fits(self, tmp_path: Path) -> None:
        """Budget below the unconsolidated tail estimate triggers the
        cascade. Each iteration summarizes one chunk and advances the
        sidecar pointer; the loop stops once the remaining tail fits."""
        memory = _summarize_succeeds("[archived]")
        policy = SummarizingConsolidationPolicy(
            memory=memory, state_dir=tmp_path, memory_window=2
        )
        # 8 messages, each ~25 chars = ~6 tokens estimated. Total
        # ~48 tokens. budget=20 forces ~5 chunks of 2 to be archived
        # before the tail fits.
        msgs = [{"role": "user", "content": f"message-{i}"} for i in range(8)]
        reader = _ListReader("ut:cascade", msgs)
        out = [m async for m in policy.transform(reader, budget=20)]
        # Cascade ran at least once.
        assert memory.summarize.await_count >= 1
        sidecar = ss.load_state(tmp_path, "ut:cascade")
        assert sidecar.summarized_through > 0
        assert "[archived]" in sidecar.summary
        # Output starts with the summary preamble.
        assert out[0]["role"] == "system"
        assert "[archived]" in out[0]["content"]

    async def test_cascade_caps_at_max_attempts(self, tmp_path: Path) -> None:
        """If the budget is so tight that even max_overflow_attempts
        archival passes can't shrink the tail enough, the policy gives
        up rather than looping forever. The caller sees a still-too-big
        stream and surfaces it as an honest overflow."""
        memory = _summarize_succeeds("[chunk]")
        policy = SummarizingConsolidationPolicy(
            memory=memory,
            state_dir=tmp_path,
            memory_window=1,
            max_overflow_attempts=2,
        )
        msgs = [{"role": "user", "content": f"message-{i}" * 5} for i in range(10)]
        reader = _ListReader("ut:cap", msgs)
        # Tiny budget — won't be reachable.
        _ = [m async for m in policy.transform(reader, budget=1)]
        assert memory.summarize.await_count == 2, (
            "expected exactly max_overflow_attempts summarize calls; "
            f"got {memory.summarize.await_count}"
        )

    async def test_cascade_breaks_when_summarize_fails(self, tmp_path: Path) -> None:
        """If the LLM declines to summarize (returns None), the
        cascade can't make progress — the pointer doesn't advance,
        ``_summarize_one_chunk`` returns False, and the loop breaks
        rather than spinning."""
        memory = _summarize_fails()
        policy = SummarizingConsolidationPolicy(
            memory=memory,
            state_dir=tmp_path,
            memory_window=2,
            max_overflow_attempts=5,
        )
        msgs = [{"role": "user", "content": f"big-{i}" * 20} for i in range(6)]
        reader = _ListReader("ut:fail", msgs)
        _ = [m async for m in policy.transform(reader, budget=10)]
        # Tried once, failed, broke. Did NOT spin to max_overflow_attempts.
        assert memory.summarize.await_count == 1
        sidecar = ss.load_state(tmp_path, "ut:fail")
        assert sidecar.summarized_through == 0
        assert sidecar.summary == ""

    async def test_cascade_persists_progress_each_iteration(
        self, tmp_path: Path
    ) -> None:
        """The sidecar is written after every successful chunk —
        partial progress survives a crash mid-cascade."""
        memory = _summarize_succeeds("[partial]")
        policy = SummarizingConsolidationPolicy(
            memory=memory, state_dir=tmp_path, memory_window=2
        )
        msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(6)]
        reader = _ListReader("ut:persist", msgs)
        _ = [m async for m in policy.transform(reader, budget=5)]
        # Reload the sidecar from disk — verify it matches in-memory state.
        from_disk = ss.load_state(tmp_path, "ut:persist")
        assert from_disk.summarized_through > 0


@pytest.mark.asyncio
class TestOnTurnCompleteThreshold:
    async def test_below_window_only_refreshes_estimate(self, tmp_path: Path) -> None:
        """Under the memory_window threshold, on_turn_complete
        refreshes the running token estimate but doesn't summarize.
        The estimate matters because budget-mode reads it first."""
        memory = _summarize_succeeds()
        policy = SummarizingConsolidationPolicy(
            memory=memory, state_dir=tmp_path, memory_window=10
        )
        reader = _ListReader("ut:idle", [
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "ok"},
        ])
        await policy.on_turn_complete(reader)
        memory.summarize.assert_not_called()
        sidecar = ss.load_state(tmp_path, "ut:idle")
        assert sidecar.summarized_through == 0
        assert sidecar.unconsolidated_token_estimate > 0

    async def test_above_window_summarizes_one_chunk(self, tmp_path: Path) -> None:
        """At/above the threshold, on_turn_complete summarizes ONE
        chunk per call — long sessions catch up over multiple turns
        rather than freezing the current turn for a multi-chunk batch."""
        memory = _summarize_succeeds("[hop]")
        policy = SummarizingConsolidationPolicy(
            memory=memory, state_dir=tmp_path, memory_window=4
        )
        msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
        reader = _ListReader("ut:hop", msgs)
        await policy.on_turn_complete(reader)
        # Exactly one summarize call — the loop is per-turn, not catch-all.
        assert memory.summarize.await_count == 1
        sidecar = ss.load_state(tmp_path, "ut:hop")
        assert sidecar.summarized_through > 0


@pytest.mark.asyncio
class TestBoundaryRepair:
    async def test_advance_past_orphan_tool_pair(self, tmp_path: Path) -> None:
        """A naive cut between an assistant(tool_calls=Tn) and its
        matching tool(Tn) leaves an orphan tool_result in the kept
        tail. The policy walks the boundary forward past the pair so
        the emitted view never starts mid-pair — that's the MiniMax
        400-invalid_params guard from the production incident."""
        memory = _summarize_succeeds("[ok]")
        policy = SummarizingConsolidationPolicy(
            memory=memory, state_dir=tmp_path, memory_window=2
        )
        # 1 user, then assistant(tool_calls=T1) at idx 1, tool(T1) at idx 2,
        # plus more pairs. memory_window=2 cuts at idx 2 — but the cut
        # would split T1 across the boundary. Repair must advance past it.
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "T1", "function": {"name": "x", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "T1", "content": "result"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "T2", "function": {"name": "x", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "T2", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        reader = _ListReader("ut:repair", msgs)
        await policy.on_turn_complete(reader)
        sidecar = ss.load_state(tmp_path, "ut:repair")
        # Boundary must NOT land on idx 2 (the orphan tool_result).
        assert sidecar.summarized_through != 2, (
            "boundary repair didn't advance past the tool_use/tool_result pair"
        )


class TestConsolidationStateMigration:
    def test_migrates_legacy_last_consolidated(self, tmp_path: Path) -> None:
        """Legacy session JSONL with ``last_consolidated > 0`` gets
        migrated into a fresh sidecar on first ``load_state``."""
        legacy = tmp_path / "telegram_42.jsonl"
        legacy.write_text(
            json.dumps({
                "_type": "metadata",
                "key": "telegram:42",
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
                "metadata": {"summary": "carryover from old session"},
                "last_consolidated": 17,
            }) + "\n"
        )
        state = ss.load_state(tmp_path, "telegram:42")
        assert state.summarized_through == 17
        assert state.summary == "carryover from old session"
        # Sidecar persisted — second load reads it instead of re-migrating.
        assert (tmp_path / "telegram_42.consolidation.json").exists()

    def test_no_migration_when_legacy_is_clean(self, tmp_path: Path) -> None:
        """Sessions with ``last_consolidated=0`` and no summary don't
        trigger a sidecar write — saves an unnecessary file for fresh
        sessions."""
        legacy = tmp_path / "telegram_99.jsonl"
        legacy.write_text(
            json.dumps({
                "_type": "metadata",
                "key": "telegram:99",
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
                "metadata": {},
                "last_consolidated": 0,
            }) + "\n"
        )
        state = ss.load_state(tmp_path, "telegram:99")
        assert state.summarized_through == 0
        assert not (tmp_path / "telegram_99.consolidation.json").exists()

    def test_corrupt_sidecar_falls_back_to_empty(self, tmp_path: Path) -> None:
        """A corrupt JSON sidecar must not crash the policy — load
        returns a fresh empty state. The next save will overwrite the
        bad file."""
        sidecar = tmp_path / "telegram_55.consolidation.json"
        sidecar.write_text("not json {{{")
        state = ss.load_state(tmp_path, "telegram:55")
        assert state.summarized_through == 0
        assert state.summary == ""

    def test_delete_state_is_noop_when_absent(self, tmp_path: Path) -> None:
        """``delete_state`` for a session with no sidecar must not
        raise — used by ``DefaultConversation.clear`` which doesn't
        know whether a sidecar was ever written."""
        ss.delete_state(tmp_path, "ut:never-existed")  # no exception

    def test_delete_state_removes_existing_sidecar(self, tmp_path: Path) -> None:
        ss.save_state(tmp_path, "ut:to-delete", ss.ConsolidationState(summarized_through=5))
        path = tmp_path / "ut_to-delete.consolidation.json"
        assert path.exists()
        ss.delete_state(tmp_path, "ut:to-delete")
        assert not path.exists()


@pytest.mark.asyncio
class TestDefaultSessionReader:
    """``HistoryStore.reader`` ships a default fallback impl
    (``_DefaultSessionReader``) for backends that don't override it.
    Production ``SessionManager`` overrides with its own line-by-line
    streaming reader, so the fallback is only exercised when a
    custom backend doesn't bother. Verify the default contract."""

    async def test_stream_count_at_round_trip(self) -> None:
        store = MagicMock(spec=["load_range"])
        msgs = [{"role": "user", "content": str(i)} for i in range(5)]

        def _load_range(_key: str, start: int, end: int) -> list[dict[str, Any]]:
            return msgs[start:end]

        store.load_range = _load_range
        reader = _DefaultSessionReader(store, "ut:default")
        assert reader.key == "ut:default"
        assert await reader.count() == 5
        assert (await reader.at(2))["content"] == "2"
        assert await reader.at(99) is None
        streamed = [m async for m in reader.stream()]
        assert [m["content"] for m in streamed] == [str(i) for i in range(5)]
        sliced = [m async for m in reader.stream(start=1, end=3)]
        assert [m["content"] for m in sliced] == ["1", "2"]
