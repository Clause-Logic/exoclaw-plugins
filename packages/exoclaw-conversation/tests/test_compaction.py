"""Tests for context compaction and overflow recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from exoclaw_conversation.context import (
    _COMPACTION_MARKER,
    _RECOVERY_HARD_CLEAR_MARKER,
    _RECOVERY_SUMMARY_PREFIX,
    _estimate_tokens,
    compact_tool_results,
    drop_oldest_half,
    summarize_old_chunks,
    truncate_oldest_tool_results,
)


def _msg(role: str, content: str, **kw: object) -> dict[str, object]:
    m: dict[str, object] = {"role": role, "content": content}
    m.update(kw)
    return m


def _tool_result(tool_call_id: str, name: str, content: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


def _assistant_with_tool_calls(*ids: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "calling tools",
        "tool_calls": [
            {"id": tc_id, "type": "function", "function": {"name": "test", "arguments": "{}"}}
            for tc_id in ids
        ],
    }


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_basic() -> None:
    msgs = [_msg("user", "hello world")]  # 11 chars -> ~3 tokens
    assert _estimate_tokens(msgs) > 0


def test_estimate_tokens_counts_tool_calls() -> None:
    msgs = [_assistant_with_tool_calls("tc1")]
    assert _estimate_tokens(msgs) > 0


# ---------------------------------------------------------------------------
# compact_tool_results
# ---------------------------------------------------------------------------


def test_compact_under_budget_is_noop() -> None:
    msgs = [
        _msg("system", "you are helpful"),
        _msg("user", "hi"),
        _tool_result("tc1", "test", "short result"),
        _msg("assistant", "done"),
    ]
    result = compact_tool_results(msgs, context_window=100_000)
    assert result == msgs


def test_compact_replaces_old_tool_results() -> None:
    big = "x" * 10_000
    msgs = [
        _msg("system", "sys"),
        _assistant_with_tool_calls("tc1"),
        _tool_result("tc1", "web_fetch", big),  # old, should compact
        _assistant_with_tool_calls("tc2"),
        _tool_result("tc2", "web_fetch", big),  # old, should compact
        _msg("user", "now what?"),
        _assistant_with_tool_calls("tc3"),
        _tool_result("tc3", "read_file", "recent result"),  # recent, keep
        _msg("assistant", "here's what I found"),
    ]
    # Set context window small enough to trigger compaction of both
    result = compact_tool_results(msgs, context_window=1_000)

    # Old tool results should be compacted
    assert result[2]["content"] == _COMPACTION_MARKER
    assert result[4]["content"] == _COMPACTION_MARKER

    # Recent tool result should be preserved
    assert result[7]["content"] == "recent result"


def test_compact_skips_last_4_messages() -> None:
    big = "x" * 10_000
    msgs = [
        _msg("system", "sys"),
        _assistant_with_tool_calls("tc1"),
        _tool_result("tc1", "test", big),  # within last 4 non-system
        _msg("assistant", "done"),
    ]
    result = compact_tool_results(msgs, context_window=1_000)
    # Even though over budget, the tool result is in the last 4 messages
    assert result[2]["content"] == big


def test_compact_skips_already_compacted() -> None:
    msgs = [
        _msg("system", "sys"),
        _tool_result("tc1", "test", _COMPACTION_MARKER),
        _msg("user", "hi"),
        _msg("assistant", "hey"),
    ]
    result = compact_tool_results(msgs, context_window=100)
    assert result[1]["content"] == _COMPACTION_MARKER


def test_compact_skips_short_results() -> None:
    msgs = [
        _msg("system", "sys"),
        _tool_result("tc1", "test", "ok"),  # too short to compact (<100 chars)
        _msg("user", "hi" * 5000),
        _msg("assistant", "hey"),
    ]
    result = compact_tool_results(msgs, context_window=1_000)
    assert result[1]["content"] == "ok"


# ---------------------------------------------------------------------------
# drop_oldest_half
# ---------------------------------------------------------------------------


def test_drop_oldest_half_keeps_system() -> None:
    msgs = [
        _msg("system", "you are helpful"),
        _msg("user", "msg1"),
        _msg("assistant", "resp1"),
        _msg("user", "msg2"),
        _msg("assistant", "resp2"),
        _msg("user", "msg3"),
        _msg("assistant", "resp3"),
    ]
    result = drop_oldest_half(msgs)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "you are helpful"
    # Should keep roughly the last half
    assert len(result) < len(msgs)


def test_drop_oldest_half_repairs_orphaned_tool_results() -> None:
    msgs = [
        _msg("system", "sys"),
        _assistant_with_tool_calls("tc1"),  # will be dropped
        _tool_result("tc1", "test", "result1"),  # orphaned after drop
        _msg("user", "msg2"),
        _assistant_with_tool_calls("tc2"),
        _tool_result("tc2", "test", "result2"),  # has parent, kept
        _msg("assistant", "done"),
    ]
    result = drop_oldest_half(msgs)
    # tc1's tool result should be dropped (orphaned)
    tool_results = [m for m in result if m.get("role") == "tool"]
    for tr in tool_results:
        assert tr.get("tool_call_id") != "tc1"


def test_drop_oldest_half_empty() -> None:
    msgs = [_msg("system", "sys")]
    result = drop_oldest_half(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "system"


# ---------------------------------------------------------------------------
# truncate_oldest_tool_results — recovery-time hard-clear
# ---------------------------------------------------------------------------


def test_truncate_under_budget_is_noop() -> None:
    msgs = [
        _msg("system", "sys"),
        _msg("user", "hi"),
        _tool_result("tc1", "test", "short"),
    ]
    result, cleared = truncate_oldest_tool_results(msgs, target_tokens=100_000)
    assert result == msgs
    assert cleared == 0


def test_truncate_clears_oldest_tool_results() -> None:
    big = "x" * 10_000
    msgs = [
        _msg("system", "sys"),
        _assistant_with_tool_calls("tc1"),
        _tool_result("tc1", "test", big),  # oldest tool result
        _assistant_with_tool_calls("tc2"),
        _tool_result("tc2", "test", big),  # second oldest
        _msg("user", "now what?"),
    ]
    result, cleared = truncate_oldest_tool_results(msgs, target_tokens=500)
    assert cleared >= 1
    # First tool result should be cleared
    assert result[2]["content"] == _RECOVERY_HARD_CLEAR_MARKER


def test_truncate_does_not_protect_recent_unlike_compact() -> None:
    """Recovery-time truncation can touch the most-recent tool results,
    unlike compact_tool_results which protects the last 4."""
    big = "x" * 10_000
    msgs = [
        _msg("system", "sys"),
        _assistant_with_tool_calls("tc1"),
        _tool_result("tc1", "test", big),  # within last-4, but truncate ignores that
    ]
    result, cleared = truncate_oldest_tool_results(msgs, target_tokens=100)
    assert cleared == 1
    assert result[2]["content"] == _RECOVERY_HARD_CLEAR_MARKER


def test_truncate_skips_already_cleared() -> None:
    msgs = [
        _msg("system", "sys"),
        _tool_result("tc1", "test", _RECOVERY_HARD_CLEAR_MARKER),
        _tool_result("tc2", "test", _COMPACTION_MARKER),
        _msg("user", "x" * 100_000),  # over budget but no tool results to clear
    ]
    result, cleared = truncate_oldest_tool_results(msgs, target_tokens=100)
    assert cleared == 0  # nothing eligible


def test_truncate_returns_zero_when_nothing_eligible() -> None:
    """When nothing can be cleared, caller knows recovery can't progress."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "x" * 100_000),  # huge user message, no tool results
    ]
    result, cleared = truncate_oldest_tool_results(msgs, target_tokens=100)
    assert cleared == 0


# ---------------------------------------------------------------------------
# summarize_old_chunks — recovery-time LLM summarization
# ---------------------------------------------------------------------------


_T = TypeVar("_T")


def _run(coro: Awaitable[_T]) -> _T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_summarize_under_budget_is_noop() -> None:
    msgs = [_msg("system", "sys"), _msg("user", "hi")]

    async def summarizer(_: list[dict[str, object]]) -> str:
        raise AssertionError("should not be called when under budget")

    result, did = _run(summarize_old_chunks(msgs, target_tokens=100_000, summarizer=summarizer))
    assert result == msgs
    assert did is False


def test_summarize_replaces_old_chunk() -> None:
    big = "x" * 10_000
    msgs = [
        _msg("system", "sys"),
        _msg("user", big),
        _msg("assistant", big),
        _msg("user", "older still"),
        _msg("assistant", "older still"),
        _msg("user", "recent 1"),
        _msg("assistant", "recent 2"),
        _msg("user", "recent 3"),
        _msg("assistant", "recent 4"),
    ]
    calls: list[int] = []

    async def summarizer(chunk: list[dict[str, object]]) -> str:
        calls.append(len(chunk))
        return "USER ASKED ABOUT X; ASSISTANT RESPONDED Y"

    result, did = _run(summarize_old_chunks(msgs, target_tokens=500, summarizer=summarizer))
    assert did is True
    assert calls, "summarizer must be invoked"
    # System preserved
    assert result[0]["role"] == "system"
    # Summary message inserted
    summary_msgs = [
        m
        for m in result
        if isinstance(m.get("content"), str) and m["content"].startswith(_RECOVERY_SUMMARY_PREFIX)
    ]
    assert len(summary_msgs) == 1
    # Last 4 non-system messages preserved
    non_system = [m for m in result if m.get("role") != "system"]
    assert non_system[-1]["content"] == "recent 4"
    assert non_system[-4]["content"] == "recent 1"


def test_summarize_skips_when_only_recent_messages() -> None:
    msgs = [
        _msg("system", "sys"),
        _msg("user", "hi"),
        _msg("assistant", "hello"),
    ]

    async def summarizer(_: list[dict[str, object]]) -> str:
        raise AssertionError("should not be called when nothing is eligible")

    result, did = _run(
        summarize_old_chunks(msgs, target_tokens=10, summarizer=summarizer, keep_recent=4)
    )
    assert did is False
    assert result == msgs


def test_summarize_repairs_orphaned_tool_results() -> None:
    """If an assistant tool_call gets summarized, its dangling tool result must be dropped."""
    big = "x" * 5_000
    msgs = [
        _msg("system", "sys"),
        _assistant_with_tool_calls("old-tc"),  # gets summarized
        _tool_result("old-tc", "test", big),  # gets summarized
        _msg("user", "older 1"),
        _msg("assistant", "older 2"),
        _msg("user", "recent 1"),
        _msg("assistant", "recent 2"),
        _msg("user", "recent 3"),
        _assistant_with_tool_calls("recent-tc"),  # parent in tail
        _tool_result("recent-tc", "test", "ok"),  # has parent in tail
    ]

    async def summarizer(_: list[dict[str, object]]) -> str:
        return "summary"

    result, did = _run(
        summarize_old_chunks(msgs, target_tokens=200, summarizer=summarizer, keep_recent=5)
    )
    assert did is True
    # No orphan with old-tc id
    for m in result:
        if m.get("role") == "tool":
            assert m.get("tool_call_id") != "old-tc"
    # recent-tc tool result preserved
    recent_tool = [
        m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "recent-tc"
    ]
    assert len(recent_tool) == 1


def test_summarize_returns_unchanged_when_summarizer_returns_empty() -> None:
    big = "x" * 10_000
    msgs = [
        _msg("system", "sys"),
        _msg("user", big),
        _msg("assistant", big),
        _msg("user", "recent 1"),
        _msg("assistant", "recent 2"),
        _msg("user", "recent 3"),
        _msg("assistant", "recent 4"),
    ]

    async def empty_summarizer(_: list[dict[str, object]]) -> str:
        return ""

    result, did = _run(summarize_old_chunks(msgs, target_tokens=200, summarizer=empty_summarizer))
    assert did is False
    assert result == msgs
