"""Tests for context compaction and overflow recovery."""

from __future__ import annotations

import pytest

from exoclaw_conversation.context import (
    _COMPACTION_MARKER,
    _estimate_tokens,
    compact_tool_results,
    drop_oldest_half,
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
