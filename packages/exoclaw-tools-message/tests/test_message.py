"""Tests for exoclaw-tools-message package."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from exoclaw.bus.events import OutboundMessage
from exoclaw_tools_message.tool import MessageTool, filter_suppressed


# ---------------------------------------------------------------------------
# filter_suppressed
# ---------------------------------------------------------------------------


class TestFilterSuppressed:
    def test_no_patterns_returns_content(self) -> None:
        assert filter_suppressed("hello world", []) == "hello world"

    def test_matching_line_removed(self) -> None:
        result = filter_suppressed("hello\nthinking...\nworld", ["thinking"])
        assert result == "hello\nworld"

    def test_all_lines_suppressed_returns_none(self) -> None:
        result = filter_suppressed("thinking...", ["thinking"])
        assert result is None

    def test_case_insensitive(self) -> None:
        result = filter_suppressed("THINKING aloud", ["thinking"])
        assert result is None

    def test_blank_lines_kept(self) -> None:
        result = filter_suppressed("hello\n\nworld", ["xxx"])
        assert result == "hello\n\nworld"

    def test_blank_only_content_returns_none(self) -> None:
        result = filter_suppressed("   \n  \n", ["xxx"])
        assert result is None

    def test_invalid_regex_pattern_skipped(self) -> None:
        # Should not raise; invalid pattern is skipped
        result = filter_suppressed("hello", ["[invalid"])
        assert result == "hello"

    def test_multiple_patterns_any_match_suppresses(self) -> None:
        result = filter_suppressed("thinking step 1", ["thinking", "step"])
        assert result is None

    def test_partial_content_kept(self) -> None:
        result = filter_suppressed("keep this\nthinking...\nkeep this too", ["thinking"])
        assert "keep this" in result  # type: ignore[operator]
        assert "thinking" not in result  # type: ignore[operator]


# ---------------------------------------------------------------------------
# MessageTool
# ---------------------------------------------------------------------------


@pytest.fixture
def send_cb() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def tool(send_cb: AsyncMock) -> MessageTool:
    t = MessageTool(
        send_callback=send_cb,
        default_channel="cli",
        default_chat_id="user1",
    )
    return t


class TestMessageToolProperties:
    def test_name(self, tool: MessageTool) -> None:
        assert tool.name == "message"

    def test_description(self, tool: MessageTool) -> None:
        assert "message" in tool.description.lower()

    def test_parameters(self, tool: MessageTool) -> None:
        p = tool.parameters
        assert p["type"] == "object"
        assert "content" in p["properties"]
        assert p["required"] == ["content"]


class TestMessageToolSetContext:
    def test_set_context(self, tool: MessageTool) -> None:
        tool.set_context("telegram", "chat123", message_id="msg1")
        assert tool._default_channel == "telegram"
        assert tool._default_chat_id == "chat123"
        assert tool._default_message_id == "msg1"

    def test_set_send_callback(self) -> None:
        t = MessageTool()
        cb = AsyncMock()
        t.set_send_callback(cb)
        assert t._send_callback is cb

    def test_start_turn_resets(self, tool: MessageTool) -> None:
        tool._sent_in_turn = True
        tool.start_turn()
        assert tool._sent_in_turn is False


class TestMessageToolExecute:
    async def test_sends_message(self, tool: MessageTool, send_cb: AsyncMock) -> None:
        result = await tool.execute(content="hello")
        assert "Message sent" in result
        send_cb.assert_called_once()

    async def test_marks_sent_in_turn(self, tool: MessageTool) -> None:
        await tool.execute(content="hi")
        assert tool._sent_in_turn is True

    async def test_different_channel_does_not_mark_sent(self, tool: MessageTool) -> None:
        result = await tool.execute(content="hi", channel="telegram", chat_id="other")
        assert "Message sent" in result
        assert tool._sent_in_turn is False

    async def test_no_channel_error(self) -> None:
        t = MessageTool(send_callback=AsyncMock())
        result = await t.execute(content="hi")
        assert "Error" in result

    async def test_no_callback_error(self) -> None:
        t = MessageTool(default_channel="cli", default_chat_id="u1")
        result = await t.execute(content="hi")
        assert "Error" in result

    async def test_explicit_channel_and_chat_id(self, tool: MessageTool, send_cb: AsyncMock) -> None:
        result = await tool.execute(content="hi", channel="slack", chat_id="C123")
        assert "slack:C123" in result

    async def test_with_media(self, tool: MessageTool, send_cb: AsyncMock) -> None:
        result = await tool.execute(content="hi", media=["/tmp/a.png", "/tmp/b.png"])
        assert "2 attachments" in result

    async def test_send_callback_exception(self, tool: MessageTool, send_cb: AsyncMock) -> None:
        send_cb.side_effect = RuntimeError("network error")
        result = await tool.execute(content="hi")
        assert "Error sending message" in result

    async def test_uses_default_message_id(self, tool: MessageTool, send_cb: AsyncMock) -> None:
        tool.set_context("cli", "user1", message_id="orig_msg")
        await tool.execute(content="reply")
        call_args = send_cb.call_args[0][0]
        assert isinstance(call_args, OutboundMessage)
        assert call_args.metadata["message_id"] == "orig_msg"

    async def test_explicit_message_id_overrides(self, tool: MessageTool, send_cb: AsyncMock) -> None:
        tool.set_context("cli", "user1", message_id="orig")
        await tool.execute(content="reply", message_id="override")
        call_args = send_cb.call_args[0][0]
        assert call_args.metadata["message_id"] == "override"


class TestMessageToolSuppressPatterns:
    async def test_suppress_matching_content(self, send_cb: AsyncMock) -> None:
        t = MessageTool(
            send_callback=send_cb,
            default_channel="cli",
            default_chat_id="u1",
            suppress_patterns=["thinking"],
        )
        result = await t.execute(content="thinking aloud")
        assert "suppressed" in result
        send_cb.assert_not_called()

    async def test_suppress_with_retrieval_fn(self, send_cb: AsyncMock) -> None:
        async def retrieval(original: str) -> str | None:
            return "relevant memory"

        t = MessageTool(
            send_callback=send_cb,
            default_channel="cli",
            default_chat_id="u1",
            suppress_patterns=["thinking"],
            retrieval_fn=retrieval,
        )
        result = await t.execute(content="thinking aloud")
        assert "suppressed" in result
        assert "relevant memory" in result

    async def test_suppress_retrieval_returns_none(self, send_cb: AsyncMock) -> None:
        async def retrieval(original: str) -> str | None:
            return None

        t = MessageTool(
            send_callback=send_cb,
            default_channel="cli",
            default_chat_id="u1",
            suppress_patterns=["thinking"],
            retrieval_fn=retrieval,
        )
        result = await t.execute(content="thinking aloud")
        assert "suppressed" in result

    async def test_non_matching_content_sent(self, send_cb: AsyncMock) -> None:
        t = MessageTool(
            send_callback=send_cb,
            default_channel="cli",
            default_chat_id="u1",
            suppress_patterns=["thinking"],
        )
        result = await t.execute(content="hello world")
        assert "Message sent" in result
        send_cb.assert_called_once()

    async def test_partial_suppression_sends_filtered(self, send_cb: AsyncMock) -> None:
        t = MessageTool(
            send_callback=send_cb,
            default_channel="cli",
            default_chat_id="u1",
            suppress_patterns=["thinking"],
        )
        result = await t.execute(content="thinking aloud\nhello world")
        assert "Message sent" in result
        call_args = send_cb.call_args[0][0]
        assert "hello world" in call_args.content
        assert "thinking" not in call_args.content
