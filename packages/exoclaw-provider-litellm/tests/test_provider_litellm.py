"""Tests for exoclaw-provider-litellm package."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw_provider_litellm.provider import (
    LiteLLMProvider,
    _is_anthropic,
    _normalize_tool_call_id,
    _sanitize_empty_content,
    _sanitize_request_messages,
    _short_tool_id,
    _ALLOWED_MSG_KEYS,
)
from exoclaw.providers.types import LLMResponse, ToolCallRequest


# ---------------------------------------------------------------------------
# _short_tool_id
# ---------------------------------------------------------------------------

class TestShortToolId:
    def test_length(self) -> None:
        assert len(_short_tool_id()) == 9

    def test_alphanumeric(self) -> None:
        tid = _short_tool_id()
        assert tid.isalnum()

    def test_unique(self) -> None:
        assert _short_tool_id() != _short_tool_id()


# ---------------------------------------------------------------------------
# _normalize_tool_call_id
# ---------------------------------------------------------------------------

class TestNormalizeToolCallId:
    def test_already_9_alnum(self) -> None:
        tid = "abc123XYZ"
        assert _normalize_tool_call_id(tid) == tid

    def test_long_id_hashed(self) -> None:
        long_id = "call_" + "a" * 40
        result = _normalize_tool_call_id(long_id)
        assert len(result) == 9
        assert result == hashlib.sha1(long_id.encode()).hexdigest()[:9]

    def test_non_string_passthrough(self) -> None:
        assert _normalize_tool_call_id(None) is None
        assert _normalize_tool_call_id(42) == 42


# ---------------------------------------------------------------------------
# _is_anthropic
# ---------------------------------------------------------------------------

class TestIsAnthropic:
    def test_claude_model(self) -> None:
        assert _is_anthropic("claude-3-opus")
        assert _is_anthropic("claude-sonnet-4-5")

    def test_anthropic_prefix(self) -> None:
        assert _is_anthropic("anthropic/claude-opus")

    def test_openai_not_anthropic(self) -> None:
        assert not _is_anthropic("gpt-4o")
        assert not _is_anthropic("gemini-pro")


# ---------------------------------------------------------------------------
# _sanitize_empty_content
# ---------------------------------------------------------------------------

class TestSanitizeEmptyContent:
    def test_empty_string_user(self) -> None:
        msgs = [{"role": "user", "content": ""}]
        result = _sanitize_empty_content(msgs)
        assert result[0]["content"] == "(empty)"

    def test_empty_string_assistant_with_tool_calls(self) -> None:
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]
        result = _sanitize_empty_content(msgs)
        assert result[0]["content"] is None

    def test_nonempty_passthrough(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        result = _sanitize_empty_content(msgs)
        assert result[0]["content"] == "hello"

    def test_list_content_filters_empty_text(self) -> None:
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "hello"},
        ]}]
        result = _sanitize_empty_content(msgs)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["text"] == "hello"

    def test_list_content_all_empty_becomes_empty_str(self) -> None:
        msgs = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        result = _sanitize_empty_content(msgs)
        assert result[0]["content"] == "(empty)"

    def test_dict_content_wrapped_in_list(self) -> None:
        msgs = [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        result = _sanitize_empty_content(msgs)
        assert isinstance(result[0]["content"], list)

    def test_none_content_passthrough(self) -> None:
        msgs = [{"role": "assistant", "content": None, "tool_calls": []}]
        result = _sanitize_empty_content(msgs)
        assert result[0]["content"] is None


# ---------------------------------------------------------------------------
# _sanitize_request_messages
# ---------------------------------------------------------------------------

class TestSanitizeRequestMessages:
    def test_strips_unknown_keys(self) -> None:
        msgs = [{"role": "user", "content": "hi", "unknown_key": "strip_me"}]
        result = _sanitize_request_messages(msgs, _ALLOWED_MSG_KEYS)
        assert "unknown_key" not in result[0]

    def test_keeps_allowed_keys(self) -> None:
        msgs = [{"role": "user", "content": "hi", "tool_call_id": "x"}]
        result = _sanitize_request_messages(msgs, _ALLOWED_MSG_KEYS)
        assert result[0]["tool_call_id"] == "x"

    def test_adds_content_none_for_assistant(self) -> None:
        msgs = [{"role": "assistant", "tool_calls": [{"id": "1"}]}]
        result = _sanitize_request_messages(msgs, _ALLOWED_MSG_KEYS)
        assert result[0]["content"] is None


# ---------------------------------------------------------------------------
# LiteLLMProvider
# ---------------------------------------------------------------------------

def _make_litellm_response(
    content: str = "hello",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.finish_reason = finish_reason
    if tool_calls:
        choice.finish_reason = "tool_calls"
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    return response


class TestLiteLLMProvider:
    def test_init_sets_defaults(self) -> None:
        p = LiteLLMProvider()
        assert p.default_model == "anthropic/claude-opus-4-5"

    def test_init_custom_model(self) -> None:
        p = LiteLLMProvider(default_model="gpt-4o")
        assert p.default_model == "gpt-4o"

    def test_get_default_model(self) -> None:
        p = LiteLLMProvider(default_model="gpt-4o")
        assert p.get_default_model() == "gpt-4o"

    async def test_chat_basic(self) -> None:
        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response("hello world")
            result = await p.chat([{"role": "user", "content": "hi"}])
        assert isinstance(result, LLMResponse)
        assert result.content == "hello world"
        assert result.finish_reason == "stop"

    async def test_chat_with_tools(self) -> None:
        tc = MagicMock()
        tc.function.name = "exec"
        tc.function.arguments = {"command": "ls"}
        response = _make_litellm_response(tool_calls=[tc])

        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = response
            result = await p.chat(
                [{"role": "user", "content": "run ls"}],
                tools=[{"type": "function", "function": {"name": "exec"}}],
            )
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "exec"
        assert result.tool_calls[0].arguments == {"command": "ls"}

    async def test_chat_tool_arguments_as_string(self) -> None:
        import json
        tc = MagicMock()
        tc.function.name = "exec"
        tc.function.arguments = json.dumps({"command": "ls"})
        response = _make_litellm_response(tool_calls=[tc])

        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = response
            result = await p.chat([{"role": "user", "content": "run"}])
        assert result.tool_calls[0].arguments["command"] == "ls"

    async def test_chat_exception_returns_error_response(self) -> None:
        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("network error")
            result = await p.chat([{"role": "user", "content": "hi"}])
        assert result.finish_reason == "error"
        assert "network error" in (result.content or "")

    async def test_chat_with_reasoning_effort(self) -> None:
        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response()
            await p.chat([{"role": "user", "content": "hi"}], reasoning_effort="high")
        kwargs = mock.call_args[1]
        assert kwargs["reasoning_effort"] == "high"

    async def test_chat_max_tokens_floor(self) -> None:
        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response()
            await p.chat([{"role": "user", "content": "hi"}], max_tokens=0)
        kwargs = mock.call_args[1]
        assert kwargs["max_tokens"] >= 1

    async def test_chat_logging_enabled(self, capsys: Any) -> None:
        import os
        p = LiteLLMProvider()
        p._llm_logging = True
        p._llm_log_truncate = 100
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response("logged response")
            await p.chat([{"role": "user", "content": "hi"}])

    def test_sanitize_normalizes_tool_call_ids(self) -> None:
        p = LiteLLMProvider()
        msgs = [{"role": "assistant", "content": None, "tool_calls": [{"id": "a" * 50}]}]
        result = p._sanitize_messages(msgs)
        assert len(result[0]["tool_calls"][0]["id"]) == 9

    def test_sanitize_normalizes_tool_call_id_in_tool_msg(self) -> None:
        p = LiteLLMProvider()
        long_id = "a" * 50
        msgs = [{"role": "tool", "content": "result", "tool_call_id": long_id}]
        result = p._sanitize_messages(msgs)
        assert len(result[0]["tool_call_id"]) == 9

    def test_sanitize_consistent_id_mapping(self) -> None:
        p = LiteLLMProvider()
        long_id = "call_" + "x" * 40
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": long_id}]},
            {"role": "tool", "content": "result", "tool_call_id": long_id},
        ]
        result = p._sanitize_messages(msgs)
        assert result[0]["tool_calls"][0]["id"] == result[1]["tool_call_id"]

    async def test_parse_response_usage(self) -> None:
        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response()
            result = await p.chat([{"role": "user", "content": "hi"}])
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["total_tokens"] == 15


# ---------------------------------------------------------------------------
# Additional coverage: provider edge cases
# ---------------------------------------------------------------------------

class TestLiteLLMProviderExtra:
    async def test_chat_no_choices(self) -> None:
        p = LiteLLMProvider()
        mock_resp = MagicMock()
        mock_resp.choices = []
        mock_resp.usage = MagicMock()
        mock_resp.usage.prompt_tokens = 0
        mock_resp.usage.completion_tokens = 0
        mock_resp.usage.total_tokens = 0
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            result = await p.chat([{"role": "user", "content": "hi"}])
        assert result.finish_reason == "error" or result.content is None or result.content == ""

    async def test_chat_with_anthropic_model_thinking(self) -> None:
        choice = MagicMock()
        choice.message.content = [
            {"type": "thinking", "thinking": "my thoughts"},
            {"type": "text", "text": "answer"},
        ]
        choice.message.tool_calls = []
        choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [choice]
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 10

        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            result = await p.chat(
                [{"role": "user", "content": "think"}],
                model="claude-opus-4-5",
            )
        assert result.content == "answer" or result.thinking_blocks is not None

    async def test_chat_system_message_handling(self) -> None:
        p = LiteLLMProvider()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response("hello")
            result = await p.chat(msgs, model="claude-opus-4-5")
        assert result.content == "hello"

    def test_sanitize_messages_strips_unknown_top_level_keys(self) -> None:
        p = LiteLLMProvider()
        msgs = [{"role": "user", "content": "hi", "timestamp": "2024", "reasoning_content": "thoughts"}]
        result = p._sanitize_messages(msgs)
        assert "timestamp" not in result[0]
        # reasoning_content is allowed
        assert "reasoning_content" in result[0]

    async def test_chat_with_temperature_zero(self) -> None:
        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response()
            await p.chat([{"role": "user", "content": "hi"}], temperature=0.0)
        kwargs = mock.call_args[1]
        assert kwargs["temperature"] == 0.0
