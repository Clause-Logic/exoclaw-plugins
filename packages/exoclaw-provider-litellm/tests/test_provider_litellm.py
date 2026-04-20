"""Tests for exoclaw-provider-litellm package."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from exoclaw.providers.types import LLMResponse
from exoclaw_provider_litellm.provider import (
    _ALLOWED_MSG_KEYS,
    LiteLLMProvider,
    _apply_anthropic_cache_control_to_system,
    _apply_anthropic_cache_control_to_tools,
    _is_anthropic,
    _normalize_tool_call_id,
    _sanitize_empty_content,
    _sanitize_request_messages,
    _short_tool_id,
)

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
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "hello"},
                ],
            }
        ]
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
# _apply_anthropic_cache_control_to_system
# ---------------------------------------------------------------------------


class TestApplyAnthropicCacheControlToSystem:
    def test_string_content_converted_to_list_with_cache_control(self) -> None:
        messages = [{"role": "system", "content": "You are helpful."}]
        result = _apply_anthropic_cache_control_to_system(messages)
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["type"] == "text"
        assert sys_content[0]["text"] == "You are helpful."
        assert sys_content[0]["cache_control"] == {"type": "ephemeral"}

    def test_list_content_stamps_last_block(self) -> None:
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ],
            }
        ]
        result = _apply_anthropic_cache_control_to_system(messages)
        sys_content = result[0]["content"]
        assert "cache_control" not in sys_content[0]
        assert sys_content[1]["cache_control"] == {"type": "ephemeral"}

    def test_non_system_messages_unchanged(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _apply_anthropic_cache_control_to_system(messages)
        assert result == messages

    def test_no_system_message(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        result = _apply_anthropic_cache_control_to_system(messages)
        assert result == messages

    def test_original_not_mutated(self) -> None:
        messages = [{"role": "system", "content": "stable prompt"}]
        _apply_anthropic_cache_control_to_system(messages)
        assert isinstance(messages[0]["content"], str)


# ---------------------------------------------------------------------------
# _apply_anthropic_cache_control_to_tools
# ---------------------------------------------------------------------------


class TestApplyAnthropicCacheControlToTools:
    def test_stamps_last_tool(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "tool_a"}},
            {"type": "function", "function": {"name": "tool_b"}},
        ]
        result = _apply_anthropic_cache_control_to_tools(tools)
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_single_tool(self) -> None:
        tools = [{"type": "function", "function": {"name": "tool_a"}}]
        result = _apply_anthropic_cache_control_to_tools(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_empty_tools(self) -> None:
        assert _apply_anthropic_cache_control_to_tools([]) == []

    def test_original_not_mutated(self) -> None:
        tools = [{"type": "function", "function": {"name": "tool_a"}}]
        _apply_anthropic_cache_control_to_tools(tools)
        assert "cache_control" not in tools[0]


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

    async def test_chat_exception_propagates(self) -> None:
        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("network error")
            with pytest.raises(Exception, match="network error"):
                await p.chat([{"role": "user", "content": "hi"}])

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


class TestLiteLLMProviderRouter:
    """Router path: when a ``litellm.Router`` is supplied, chat dispatches
    through ``router.acompletion`` instead of module-level ``acompletion``,
    and per-provider auth fields are stripped so the router's own
    deployment params win."""

    async def test_chat_routes_through_router(self) -> None:
        router = MagicMock()
        router.acompletion = AsyncMock(return_value=_make_litellm_response("via-router"))
        p = LiteLLMProvider(router=router)
        with patch(
            "exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            result = await p.chat(
                [{"role": "user", "content": "hi"}],
                model="some-group",
            )
        assert result.content == "via-router"
        mock_acompletion.assert_not_called()
        router.acompletion.assert_awaited_once()
        assert router.acompletion.await_args is not None
        kwargs = router.acompletion.await_args.kwargs
        assert kwargs["model"] == "some-group"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    async def test_router_call_strips_provider_defaults(self) -> None:
        """Router deployments carry their own api_key/api_base. When the
        provider was constructed with fallback values, they must NOT shadow
        what the router's ``model_list`` says for a given deployment."""
        router = MagicMock()
        router.acompletion = AsyncMock(return_value=_make_litellm_response("ok"))
        p = LiteLLMProvider(
            api_key="fallback-key",
            api_base="https://fallback.example",
            extra_headers={"X-Fallback": "1"},
            router=router,
        )
        await p.chat([{"role": "user", "content": "hi"}], model="g")
        assert router.acompletion.await_args is not None
        kwargs = router.acompletion.await_args.kwargs
        assert "api_key" not in kwargs
        assert "api_base" not in kwargs
        assert "extra_headers" not in kwargs

    async def test_router_absent_falls_back_to_acompletion(self) -> None:
        p = LiteLLMProvider()
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response("direct")
            result = await p.chat([{"role": "user", "content": "hi"}])
        assert result.content == "direct"
        mock.assert_awaited_once()

    async def test_streaming_uses_router_and_strips_auth(self) -> None:
        """The streaming path (``_stream_to_completion``) must route through
        the router when one is configured and must not leak provider-level
        ``api_key`` / ``api_base`` / ``extra_headers`` into the router call.
        """

        # ``litellm.stream_chunk_builder`` normally reassembles a chunked
        # stream into a non-streaming response. Patch it to return a canned
        # completion so the test doesn't depend on streaming internals.
        sentinel_chunk = MagicMock()

        async def _fake_stream() -> Any:
            yield sentinel_chunk

        router = MagicMock()
        router.acompletion = AsyncMock(return_value=_fake_stream())

        p = LiteLLMProvider(
            api_key="fallback-key",
            api_base="https://fallback.example",
            extra_headers={"X-Fallback": "1"},
            stream=True,
            router=router,
        )

        with (
            patch(
                "exoclaw_provider_litellm.provider.acompletion",
                new_callable=AsyncMock,
            ) as mock_acompletion,
            patch(
                "exoclaw_provider_litellm.provider.litellm.stream_chunk_builder",
                return_value=_make_litellm_response("streamed"),
            ),
        ):
            result = await p.chat(
                [{"role": "user", "content": "hi"}],
                model="some-group",
            )

        assert result.content == "streamed"
        mock_acompletion.assert_not_called()
        router.acompletion.assert_awaited_once()
        assert router.acompletion.await_args is not None
        kwargs = router.acompletion.await_args.kwargs
        assert kwargs["model"] == "some-group"
        assert kwargs["stream"] is True
        assert "api_key" not in kwargs
        assert "api_base" not in kwargs
        assert "extra_headers" not in kwargs


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
            with pytest.raises(IndexError):
                await p.chat([{"role": "user", "content": "hi"}])

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
        msgs = [
            {"role": "user", "content": "hi", "timestamp": "2024", "reasoning_content": "thoughts"}
        ]
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

    async def test_chat_anthropic_injects_cache_control_on_system(self) -> None:
        p = LiteLLMProvider()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response()
            await p.chat(msgs, model="claude-opus-4-5")
        sent_msgs = mock.call_args[1]["messages"]
        sys_msg = next(m for m in sent_msgs if m["role"] == "system")
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    async def test_chat_anthropic_injects_cache_control_on_tools(self) -> None:
        p = LiteLLMProvider()
        tools = [{"type": "function", "function": {"name": "exec"}}]
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response()
            await p.chat([{"role": "user", "content": "hi"}], model="claude-opus-4-5", tools=tools)
        sent_tools = mock.call_args[1]["tools"]
        assert sent_tools[-1]["cache_control"] == {"type": "ephemeral"}

    async def test_chat_non_anthropic_no_cache_control(self) -> None:
        p = LiteLLMProvider()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        with patch("exoclaw_provider_litellm.provider.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_litellm_response()
            await p.chat(msgs, model="gpt-4o")
        sent_msgs = mock.call_args[1]["messages"]
        sys_msg = next(m for m in sent_msgs if m["role"] == "system")
        assert isinstance(sys_msg["content"], str)


class TestModelConcurrency:
    def test_init_builds_semaphores(self) -> None:
        p = LiteLLMProvider(model_max_concurrent={"gpt-4o": 2, "claude": 5})
        assert set(p._model_semaphores) == {"gpt-4o", "claude"}

    def test_init_skips_non_positive(self) -> None:
        p = LiteLLMProvider(model_max_concurrent={"a": 0, "b": -1, "c": 3})
        assert set(p._model_semaphores) == {"c"}

    def test_init_no_semaphores_by_default(self) -> None:
        assert LiteLLMProvider()._model_semaphores == {}

    async def test_chat_limits_concurrency_for_configured_model(self) -> None:
        p = LiteLLMProvider(model_max_concurrent={"gpt-4o": 2})
        in_flight = 0
        peak = 0
        gate = asyncio.Event()

        async def fake_acompletion(**_: Any) -> Any:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await gate.wait()
            in_flight -= 1
            return _make_litellm_response("ok")

        with patch(
            "exoclaw_provider_litellm.provider.acompletion",
            side_effect=fake_acompletion,
        ):
            tasks = [
                asyncio.create_task(p.chat([{"role": "user", "content": "hi"}], model="gpt-4o"))
                for _ in range(5)
            ]
            # Yield enough times for the first wave to reach the gate.
            for _ in range(10):
                await asyncio.sleep(0)
            gate.set()
            await asyncio.gather(*tasks)

        assert peak == 2

    async def test_chat_unlimited_for_unconfigured_model(self) -> None:
        p = LiteLLMProvider(model_max_concurrent={"gpt-4o": 1})
        in_flight = 0
        peak = 0
        gate = asyncio.Event()

        async def fake_acompletion(**_: Any) -> Any:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await gate.wait()
            in_flight -= 1
            return _make_litellm_response("ok")

        with patch(
            "exoclaw_provider_litellm.provider.acompletion",
            side_effect=fake_acompletion,
        ):
            tasks = [
                asyncio.create_task(
                    p.chat([{"role": "user", "content": "hi"}], model="claude-opus-4-5")
                )
                for _ in range(4)
            ]
            for _ in range(10):
                await asyncio.sleep(0)
            observed_peak = peak
            gate.set()
            await asyncio.gather(*tasks)

        assert observed_peak == 4
