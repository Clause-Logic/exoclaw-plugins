"""Tests for ``run_turn``.

We mock the LLMProvider so the tests are hermetic — no network. The
goal is to verify that ``run_turn`` correctly assembles an AgentLoop
and that the result shape matches the contract.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from exoclaw.providers.types import LLMResponse, ToolCallRequest
from exoclaw_turn import TurnResult, run_turn
from exoclaw_turn.turn import _EphemeralConversation


class _ScriptedProvider:
    """Returns a queued sequence of ``LLMResponse`` objects per chat call.

    Captures the ``messages`` arg of each call so tests can assert on
    the prompt the loop built.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        response_format: Any = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "model": model,
            }
        )
        if not self._responses:
            return LLMResponse(content="(no more scripted responses)")
        return self._responses.pop(0)


class _EchoTool:
    """Minimal Tool — echoes its ``text`` argument back."""

    name = "echo"
    description = "Echo back the input text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"echo:{kwargs.get('text', '')}"


class TestEphemeralConversation:
    async def test_build_prompt_with_system(self) -> None:
        conv = _EphemeralConversation(system="You are helpful.")
        msgs = await conv.build_prompt("sid", "hi")
        assert msgs == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]

    async def test_build_prompt_without_system(self) -> None:
        conv = _EphemeralConversation(system=None)
        msgs = await conv.build_prompt("sid", "hi")
        assert msgs == [{"role": "user", "content": "hi"}]

    async def test_build_prompt_merges_plugin_context_into_system(self) -> None:
        conv = _EphemeralConversation(system="Base.")
        msgs = await conv.build_prompt("sid", "hi", plugin_context=["Extra A", "Extra B"])
        assert msgs[0]["content"] == "Base.\n\nExtra A\n\nExtra B"
        assert msgs[1] == {"role": "user", "content": "hi"}

    async def test_build_prompt_uses_plugin_context_as_system_when_no_system(self) -> None:
        conv = _EphemeralConversation(system=None)
        msgs = await conv.build_prompt("sid", "hi", plugin_context=["Ctx"])
        assert msgs == [
            {"role": "system", "content": "Ctx"},
            {"role": "user", "content": "hi"},
        ]


class TestRunTurnBasic:
    async def test_returns_text_when_provider_replies_directly(self) -> None:
        provider = _ScriptedProvider([LLMResponse(content="Hello back.")])
        result = await run_turn(
            provider=provider,
            message="Hello.",
            system="Be brief.",
        )

        assert isinstance(result, TurnResult)
        assert result.text == "Hello back."
        assert result.tool_calls == []
        assert len(provider.calls) == 1
        sent = provider.calls[0]["messages"]
        assert sent[0] == {"role": "system", "content": "Be brief."}
        assert sent[1] == {"role": "user", "content": "Hello."}

    async def test_messages_contain_user_and_assistant(self) -> None:
        provider = _ScriptedProvider([LLMResponse(content="Reply.")])
        result = await run_turn(provider=provider, message="Q")

        roles = [m.get("role") for m in result.messages]
        assert "user" in roles
        assert "assistant" in roles

    async def test_model_override_forwarded_to_provider(self) -> None:
        provider = _ScriptedProvider([LLMResponse(content="ok")])
        await run_turn(provider=provider, message="hi", model="other-model")
        assert provider.calls[0]["model"] == "other-model"


class TestRunTurnTools:
    async def test_tool_called_then_text_returned(self) -> None:
        provider = _ScriptedProvider(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="call_1",
                            name="echo",
                            arguments={"text": "hi"},
                        ),
                    ],
                ),
                LLMResponse(content="Done."),
            ]
        )
        echo = _EchoTool()
        result = await run_turn(
            provider=provider,
            message="Echo 'hi' then say done.",
            tools=[echo],
        )

        assert result.text == "Done."
        assert len(echo.calls) == 1
        assert echo.calls[0] == {"text": "hi"}

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "echo"
        assert result.tool_calls[0].arguments == {"text": "hi"}

        tool_result_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert any("echo:hi" in str(m.get("content", "")) for m in tool_result_msgs)

    async def test_tools_advertised_to_provider(self) -> None:
        provider = _ScriptedProvider([LLMResponse(content="ok")])
        await run_turn(provider=provider, message="hi", tools=[_EchoTool()])

        first_call_tools = provider.calls[0]["tools"]
        assert first_call_tools is not None
        names = [
            t.get("function", {}).get("name") if isinstance(t, dict) else None
            for t in first_call_tools
        ]
        assert "echo" in names


class TestExtractToolCalls:
    async def test_extracts_with_json_string_arguments(self) -> None:
        """The loop persists tool calls with arguments as a JSON string."""
        provider = _ScriptedProvider(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="c1",
                            name="echo",
                            arguments={"text": "hello world"},
                        ),
                    ],
                ),
                LLMResponse(content="done"),
            ]
        )
        result = await run_turn(provider=provider, message="x", tools=[_EchoTool()])

        # The persisted assistant message should carry tool_calls with a
        # JSON-string ``arguments`` payload (that's what the loop writes).
        assistant_msgs = [
            m for m in result.messages if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert assistant_msgs
        raw_args = assistant_msgs[0]["tool_calls"][0]["function"]["arguments"]
        # Should round-trip back to the original dict in TurnResult.tool_calls.
        assert json.loads(raw_args) == {"text": "hello world"}
        assert result.tool_calls[0].arguments == {"text": "hello world"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
