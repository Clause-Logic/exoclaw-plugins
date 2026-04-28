"""``WebSearchTool`` integration tests — drive a fake LLM provider,
verify the query routes through ``provider.chat`` with the
configured search-dedicated model."""

from __future__ import annotations

from typing import Any

import pytest
from exoclaw.providers.types import LLMResponse
from exoclaw_tools_web import WebSearchTool


class _FakeProvider:
    def __init__(self, response: str = "grounded answer") -> None:
        self._response = response
        self.last_call: dict[str, Any] = {}
        self.fail_with: Exception | None = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        response_format: Any = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.last_call = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
        }
        if self.fail_with is not None:
            raise self.fail_with
        return LLMResponse(content=self._response, tool_calls=[])

    def get_default_model(self) -> str:
        return "fake/default"

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_search_routes_query_to_provider_with_model() -> None:
    provider = _FakeProvider(response="The answer is 42 [src](https://x.test).")
    tool = WebSearchTool(
        provider=provider,
        model="openai/gemini-flash:search",
        max_tokens=512,
    )

    out = await tool.execute(query="what is the answer")

    assert "42" in out
    assert provider.last_call["model"] == "openai/gemini-flash:search"
    assert provider.last_call["max_tokens"] == 512
    assert provider.last_call["messages"] == [{"role": "user", "content": "what is the answer"}]


@pytest.mark.asyncio
async def test_search_surfaces_provider_errors() -> None:
    provider = _FakeProvider()
    provider.fail_with = RuntimeError("rate limit exceeded")
    tool = WebSearchTool(provider=provider, model="x")

    out = await tool.execute(query="anything")
    assert "Error" in out
    assert "rate limit exceeded" in out


@pytest.mark.asyncio
async def test_search_handles_empty_response() -> None:
    provider = _FakeProvider(response="")
    tool = WebSearchTool(provider=provider, model="x")

    out = await tool.execute(query="hello")
    assert "empty response" in out


def test_tool_metadata_shape() -> None:
    provider = _FakeProvider()
    tool = WebSearchTool(provider=provider, model="x")
    assert tool.name == "web_search"
    params = tool.parameters
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["required"] == ["query"]


def test_skill_entry_point_returns_dict() -> None:
    from exoclaw_tools_web.skills import web

    skill = web()
    assert isinstance(skill, dict)
    assert skill["name"] == "web"
    assert "content" in skill
    assert "path" not in skill
