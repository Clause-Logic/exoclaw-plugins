"""Tests for LLMCallTool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from exoclaw.providers.types import LLMResponse
from exoclaw_tools_llm_call import LLMCallTool


def _make_provider(response_text: str = "LLM says hello") -> AsyncMock:
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=LLMResponse(content=response_text))
    provider.get_default_model = lambda: "test-model"
    return provider


@pytest.mark.asyncio
async def test_basic_call() -> None:
    provider = _make_provider("result text")
    tool = LLMCallTool(provider=provider)

    result = await tool.execute(prompt="What is 2+2?")
    assert result == "result text"
    provider.chat.assert_called_once()
    call_args = provider.chat.call_args
    assert call_args.kwargs["messages"][0]["content"] == "What is 2+2?"


@pytest.mark.asyncio
async def test_template_vars() -> None:
    provider = _make_provider("ok")
    tool = LLMCallTool(provider=provider)

    await tool.execute(
        prompt="Hello {{ name }}, you are {{ age }}",
        vars={"name": "Stephen", "age": "30"},
    )
    call_args = provider.chat.call_args
    assert "Hello Stephen, you are 30" in call_args.kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_file_function(tmp_path: Path) -> None:
    test_file = tmp_path / "data.txt"
    test_file.write_text("file contents here")

    provider = _make_provider("processed")
    tool = LLMCallTool(provider=provider)

    await tool.execute(
        prompt="{{ file('" + str(test_file) + "') }}",
    )
    call_args = provider.chat.call_args
    assert "file contents here" in call_args.kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_file_not_found() -> None:
    provider = _make_provider("ok")
    tool = LLMCallTool(provider=provider)

    await tool.execute(prompt="{{ file('/nonexistent/path') }}")
    call_args = provider.chat.call_args
    assert "file not found" in call_args.kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_model_selection() -> None:
    provider = _make_provider("ok")
    tool = LLMCallTool(provider=provider, allowed_models=["haiku", "sonnet"])

    await tool.execute(prompt="hi", model="haiku")
    assert provider.chat.call_args.kwargs["model"] == "haiku"


@pytest.mark.asyncio
async def test_model_not_allowed() -> None:
    provider = _make_provider("ok")
    tool = LLMCallTool(provider=provider, allowed_models=["haiku"])

    result = await tool.execute(prompt="hi", model="opus")
    assert "Error" in result
    assert "not allowed" in result


@pytest.mark.asyncio
async def test_default_model() -> None:
    provider = _make_provider("ok")
    tool = LLMCallTool(provider=provider, default_model="haiku")

    await tool.execute(prompt="hi")
    assert provider.chat.call_args.kwargs["model"] == "haiku"


@pytest.mark.asyncio
async def test_output_to_file(tmp_path: Path) -> None:
    provider = _make_provider("file output content")
    tool = LLMCallTool(provider=provider)

    output_path = str(tmp_path / "out.txt")
    result = await tool.execute(prompt="hi", output=output_path)

    meta = json.loads(result)
    assert meta["output_path"] == output_path
    assert meta["chars"] == len("file output content")
    assert Path(output_path).read_text() == "file output content"


@pytest.mark.asyncio
async def test_inline_return() -> None:
    provider = _make_provider("inline result")
    tool = LLMCallTool(provider=provider)

    result = await tool.execute(prompt="hi")
    assert result == "inline result"


@pytest.mark.asyncio
async def test_template_error() -> None:
    provider = _make_provider("ok")
    tool = LLMCallTool(provider=provider)

    result = await tool.execute(prompt="{{ undefined_var }}")
    assert "Error rendering template" in result


@pytest.mark.asyncio
async def test_llm_error() -> None:
    provider = AsyncMock()
    provider.chat = AsyncMock(side_effect=RuntimeError("API down"))
    provider.get_default_model = lambda: "test"
    tool = LLMCallTool(provider=provider)

    result = await tool.execute(prompt="hi")
    assert "Error calling LLM" in result
    assert "API down" in result


@pytest.mark.asyncio
async def test_combined_with_vars_and_file(tmp_path: Path) -> None:
    data_file = tmp_path / "feeds.json"
    data_file.write_text('[{"title": "Post 1"}, {"title": "Post 2"}]')

    provider = _make_provider('[{"url": "http://example.com", "title": "Post 1"}]')
    tool = LLMCallTool(
        provider=provider,
        allowed_models=["haiku"],
        default_model="haiku",
    )

    await tool.execute(
        prompt=(
            "Feed: {{ feed_name }}\n\n"
            "Entries:\n{{ file(data_path) }}\n\n"
            "Extract interesting URLs as JSON."
        ),
        vars={"feed_name": "Simon Willison", "data_path": str(data_file)},
    )

    call_args = provider.chat.call_args
    rendered = call_args.kwargs["messages"][0]["content"]
    assert "Simon Willison" in rendered
    assert "Post 1" in rendered
    assert "Post 2" in rendered
