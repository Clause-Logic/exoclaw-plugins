"""Tests for OpenRouterSearchTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw_openrouter_search.tool import OpenRouterSearchTool


def _make_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture
def tool() -> OpenRouterSearchTool:
    return OpenRouterSearchTool(
        model="google/gemini-2.0-flash-001",
        api_key="test-key",
    )


class TestOpenRouterSearchTool:
    def test_name(self, tool: OpenRouterSearchTool) -> None:
        assert tool.name == "web_search"

    def test_description(self, tool: OpenRouterSearchTool) -> None:
        assert "search" in tool.description.lower()

    def test_parameters(self, tool: OpenRouterSearchTool) -> None:
        params = tool.parameters
        assert "query" in params["properties"]
        assert params["required"] == ["query"]

    async def test_execute_success(self, tool: OpenRouterSearchTool) -> None:
        with patch("exoclaw_openrouter_search.tool.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = _make_response("Paris is the capital of France.")
            result = await tool.execute(query="capital of France")

        assert "Paris" in result
        mock_ac.assert_called_once()
        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/google/gemini-2.0-flash-001"
        assert call_kwargs["extra_body"] == {"plugins": [{"id": "web"}]}
        assert call_kwargs["api_key"] == "test-key"

    async def test_model_prefix_not_doubled(self) -> None:
        tool = OpenRouterSearchTool(
            model="openrouter/google/gemini-2.0-flash-001",
            api_key="test-key",
        )
        with patch("exoclaw_openrouter_search.tool.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = _make_response("result")
            await tool.execute(query="test")

        model = mock_ac.call_args.kwargs["model"]
        assert model.count("openrouter/") == 1

    async def test_missing_api_key(self) -> None:
        tool = OpenRouterSearchTool(model="google/gemini-2.0-flash-001")
        result = await tool.execute(query="test")
        assert "OPENROUTER_API_KEY" in result

    async def test_api_error_returns_error_string(self, tool: OpenRouterSearchTool) -> None:
        with patch("exoclaw_openrouter_search.tool.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.side_effect = RuntimeError("rate limited")
            result = await tool.execute(query="test")
        assert "Error" in result
        assert "rate limited" in result

    async def test_env_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        tool = OpenRouterSearchTool(model="google/gemini-2.0-flash-001")
        with patch("exoclaw_openrouter_search.tool.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = _make_response("result")
            await tool.execute(query="test")
        assert mock_ac.call_args.kwargs["api_key"] == "env-key"

    async def test_max_tokens_passed(self, tool: OpenRouterSearchTool) -> None:
        tool2 = OpenRouterSearchTool(
            model="google/gemini-2.0-flash-001",
            api_key="test-key",
            max_tokens=512,
        )
        with patch("exoclaw_openrouter_search.tool.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = _make_response("result")
            await tool2.execute(query="test")
        assert mock_ac.call_args.kwargs["max_tokens"] == 512

    async def test_empty_content_returns_empty_string(self, tool: OpenRouterSearchTool) -> None:
        with patch("exoclaw_openrouter_search.tool.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = _make_response("")
            result = await tool.execute(query="test")
        assert result == ""
