"""Web search tool using OpenRouter's web search plugin."""

import os
from typing import Any

from litellm import acompletion
from loguru import logger

from exoclaw.agent.tools.protocol import ToolBase


class OpenRouterSearchTool(ToolBase):
    """Search the web by calling an OpenRouter model with the web search plugin.

    The model receives the query and returns a grounded response with citations.
    Any OpenRouter model that supports the ``{"plugins": [{"id": "web"}]}``
    extra_body is compatible (e.g. ``google/gemini-2.0-flash-001``).
    """

    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-001",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens

    @property
    def _resolved_api_key(self) -> str:
        return self._api_key or os.environ.get("OPENROUTER_API_KEY", "")

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web. Returns a grounded answer with sources."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }

    async def execute(self, query: str, **kwargs: Any) -> str:
        if not self._resolved_api_key:
            return (
                "Error: OPENROUTER_API_KEY is not set. "
                "Export it or pass api_key to OpenRouterSearchTool."
            )

        try:
            model = self._model
            if not model.startswith("openrouter/"):
                model = f"openrouter/{model}"

            logger.debug("OpenRouterSearch: querying {} for: {}", model, query)

            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": query}],
                max_tokens=self._max_tokens,
                api_key=self._resolved_api_key,
                extra_body={"plugins": [{"id": "web"}]},
            )

            content: str = response.choices[0].message.content or ""
            return content

        except Exception as e:
            logger.error("OpenRouterSearch error: {}", e)
            return f"Error: {e}"
