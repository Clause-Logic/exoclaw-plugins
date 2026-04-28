"""``WebSearchTool`` — search via an OpenAI-compatible LLM
provider's web-search plugin.

Cross-runtime port of the previous ``exoclaw-openrouter-search``
plugin. Drops ``litellm`` (CPython-only, ~50 MB of deps) — uses
the workspace's ``OpenAIStreamingProvider`` directly. The agent
configures a search-dedicated ``Deployment`` with
``extra_body={"plugins": [{"id": "web"}]}`` and the tool routes
queries through that deployment.
"""

from __future__ import annotations

from typing import Any

from exoclaw._compat import get_logger
from exoclaw.agent.tools.protocol import ToolBase
from exoclaw.providers.protocol import LLMProvider

logger = get_logger()


class WebSearchTool(ToolBase):
    """Search the web via an LLM with web-search plugin enabled."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        max_tokens: int = 1024,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web. Returns a grounded answer with sources. "
            "Use this when you need current information you don't "
            "already know — recent events, prices, weather, latest "
            "docs. The reply already has citations; you don't need "
            "to fetch the source URLs unless the user asks."
        )

    @property
    def parameters(self) -> "dict[str, Any]":
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, **kwargs: Any) -> str:
        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": query}],
                model=self._model,
                max_tokens=self._max_tokens,
            )
        except Exception as e:  # noqa: BLE001 — surface backend errors verbatim
            logger.error(
                "web_search_failed",
                **{"search.query": query, "llm.model": self._model, "error": str(e)},
            )
            return "Error: web_search failed: {}".format(e)
        return resp.content or "(empty response)"
