"""Public surface for ``exoclaw-tools-web``.

Three exports:

- ``convert`` / ``MarkdownConverter`` — HTML→Markdown converter.
  Pure-Python state machine, no ``re``, no ``html.parser`` —
  runs on chip MP. ~700 LOC, passes 113/131 turndown fixtures
  on default options.
- ``WebFetchTool`` — agent-facing tool: GET a URL, return
  Markdown. Uses ``exoclaw.http.HTTPClient`` (cross-runtime)
  for the fetch and the converter above for the HTML→Markdown
  step. Output bounded by per-runtime cap (32 KB MP / 128 KB
  CPython).
- ``WebSearchTool`` — agent-facing tool: query the web through
  an LLM-provider deployment with web-search plugin enabled
  (e.g. OpenRouter's ``{"plugins": [{"id": "web"}]}`` extra-body
  toggle). Drops the litellm dep that the previous
  ``exoclaw-openrouter-search`` plugin had — uses the same
  ``OpenAIStreamingProvider`` the agent already speaks to."""

from exoclaw_tools_web.fetch import WebFetchTool
from exoclaw_tools_web.html_to_markdown import (
    MarkdownConverter,
    StreamingMarkdownConverter,
    convert,
)
from exoclaw_tools_web.search import WebSearchTool

__all__ = [
    "MarkdownConverter",
    "StreamingMarkdownConverter",
    "WebFetchTool",
    "WebSearchTool",
    "convert",
]
