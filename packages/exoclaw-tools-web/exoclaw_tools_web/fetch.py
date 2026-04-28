"""``WebFetchTool`` — fetch a URL and return clean Markdown.

Cross-runtime: uses ``exoclaw.http.HTTPClient`` (chip MP +
CPython, same Protocol) for the GET, and
``StreamingMarkdownConverter`` (pure-Python state machine, no
``re``) for the HTML→Markdown conversion. Drops the previous
``httpx`` + ``readability-lxml`` stack — those don't run on chip.

Truly streaming end-to-end:

- Wire bytes flow in via ``aiter_lines`` chunk-by-chunk.
- Each chunk feeds the converter, which incrementally emits
  markdown via a sink callback.
- The sink writes into a bounded list; once it hits
  ``_MAX_BYTES`` it raises ``_OutputCapped`` to abort further
  emit work.
- On ``_OutputCapped``, we close the wire stream + finalise the
  converter and return the bounded markdown plus a truncation
  notice.

Memory ceiling: parser internal state (~few KB for the open-tag
stack and pending-block buffer) + the bounded output buffer
(``_MAX_BYTES``) + one in-flight wire chunk. ~45 KB on MP. The
chip can fetch a 5 MB news article and return clean markdown
without OOM — the input is never materialised in heap.
"""

from __future__ import annotations

from typing import Any

from exoclaw._compat import IS_MICROPYTHON, get_logger
from exoclaw.agent.tools.protocol import ToolBase
from exoclaw.http import ClientProto, HTTPClient

from exoclaw_tools_web.html_to_markdown import StreamingMarkdownConverter

logger = get_logger()


_MAX_BYTES_CPYTHON = 128_000
_MAX_BYTES_MP = 32_000
_MAX_BYTES = _MAX_BYTES_MP if IS_MICROPYTHON else _MAX_BYTES_CPYTHON


class _OutputCapped(Exception):  # noqa: N818 — internal sentinel, not a public exception type
    """Sink-stop sentinel — raised by ``_BoundedSink`` when the
    output buffer hits the cap. The fetch loop catches this to
    terminate cleanly + add the truncation notice."""


class _BoundedSink:
    """Sink callback object for ``StreamingMarkdownConverter``.
    Accumulates fragments into a list bounded by ``cap``; raises
    ``_OutputCapped`` once accumulated size ≥ ``cap``."""

    __slots__ = ("_parts", "_size", "_cap")

    def __init__(self, cap: int) -> None:
        self._parts: list[str] = []
        self._size = 0
        self._cap = cap

    def __call__(self, fragment: str) -> None:
        if self._size >= self._cap:
            raise _OutputCapped()
        remaining = self._cap - self._size
        if len(fragment) > remaining:
            self._parts.append(fragment[:remaining])
            self._size = self._cap
            raise _OutputCapped()
        self._parts.append(fragment)
        self._size += len(fragment)

    @property
    def text(self) -> str:
        return "".join(self._parts)

    @property
    def size(self) -> int:
        return self._size


class WebFetchTool(ToolBase):
    """Fetch a URL and return its content as Markdown.

    Constructor takes an optional ``ClientProto`` for dependency
    injection (mostly so tests can wire a fake transport). When
    omitted, a fresh ``HTTPClient`` is constructed lazily on first
    call.

    ``max_bytes`` overrides the runtime-default output cap.
    Useful on hosts with abundant memory or when the agent wants
    to clamp further for prompt-budget reasons. There's no input
    cap — the converter streams arbitrarily large HTML in
    bounded memory.
    """

    def __init__(
        self,
        client: "ClientProto | None" = None,
        max_bytes: "int | None" = None,
        timeout: float = 30.0,
    ) -> None:
        self._client: "ClientProto | None" = client
        self._owns_client = client is None
        self._max_bytes = max_bytes if max_bytes is not None else _MAX_BYTES
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a URL and return its content as Markdown. Use this "
            "to read a web page the agent has been given a link to. "
            "Output is capped — large pages return the head with a "
            "truncation notice."
        )

    @property
    def parameters(self) -> "dict[str, Any]":
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http:// or https://).",
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, **kwargs: Any) -> str:
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Error: url must start with http:// or https://"

        client = self._get_client()
        sink = _BoundedSink(self._max_bytes)
        converter = StreamingMarkdownConverter(sink=sink)
        capped = False

        try:
            async with client.stream_post(url, method="GET", timeout=self._timeout) as resp:
                if resp.status_code >= 400:
                    return "Error: HTTP {} fetching {}".format(resp.status_code, url)
                # True streaming — chunk in, convert, sink, repeat.
                # If sink raises ``_OutputCapped`` we've hit the
                # output budget — finalise the converter (best
                # effort; close() may itself raise OutputCapped if
                # there's more pending) and break.
                async for line in resp.aiter_lines():
                    try:
                        converter.feed(line + "\n")
                    except _OutputCapped:
                        capped = True
                        break
        except _OutputCapped:
            capped = True
        except Exception as e:  # noqa: BLE001 — surface backend errors verbatim
            logger.error("web_fetch_failed", **{"url": url, "error": str(e)})
            return "Error: failed to fetch {}: {}".format(url, e)

        # Drain any pending state. If the cap was hit during feed,
        # close() may also raise; treat that as "more truncated"
        # and swallow.
        try:
            converter.close()
        except _OutputCapped:
            capped = True
        except Exception as e:  # noqa: BLE001
            logger.error(
                "web_fetch_convert_failed",
                **{"url": url, "error": str(e), "output.bytes": sink.size},
            )
            return "Error: failed to convert HTML from {}: {}".format(url, e)

        md = sink.text
        if not md:
            return "Error: empty response from {}".format(url)

        if capped:
            md = md + "\n\n... (truncated — output reached the {}-byte cap)".format(self._max_bytes)
        return md

    def _get_client(self) -> "ClientProto":
        if self._client is None:
            self._client = HTTPClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying client if we own it. Safe to call
        multiple times."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
