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


def _sniff_image(body: bytes) -> "tuple[str, int, int] | None":
    """Read width/height from PNG/JPEG/GIF/WebP/BMP headers.
    Pure-stdlib — no Pillow — so this runs on chip MP too.

    Returns ``(format, width, height)`` or ``None`` if the body
    isn't a recognised image. The agent uses these dimensions to
    write IAL like ``{h=300}`` on the screen image directive so
    the layout engine sizes a slot proportional to the source.
    """
    if len(body) < 24:
        return None
    # PNG: 8-byte signature, then IHDR chunk at offset 8 with
    # length(4) + "IHDR"(4) + width(4 BE) + height(4 BE).
    if body[:8] == b"\x89PNG\r\n\x1a\n":
        w = int.from_bytes(body[16:20], "big")
        h = int.from_bytes(body[20:24], "big")
        return ("PNG", w, h)
    # GIF: "GIF87a" or "GIF89a" + width(2 LE) + height(2 LE).
    if body[:6] in (b"GIF87a", b"GIF89a"):
        w = int.from_bytes(body[6:8], "little")
        h = int.from_bytes(body[8:10], "little")
        return ("GIF", w, h)
    # BMP: "BM" + 16 bytes header, then width(4 LE) + height(4 LE).
    if body[:2] == b"BM" and len(body) >= 26:
        w = int.from_bytes(body[18:22], "little")
        h = int.from_bytes(body[22:26], "little")
        return ("BMP", w, h)
    # WebP: "RIFF" + size(4) + "WEBP" — VP8/VP8L/VP8X variants.
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP" and len(body) >= 30:
        chunk = body[12:16]
        if chunk == b"VP8 " and len(body) >= 30:
            # Lossy VP8 — width/height at offset 26 (14-bit each).
            w = int.from_bytes(body[26:28], "little") & 0x3FFF
            h = int.from_bytes(body[28:30], "little") & 0x3FFF
            return ("WebP", w, h)
        if chunk == b"VP8L" and len(body) >= 25:
            # Lossless VP8L: signature byte 0x2F + 14-bit dims.
            b0, b1, b2, b3 = body[21], body[22], body[23], body[24]
            w = ((b1 & 0x3F) << 8 | b0) + 1
            h = ((b3 & 0x0F) << 10 | b2 << 2 | (b1 & 0xC0) >> 6) + 1
            return ("WebP", w, h)
        if chunk == b"VP8X" and len(body) >= 30:
            # Extended VP8X — width/height at offset 24 (24-bit LE).
            w = int.from_bytes(body[24:27], "little") + 1
            h = int.from_bytes(body[27:30], "little") + 1
            return ("WebP", w, h)
    # JPEG: 0xFFD8 then scan for SOFn marker (0xFFC0..0xFFCF
    # except DHT/DAC/JPG markers). Each marker has a 2-byte
    # length, then a 1-byte precision, height(2 BE), width(2 BE).
    if body[:2] == b"\xff\xd8":
        i = 2
        n = len(body)
        while i + 9 < n:
            if body[i] != 0xFF:
                return None
            marker = body[i + 1]
            i += 2
            # Skip standalone fill bytes / marker padding.
            while i < n and body[i] == 0xFF:
                i += 1
            if marker in (0xD8, 0xD9):
                # SOI / EOI — no length.
                continue
            if i + 1 >= n:
                return None
            seg_len = int.from_bytes(body[i : i + 2], "big")
            # SOFn (0xC0..0xCF) excluding DHT(0xC4), JPG(0xC8), DAC(0xCC).
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if i + 7 >= n:
                    return None
                h = int.from_bytes(body[i + 3 : i + 5], "big")
                w = int.from_bytes(body[i + 5 : i + 7], "big")
                return ("JPEG", w, h)
            i += seg_len
        return None
    return None


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
        workspace: "Any | None" = None,
    ) -> None:
        self._client: "ClientProto | None" = client
        self._owns_client = client is None
        self._max_bytes = max_bytes if max_bytes is not None else _MAX_BYTES
        self._timeout = timeout
        # Optional sandboxing root for ``save_to`` mode. When set,
        # save_to is resolved against this directory and ``..``
        # segments are rejected. Leaving it unset lets the agent
        # write to any path it can name (matches the simpler pre-
        # save-mode behaviour for hosts where the workspace is the
        # whole filesystem anyway).
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a URL. Default returns its content as Markdown. "
            "Pass save_to=<path> to download raw bytes (e.g. an "
            "image) into the workspace; the result includes the "
            "saved path, byte count, and (for images) the source "
            "width × height so the screen layout can size a slot "
            "correctly."
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
                "save_to": {
                    "type": "string",
                    "description": (
                        "Workspace-relative path to save the raw "
                        "response bytes to. Use this for images "
                        "(JPEG/PNG/GIF) or other binary content. "
                        "Omit to receive the response as Markdown."
                    ),
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, save_to: "str | None" = None, **kwargs: Any) -> str:
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Error: url must start with http:// or https://"

        if save_to:
            return await self._fetch_to_file(url, save_to)

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

    async def _fetch_to_file(self, url: str, save_to: str) -> str:
        """Download raw bytes to ``save_to``. Reports byte count
        plus image dimensions when the content sniffs as an image.

        ``save_to`` is resolved against ``self._workspace`` if set
        (rejecting ``..`` traversal); otherwise taken as-is so the
        agent can write to absolute paths on hosts where the
        workspace is the whole filesystem.
        """
        from exoclaw._compat import Path

        # Resolve + sandbox.
        if self._workspace is not None:
            for seg in save_to.split("/"):
                if seg == "..":
                    return "Error: save_to must not contain '..' segments"
            target = Path(str(self._workspace)) / save_to
        else:
            target = Path(save_to)

        client = self._get_client()
        try:
            async with client.stream_post(url, method="GET", timeout=self._timeout) as resp:
                if resp.status_code >= 400:
                    return "Error: HTTP {} fetching {}".format(resp.status_code, url)
                body = await resp.aread()
        except Exception as e:  # noqa: BLE001
            logger.error("web_fetch_save_failed", **{"url": url, "error": str(e)})
            return "Error: failed to fetch {}: {}".format(url, e)

        if not body:
            return "Error: empty response from {}".format(url)

        # Make sure parent dir exists before writing — agents
        # often save into a fresh subdirectory like ``images/``
        # without ``mkdir`` first.
        try:
            parent = Path(str(target.parent))
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # ``mkdir`` may legitimately fail (target.parent is the
            # filesystem root, etc.) — let the actual write surface
            # the real error if it matters.
            pass

        try:
            with open(str(target), "wb") as f:
                f.write(body)
        except OSError as e:
            return "Error: failed to write {}: {}".format(target, e)

        info = _sniff_image(body)
        if info is not None:
            fmt, w, h = info
            return "Saved {} bytes to {} ({} {}x{})".format(len(body), target, fmt, w, h)
        return "Saved {} bytes to {}".format(len(body), target)

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
