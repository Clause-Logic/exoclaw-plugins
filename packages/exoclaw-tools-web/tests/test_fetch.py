"""``WebFetchTool`` integration tests — drive a fake httpx
transport, verify GET routing + streaming HTML→Markdown +
output-cap behaviour."""

from __future__ import annotations

import httpx
import pytest
from exoclaw.http._cpython import HttpxClient
from exoclaw_tools_web import WebFetchTool


def _make_client(handler) -> "tuple[HttpxClient, httpx.AsyncClient]":
    transport = httpx.MockTransport(handler)
    raw = httpx.AsyncClient(transport=transport)
    client = HttpxClient.__new__(HttpxClient)
    client._client = raw
    return client, raw


@pytest.mark.asyncio
async def test_fetch_converts_html_to_markdown() -> None:
    captured_methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_methods.append(request.method)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<h1>Hello</h1><p>World</p>",
        )

    client, raw = _make_client(handler)
    try:
        tool = WebFetchTool(client=client)
        result = await tool.execute(url="https://example.test/page")
    finally:
        await raw.aclose()

    assert captured_methods == ["GET"]
    assert "Hello" in result
    assert "World" in result


@pytest.mark.asyncio
async def test_fetch_rejects_non_http_url() -> None:
    tool = WebFetchTool()
    result = await tool.execute(url="ftp://nope.test/")
    assert "Error" in result
    assert "http://" in result or "https://" in result


@pytest.mark.asyncio
async def test_fetch_4xx_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found")

    client, raw = _make_client(handler)
    try:
        tool = WebFetchTool(client=client)
        result = await tool.execute(url="https://example.test/missing")
    finally:
        await raw.aclose()

    assert "Error" in result
    assert "404" in result


@pytest.mark.asyncio
async def test_fetch_truncates_oversize_body_via_output_cap() -> None:
    """Streaming converter + bounded sink. The body is 5 MB of
    paragraphs — chip-side this used to OOM under the one-shot
    converter. The streaming impl bounds memory via the output
    sink; we verify the result hits exactly the cap and includes
    the truncation notice."""
    big_html = "<p>" + "x" * 100 + "</p>"
    # Repeat the paragraph block so the result is large; each <p>
    # produces ~104 chars of markdown ("xxx...\n\n"), so 1000 of
    # them = ~104 KB of markdown.
    big_html = big_html * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=big_html.encode("utf-8"),
        )

    client, raw = _make_client(handler)
    try:
        tool = WebFetchTool(client=client, max_bytes=4_000)
        result = await tool.execute(url="https://example.test/big")
    finally:
        await raw.aclose()

    assert "truncated" in result.lower()
    # Sink stops accepting at 4000; truncation notice adds a few
    # dozen more chars.
    assert len(result) < 4_500


@pytest.mark.asyncio
async def test_fetch_handles_multi_chunk_html() -> None:
    """The HTML body comes through ``aiter_lines`` as multiple
    chunks. Verify the streaming converter stitches them
    correctly across chunk boundaries."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = b"<html>\n<body>\n<h1>Top</h1>\n<p>One</p>\n<p>Two</p>\n</body>\n</html>"
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body)

    client, raw = _make_client(handler)
    try:
        tool = WebFetchTool(client=client)
        result = await tool.execute(url="https://example.test/multiline")
    finally:
        await raw.aclose()

    assert "Top" in result
    assert "One" in result
    assert "Two" in result


@pytest.mark.asyncio
async def test_fetch_lazy_client_construction() -> None:
    tool = WebFetchTool()
    await tool.aclose()
    await tool.aclose()


def test_tool_metadata_shape() -> None:
    tool = WebFetchTool()
    assert tool.name == "web_fetch"
    params = tool.parameters
    assert params["type"] == "object"
    assert "url" in params["properties"]
    assert "save_to" in params["properties"]
    assert params["required"] == ["url"]


# Tiny valid PNG (1x1 transparent) — used as a fixture body for
# the save_to + dimension-sniffing tests below. Hand-crafted so
# we don't depend on Pillow at test time (the converter + sniff
# logic is stdlib-only by design).
_TINY_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63000100000005000100000000370000000049454e44ae"
    "426082"
)


@pytest.mark.asyncio
async def test_fetch_save_to_writes_bytes_and_reports_png_dimensions(tmp_path) -> None:
    """``save_to`` mode downloads raw bytes and reports the image
    format + dimensions for image content so the agent can size a
    screen slot without a separate ``image_info`` round trip."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=_TINY_PNG_1x1,
        )

    client, raw = _make_client(handler)
    try:
        tool = WebFetchTool(client=client, workspace=tmp_path)
        result = await tool.execute(url="https://example.test/cat.png", save_to="cat.png")
    finally:
        await raw.aclose()

    out = tmp_path / "cat.png"
    assert out.exists()
    assert out.read_bytes() == _TINY_PNG_1x1
    assert "Saved" in result
    assert "PNG" in result
    assert "1x1" in result


@pytest.mark.asyncio
async def test_fetch_save_to_rejects_traversal(tmp_path) -> None:
    """``save_to`` is sandboxed when ``workspace`` is set — ``..``
    segments are rejected before any network IO. Without this
    guard, an LLM-generated path could land bytes outside the
    agent's workspace dir."""
    tool = WebFetchTool(workspace=tmp_path)
    out = await tool.execute(url="https://example.test/x", save_to="../escape.bin")
    assert "Error" in out
    assert ".." in out


@pytest.mark.asyncio
async def test_fetch_save_to_creates_parent_dirs(tmp_path) -> None:
    """Agents typically write into a fresh subdir like
    ``images/cat.jpg`` without an ``mkdir`` first — the save path
    creates parents transparently."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=_TINY_PNG_1x1,
        )

    client, raw = _make_client(handler)
    try:
        tool = WebFetchTool(client=client, workspace=tmp_path)
        result = await tool.execute(url="https://example.test/cat.png", save_to="images/cat.png")
    finally:
        await raw.aclose()

    assert (tmp_path / "images" / "cat.png").exists()
    assert "Saved" in result


def test_sniff_image_recognises_png_jpeg_gif() -> None:
    """The format-sniffer is pure stdlib so it can run on chip MP
    too. Cover the three formats the agent's most likely to
    encounter from web image search."""
    from exoclaw_tools_web.fetch import _sniff_image

    # PNG fixture above.
    info = _sniff_image(_TINY_PNG_1x1)
    assert info == ("PNG", 1, 1)

    # GIF: 'GIF89a' header + 320x200 little-endian. Padded to
    # ≥24 bytes since the sniff guards on minimum body length.
    gif = b"GIF89a" + b"\x40\x01" + b"\xc8\x00" + b"\x00" * 16
    info = _sniff_image(gif)
    assert info == ("GIF", 320, 200)

    # JPEG: SOI + APP0 + SOF0 carrying 16x32.
    jpeg = (
        b"\xff\xd8"  # SOI
        b"\xff\xc0\x00\x11\x08"  # SOF0 marker, length=17, precision=8
        b"\x00\x20"  # height=32
        b"\x00\x10" + b"\x00" * 32  # width=16  # rest of SOF0 + filler past min length
    )
    info = _sniff_image(jpeg)
    assert info == ("JPEG", 16, 32)

    # Non-image data → None.
    assert _sniff_image(b"<html><body>not an image") is None
    assert _sniff_image(b"") is None
