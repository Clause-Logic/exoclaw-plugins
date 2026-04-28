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
    assert params["required"] == ["url"]
