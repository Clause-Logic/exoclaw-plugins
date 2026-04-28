"""MicroPython smoke test for ``exoclaw-tools-web``.

Pure-Python — no pytest. Driven by the workspace's
``mise run test-micro`` task on a coverage-variant MicroPython
binary.

Verifies the streaming HTML→Markdown converter and the two
agent-facing tools (``WebFetchTool``, ``WebSearchTool``) import
+ run on chip MP. The converter is the chip-side hot path: it
must run without ``re``, without ``html.parser``, and without
materialising the whole HTML body in memory — the
state-machine tokenizer + bounded sink callback give us
streaming semantics that survive a 5 MB news article on a
chip with ~50 KB of free heap.
"""

import asyncio


def test_top_level_imports():
    from exoclaw_tools_web import (
        MarkdownConverter,
        StreamingMarkdownConverter,
        WebFetchTool,
        WebSearchTool,
        convert,
    )

    assert callable(MarkdownConverter)
    assert callable(StreamingMarkdownConverter)
    assert callable(WebFetchTool)
    assert callable(WebSearchTool)
    assert callable(convert)


def test_skill_entry_point_returns_dict():
    """Firmware stage task reads the skill payload via
    ``importlib.metadata`` (host-side); MP itself doesn't have
    that machinery, but the entry-point function is pure Python
    and runs on MP as a smoke test.

    Deliberately ``content``-only — no ``path``. Bundler does
    ``shutil.copytree`` on ``path`` if set, which would copy
    ``html_to_markdown.py``'s 1700+ lines onto the chip's flash
    twice (once via the package tree, once via the skill copy).
    Same lesson as workspace + screen packages."""
    from exoclaw_tools_web.skills import web

    skill = web()
    assert isinstance(skill, dict)
    assert skill["name"] == "web"
    assert "content" in skill
    assert skill["content"]
    assert "path" not in skill


def test_convert_html_to_markdown_no_re():
    """One-shot ``convert()`` runs on MP — proves the tokenizer
    + tree walker have no ``re`` / ``html.parser`` dependency.
    Also verifies entity decoding (``&amp;``) works without the
    stdlib ``html`` package which isn't reliably present on chip
    MP."""
    from exoclaw_tools_web import convert

    md = convert("<h1>Hello</h1><p>world &amp; chip</p>")
    assert "Hello" in md
    assert "world" in md
    assert "&" in md  # entity decoded
    assert "&amp;" not in md


def test_streaming_converter_emits_block_by_block():
    """The chip-side hot path: feed HTML in chunks, sink
    callback receives markdown fragments as each top-level
    block finishes parsing. Memory stays bounded — the document
    is never materialised whole.

    Three blocks in, three sink invocations out. Without
    block-by-block emission the chip would peak at "size of the
    whole document in markdown form" instead of "size of the
    largest single block." That's the ~5 MB → 50 KB difference
    that lets ``WebFetchTool`` survive a real news article."""
    from exoclaw_tools_web import StreamingMarkdownConverter

    chunks_out = []

    def sink(fragment):
        chunks_out.append(fragment)

    conv = StreamingMarkdownConverter(sink=sink)
    conv.feed("<h1>One</h1>")
    conv.feed("<h1>Two</h1>")
    conv.feed("<h1>Three</h1>")
    conv.close()

    joined = "".join(chunks_out)
    assert "One" in joined
    assert "Two" in joined
    assert "Three" in joined
    # Should have emitted at least one fragment per block.
    assert len(chunks_out) >= 3


def test_streaming_converter_handles_split_tag_across_chunks():
    """Tag boundaries can land mid-chunk on a real wire. The
    state machine must hold partial-tag state across feed
    calls — otherwise ``<h1>`` split as ``<h`` + ``1>`` would
    parse as text + ``1>``. Chip-side this would silently
    corrupt every page split on a TLS record boundary."""
    from exoclaw_tools_web import StreamingMarkdownConverter

    parts = []
    conv = StreamingMarkdownConverter(sink=parts.append)
    # Split inside a tag, inside an attribute, and across an
    # entity — all the boundaries that show up in real HTTP
    # streaming bodies.
    conv.feed("<h")
    conv.feed("1>Hello</h1><p>w")
    conv.feed("orld &am")
    conv.feed("p; chip</p>")
    conv.close()

    out = "".join(parts)
    assert "Hello" in out
    assert "world" in out
    assert "&" in out and "&amp;" not in out


def test_web_fetch_tool_construction_and_metadata():
    """Tool can be constructed without a client (lazy creation)
    and exposes the expected JSON-schema shape. The chip path
    constructs ``WebFetchTool()`` once at boot and reuses it
    across turns; the client is built lazily on first
    ``execute()`` so we don't pay an import + socket cost on a
    chip that never browses."""
    from exoclaw_tools_web import WebFetchTool

    tool = WebFetchTool()
    assert tool.name == "web_fetch"
    params = tool.parameters
    assert params["type"] == "object"
    assert "url" in params["properties"]
    assert params["required"] == ["url"]


def test_web_fetch_rejects_non_http_url_on_mp():
    """``ftp://`` / ``file://`` scheme rejection runs before any
    network code. Verifies the early-return path executes on MP
    without dragging in the HTTPClient (which on chip needs WiFi
    state we don't have in a smoke test)."""
    from exoclaw_tools_web import WebFetchTool

    async def _run():
        tool = WebFetchTool()
        out = await tool.execute(url="ftp://nope.test/")
        assert "Error" in out
        assert "http" in out

    asyncio.run(_run())


def test_web_search_tool_metadata_shape():
    """``WebSearchTool`` requires a provider but the metadata
    properties (``name`` / ``parameters``) are pure-Python
    accessors — they run without touching the provider. The
    chip path constructs the tool with a real provider; this
    test stubs it because we can't reach the network in a smoke
    test."""
    from exoclaw_tools_web import WebSearchTool

    class _StubProvider:
        async def chat(self, **kwargs):
            return None

        def get_default_model(self):
            return "stub"

    tool = WebSearchTool(provider=_StubProvider(), model="stub")  # type: ignore[invalid-argument-type]
    assert tool.name == "web_search"
    params = tool.parameters
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["required"] == ["query"]
