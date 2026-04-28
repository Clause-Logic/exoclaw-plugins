"""MicroPython smoke test for ``exoclaw-screen``.

Pure-Python — no pytest. Driven by the workspace's
``mise run test-micro`` task on a coverage-variant MicroPython
binary.

Covers the chip-relevant surface: parser, layout engine, IAL
parser, container layout, image-directive IAL collision. The
host-side ``renderer/pillow.py`` is **not** exercised here — it
imports Pillow lazily inside ``render_to_png()`` and chip MP
doesn't ship Pillow. The validation of the lazy-import boundary
is implicit: ``from exoclaw_screen import ...`` (top-level)
must not pull the renderer module, otherwise this whole file
would fail to import on chip. We assert that import works
below; we do not import ``exoclaw_screen.renderer.pillow``
directly.
"""


def test_top_level_imports():
    from exoclaw_screen import (
        COLOR_MONO,
        REFRESH_SLOW,
        Display,
        DisplayCapabilities,
    )

    assert callable(Display)
    assert callable(DisplayCapabilities)
    assert COLOR_MONO == "mono"
    assert REFRESH_SLOW == "slow"


def test_parser_round_trips_simple_screen():
    """Parse a small markdown screen — assert AST has expected
    top-level shape. This is the chip-side hot path: the
    ``RepaintScreenTool`` calls ``parse(file_text)`` on every
    repaint, so it must work on MP."""
    from exoclaw_screen import ast as a
    from exoclaw_screen.parser import parse

    src = "# Title\n\nbody **bold** and _italic_\n\n- a\n- b"
    doc = parse(src)
    kinds = [type(c).__name__ for c in doc.children]
    assert kinds == ["Heading", "Paragraph", "ListBlock"]
    assert isinstance(doc.children[0], a.Heading)
    assert doc.children[0].level == 1


def test_layout_emits_block_per_top_level_node():
    """V0 layout pass — flat block list, one entry per top-level
    AST node. Container-aware positioning lands in v0.1."""
    from exoclaw_screen.layout import lay_out
    from exoclaw_screen.parser import parse
    from exoclaw_screen.protocol import (
        COLOR_MONO,
        REFRESH_SLOW,
        DisplayCapabilities,
    )

    caps = DisplayCapabilities(
        width=800,
        height=480,
        color_mode=COLOR_MONO,
        refresh_class=REFRESH_SLOW,
        char_cols=80,
        char_rows=24,
        supports_partial=True,
    )
    doc = parse("# A\n\n# B\n\n# C")
    blocks = lay_out(doc, caps)
    assert len(blocks) == 3
    # Y cursor should be monotonic.
    ys = [b.y for b in blocks]
    assert ys == sorted(ys)


def test_skill_entry_point_returns_dict():
    """The skill payload — chip-side firmware bundler reads this
    via ``importlib.metadata`` (host-side); MP itself doesn't have
    that machinery, but the function is pure Python and runs on MP
    as a smoke test."""
    from exoclaw_screen.skills import screen

    skill = screen()
    assert isinstance(skill, dict)
    assert skill["name"] == "screen"
    assert "content" in skill
    assert "path" not in skill


def test_ial_parser_handles_classes_and_attrs():
    """The IAL parser (``parse_ial``) is a hot path used by both
    block IAL and image-directive IAL — verify it runs on MP and
    produces the expected dict shape."""
    from exoclaw_screen.parser import parse_ial

    attrs = parse_ial(".title align=center color=red")
    assert attrs["class"] == ["title"]
    assert attrs["align"] == "center"
    assert attrs["color"] == "red"


def test_block_ial_full_grammar_round_trip():
    """Paragraph + blockquote + code block + list IAL all wire
    correctly on MP. Mirrors the v0 grammar surface that the
    SKILL.md prompt promises agents — if this regresses on chip,
    the agent's IAL output silently loses metadata."""
    from exoclaw_screen import ast as a
    from exoclaw_screen.parser import parse

    src = (
        "Paragraph text. {color=red}\n"
        "\n"
        "> quoted line {.callout}\n"
        "\n"
        "```python {.snippet}\n"
        "print('hi')\n"
        "```\n"
        "\n"
        "{.bullets}\n"
        "- a\n"
        "- b\n"
    )
    doc = parse(src)
    para = doc.children[0]
    assert isinstance(para, a.Paragraph)
    assert para.attrs.get("color") == "red"

    bq = doc.children[1]
    assert isinstance(bq, a.Blockquote)
    assert bq.attrs.get("class") == ["callout"]

    cb = doc.children[2]
    assert isinstance(cb, a.CodeBlock)
    assert cb.lang == "python"
    assert cb.attrs.get("class") == ["snippet"]

    lst = doc.children[3]
    assert isinstance(lst, a.ListBlock)
    assert lst.attrs.get("class") == ["bullets"]


def test_image_directive_ial_not_stolen_by_paragraph():
    """The grammar's tightest corner: a paragraph that ends with
    an image directive's IAL ``){.qrcode}`` must keep the IAL
    on the image, not the paragraph."""
    from exoclaw_screen import ast as a
    from exoclaw_screen.parser import parse

    doc = parse("Scan: ![QR](https://example.com){.qrcode size=200}")
    para = doc.children[0]
    assert isinstance(para, a.Paragraph)
    assert para.attrs == {}
    img = para.content[-1]
    assert isinstance(img, a.Image)
    assert img.attrs.get("class") == ["qrcode"]
    assert img.attrs.get("size") == "200"


def test_layout_honors_container_w_h_gap():
    """Layout's ``.row`` / ``.col`` ``w`` / ``h`` / ``gap`` honoring
    must work on MP — the chip-side renderer relies on those
    rectangles being correct."""
    from exoclaw_screen.layout import lay_out
    from exoclaw_screen.parser import parse
    from exoclaw_screen.protocol import (
        COLOR_MONO,
        REFRESH_SLOW,
        DisplayCapabilities,
    )

    caps = DisplayCapabilities(
        width=800,
        height=480,
        color_mode=COLOR_MONO,
        refresh_class=REFRESH_SLOW,
        char_cols=80,
        char_rows=24,
        supports_partial=True,
    )
    src = "::: {.row gap=20}\n# A\n# B\n:::"
    blocks = lay_out(parse(src), caps)
    headings = [b for b in blocks if b.kind == "heading"]
    assert len(headings) == 2
    # gap=20 between two children: each = (800 - 20) / 2 = 390.
    assert headings[0].w == 390
    assert headings[1].x == 410
