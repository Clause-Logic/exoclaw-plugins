"""MicroPython smoke test for ``exoclaw-screen``.

Pure-Python — no pytest. Driven by the workspace's
``mise run test-micro`` task on a coverage-variant MicroPython
binary.

Verifies the parser and layout engine import + run on chip MP.
The renderer (``renderer/pillow.py``) imports Pillow lazily
inside ``render_to_png()`` so it doesn't break MP module-load —
this test imports the renderer module to confirm the gating works.
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
