"""V0 layout-engine tests.

The v0 layout pass is intentionally minimal — flat top-to-bottom
block list, no container-aware positioning yet. Tests verify the
shape of the output (one block per top-level node, monotonic
y-cursor) rather than pixel-precise positions, since the heuristic
height estimation will change in v0.1."""

from __future__ import annotations

from exoclaw_screen.layout import LayoutBlock, lay_out
from exoclaw_screen.parser import parse
from exoclaw_screen.protocol import (
    COLOR_MONO,
    REFRESH_SLOW,
    DisplayCapabilities,
)


def _caps() -> DisplayCapabilities:
    return DisplayCapabilities(
        width=800,
        height=480,
        color_mode=COLOR_MONO,
        refresh_class=REFRESH_SLOW,
        char_cols=80,
        char_rows=24,
        supports_partial=True,
    )


class TestLayout:
    def test_emits_block_per_top_level_node(self) -> None:
        doc = parse("# A\n\n# B\n\n# C")
        blocks = lay_out(doc, _caps())
        assert len(blocks) == 3
        assert all(isinstance(b, LayoutBlock) for b in blocks)

    def test_y_cursor_is_monotonic(self) -> None:
        doc = parse("# A\n\nbody\n\n---\n\n# B")
        blocks = lay_out(doc, _caps())
        ys = [b.y for b in blocks]
        assert ys == sorted(ys)

    def test_block_kind_is_lowercased_class_name(self) -> None:
        doc = parse("# Heading\n\nparagraph\n\n---")
        kinds = [b.kind for b in lay_out(doc, _caps())]
        assert kinds == ["heading", "paragraph", "horizontalrule"]

    def test_full_panel_width(self) -> None:
        doc = parse("# A")
        blocks = lay_out(doc, _caps())
        assert blocks[0].x == 0
        assert blocks[0].w == 800

    def test_attrs_preserved_on_block(self) -> None:
        doc = parse("# Title {.section align=center}")
        blocks = lay_out(doc, _caps())
        assert blocks[0].attrs.get("class") == ["section"]
        assert blocks[0].attrs.get("align") == "center"
