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


class TestContainerLayout:
    """Container ``.row`` / ``.col`` honor ``w`` / ``h`` / ``gap`` attrs."""

    def test_row_splits_width_evenly_when_no_w(self) -> None:
        # Two children, no widths — each gets half of 800.
        src = "::: {.row}\n# A\n# B\n:::"
        doc = parse(src)
        blocks = lay_out(doc, _caps())
        # Find the heading blocks (children of the container).
        headings = [b for b in blocks if b.kind == "heading"]
        assert len(headings) == 2
        assert headings[0].x == 0
        assert headings[0].w == 400
        assert headings[1].x == 400
        assert headings[1].w == 400

    def test_row_honors_explicit_w_pct(self) -> None:
        src = "::: {.row}\n::: {.col w=25%}\n# A\n:::\n::: {.col w=75%}\n# B\n:::\n:::"
        doc = parse(src)
        blocks = lay_out(doc, _caps())
        cols = [b for b in blocks if b.kind == "container" and b.attrs.get("w") in ("25%", "75%")]
        assert len(cols) == 2
        # 25% of 800 = 200; 75% of 800 = 600.
        assert cols[0].w == 200
        assert cols[1].w == 600
        assert cols[1].x == 200

    def test_row_honors_gap(self) -> None:
        # gap=20 between two children: each child = (800 - 20) / 2 = 390.
        src = "::: {.row gap=20}\n# A\n# B\n:::"
        doc = parse(src)
        blocks = lay_out(doc, _caps())
        headings = [b for b in blocks if b.kind == "heading"]
        assert len(headings) == 2
        assert headings[0].w == 390
        assert headings[1].x == 410  # 390 + 20 gap

    def test_col_honors_h_attr(self) -> None:
        src = "::: {.col h=300}\n# A\n# B\n:::"
        doc = parse(src)
        blocks = lay_out(doc, _caps())
        # Each child should get half of 300.
        headings = [b for b in blocks if b.kind == "heading"]
        assert len(headings) == 2
        assert headings[0].h == 150
        assert headings[1].h == 150
        assert headings[1].y == 150

    def test_col_honors_gap_in_height(self) -> None:
        src = "::: {.col h=300 gap=20}\n# A\n# B\n:::"
        doc = parse(src)
        blocks = lay_out(doc, _caps())
        headings = [b for b in blocks if b.kind == "heading"]
        # Available height = 300 - 20 = 280; each child = 140.
        assert headings[0].h == 140
        assert headings[1].h == 140
        assert headings[1].y == 160  # 140 + 20 gap

    def test_nested_row_in_col(self) -> None:
        src = "::: {.col}\n::: {.row}\n# A\n# B\n:::\n# C\n:::"
        doc = parse(src)
        blocks = lay_out(doc, _caps())
        # Heading C should occupy full width.
        c = [b for b in blocks if b.kind == "heading" and b.x == 0]
        # A is at x=0, C is at x=0 — both should be in this list.
        assert len(c) >= 2

    def test_grid_cols_3(self) -> None:
        src = "::: {.grid cols=3}\n# A\n# B\n# C\n:::"
        doc = parse(src)
        blocks = lay_out(doc, _caps())
        headings = [b for b in blocks if b.kind == "heading"]
        assert len(headings) == 3
        # Each cell = 800 / 3.
        assert headings[0].w == 800 // 3
        assert headings[1].x == 800 // 3
        assert headings[2].x == 2 * (800 // 3)

    def test_grid_cols_3_with_gap(self) -> None:
        src = "::: {.grid cols=3 gap=10}\n# A\n# B\n# C\n:::"
        doc = parse(src)
        blocks = lay_out(doc, _caps())
        headings = [b for b in blocks if b.kind == "heading"]
        # cell_w = (800 - 20) / 3 = 260.
        assert headings[0].w == 260
        assert headings[1].x == 270  # 260 + 10
        assert headings[2].x == 540  # 260 + 10 + 260 + 10


class TestImageBlock:
    """A lone-image paragraph (``![alt](src){h=300}``) honors the
    image's ``h`` IAL so the layout gives the renderer a slot
    proportional to the source. Without this, the paragraph
    would default to one row of body-text height and a
    1024x768 photo would render into a tiny strip."""

    def test_lone_image_honors_h_ial(self) -> None:
        src = "![cat](cat.jpg){h=300}"
        blocks = lay_out(parse(src), _caps())
        assert len(blocks) == 1
        assert blocks[0].kind == "paragraph"
        assert blocks[0].h == 300

    def test_lone_image_height_alias(self) -> None:
        src = "![cat](cat.jpg){height=240}"
        blocks = lay_out(parse(src), _caps())
        assert blocks[0].h == 240

    def test_lone_image_without_h_falls_back_to_one_row(self) -> None:
        # Without IAL the paragraph still gets one row of
        # body-text height — same as before, no regression.
        src = "![cat](cat.jpg)"
        blocks = lay_out(parse(src), _caps())
        # row_h = 480 // 24 = 20
        assert blocks[0].h == 20

    def test_image_in_paragraph_with_text_does_not_pull_h(self) -> None:
        # Mixed paragraph (text + image) shouldn't have its
        # height hijacked by the image's IAL — that's the
        # signal we treat the image as inline rather than a
        # picture block.
        src = "Look: ![cat](cat.jpg){h=300}"
        blocks = lay_out(parse(src), _caps())
        # row_h = 20.
        assert blocks[0].h == 20
