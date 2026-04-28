"""Pillow-backed host preview renderer.

CPython-only — chip MicroPython doesn't ship Pillow. Renders a
``LayoutBlock`` list into a PNG file at the panel's resolution.

Usage:

    from exoclaw_screen.parser import parse
    from exoclaw_screen.layout import lay_out
    from exoclaw_screen.protocol import DisplayCapabilities, COLOR_MONO, REFRESH_SLOW
    from exoclaw_screen.renderer.pillow import PillowRenderer

    caps = DisplayCapabilities(
        width=800, height=480, color_mode=COLOR_MONO,
        refresh_class=REFRESH_SLOW, char_cols=80, char_rows=24,
        supports_partial=True,
    )
    doc = parse(open("screen.md").read())
    blocks = lay_out(doc, caps)
    PillowRenderer(caps).render_to_png(blocks, "screen.png")

Open ``screen.png`` in macOS Preview / any image viewer with
auto-refresh-on-change to iterate on layout designs.

Minimum-viable v0:

- Renders text blocks using a default monospace font (PIL's
  built-in default if no system font is available).
- Heading levels get scaled font sizes.
- Hr lines render as horizontal rules.
- Container / Image / List / Blockquote stubs exist; layout-engine
  v0 doesn't position them well so they may overflow — fix lands
  with the v0.1 layout engine pass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from exoclaw_screen import ast as a
from exoclaw_screen.protocol import COLOR_MONO, DisplayCapabilities

if TYPE_CHECKING:
    from exoclaw_screen.layout import LayoutBlock


class PillowRenderer:
    """Host-side preview renderer. CPython + Pillow only."""

    def __init__(self, capabilities: DisplayCapabilities) -> None:
        self._caps = capabilities

    def render_to_png(self, blocks: "list[LayoutBlock]", out_path: str) -> None:
        """Render the block list to ``out_path``. Raises
        ``ImportError`` if Pillow isn't installed."""
        from PIL import Image, ImageDraw  # type: ignore[unresolved-import]

        # Mono panels: paint white background + black ink.
        # Color panels: white + black for v0 (color quantisation
        # against the IAL colour attrs lands in v0.1).
        bg = "white"
        fg = "black"

        img = Image.new("RGB", (self._caps.width, self._caps.height), bg)
        draw = ImageDraw.Draw(img)

        body_font = self._load_default_font(size=14)
        heading_fonts: dict[int, Any] = {
            1: self._load_default_font(size=28, bold=True),
            2: self._load_default_font(size=22, bold=True),
            3: self._load_default_font(size=18, bold=True),
        }

        for block in blocks:
            self._draw_block(
                draw=draw,
                block=block,
                body_font=body_font,
                heading_fonts=heading_fonts,
                fg=fg,
            )

        img.save(out_path)

    @staticmethod
    def _load_default_font(size: int, bold: bool = False) -> Any:
        """Best-effort font load. Falls back to PIL's
        ``ImageFont.load_default()`` if no system font is found."""
        from PIL import ImageFont  # type: ignore[unresolved-import]

        # Try a couple of common fonts that ship widely. ``bold`` is
        # a hint — we just pick a bolder variant where one exists.
        candidates = (
            ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"),
            ("Arial Bold.ttf" if bold else "Arial.ttf"),
            ("Helvetica.ttc"),
        )
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def _draw_block(
        self,
        draw: Any,
        block: "LayoutBlock",
        body_font: Any,
        heading_fonts: "dict[int, Any]",
        fg: str,
    ) -> None:
        node = block.payload
        x, y = block.x, block.y
        w = block.w

        if isinstance(node, a.Heading):
            font = heading_fonts.get(node.level, body_font)
            draw.text((x, y), _flatten_inline(node.content), font=font, fill=fg)
            return
        if isinstance(node, a.Paragraph):
            draw.text((x, y), _flatten_inline(node.content), font=body_font, fill=fg)
            return
        if isinstance(node, a.HorizontalRule):
            draw.line([(x, y + 4), (x + w, y + 4)], fill=fg, width=1)
            return
        if isinstance(node, a.ListBlock):
            cursor_y = y
            line_h = body_font.size + 4 if hasattr(body_font, "size") else 18
            for i, item in enumerate(node.items):
                marker = "{}.".format(i + 1) if node.ordered else "*"
                text = "{} {}".format(marker, _flatten_inline(item.content))
                draw.text((x, cursor_y), text, font=body_font, fill=fg)
                cursor_y += line_h
            return
        if isinstance(node, a.Blockquote):
            # Single-paragraph blockquote in v0.
            draw.line([(x, y), (x, y + block.h)], fill=fg, width=2)
            for child in node.content:
                if isinstance(child, a.Paragraph):
                    draw.text(
                        (x + 8, y),
                        _flatten_inline(child.content),
                        font=body_font,
                        fill=fg,
                    )
                    break
            return
        if isinstance(node, a.CodeBlock):
            draw.rectangle([x, y, x + w, y + block.h], outline=fg, width=1)
            draw.text((x + 4, y + 4), node.text, font=body_font, fill=fg)
            return
        # Container / Image / unknown — v0 stub: just outline the
        # region so the developer can see the layout slot.
        if self._caps.color_mode == COLOR_MONO:
            outline = fg
        else:
            outline = "gray"
        draw.rectangle([x, y, x + w, y + block.h], outline=outline, width=1)
        draw.text((x + 4, y + 4), block.kind, font=body_font, fill=fg)


def _flatten_inline(nodes: "list[Any]") -> str:
    """Collapse an inline-node list into plain text for v0
    rendering. Loses bold/italic/code styling visually — fixed
    in v0.1 with per-node style runs."""
    out: list[str] = []
    for n in nodes:
        if isinstance(n, a.Text):
            out.append(n.text)
        elif isinstance(n, (a.Bold, a.Italic)):
            out.append(_flatten_inline(n.children))
        elif isinstance(n, a.InlineCode):
            out.append(n.text)
        elif isinstance(n, a.Link):
            out.append(_flatten_inline(n.text))
        elif isinstance(n, a.HardBreak):
            out.append("\n")
        elif isinstance(n, a.Image):
            out.append("[{}]".format(n.alt or n.src))
    return "".join(out)
