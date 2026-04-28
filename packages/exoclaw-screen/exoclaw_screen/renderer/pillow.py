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
    blocks = lay_out(doc, caps, base_path="/path/to/screen.md/parent")
    PillowRenderer(caps).render_to_png(blocks, "screen.png")

Open ``screen.png`` in macOS Preview / any image viewer with
auto-refresh-on-change to iterate on layout designs.

Renderer features:

- Text blocks rendered with a default monospace font (PIL's
  built-in default if no system font is available).
- Heading levels get scaled font sizes.
- Hr lines render as horizontal rules.
- Image directives:
    - ``.include`` — recursively parse + render the referenced
      markdown file inline at the block's slot.
    - ``.qrcode`` — encode ``src`` as a QR PNG (gated on the
      ``qrcode`` package; falls back to italic URL text).
    - Plain image (no class) — italic alt text fallback.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from exoclaw_screen import ast as a
from exoclaw_screen.protocol import COLOR_MONO, DisplayCapabilities

if TYPE_CHECKING:
    from exoclaw_screen.layout import LayoutBlock


class PillowRenderer:
    """Host-side preview renderer. CPython + Pillow only."""

    def __init__(self, capabilities: DisplayCapabilities) -> None:
        self._caps = capabilities

    def render_to_png(
        self,
        blocks: "list[LayoutBlock]",
        out_path: str,
        base_path: "str | None" = None,
    ) -> None:
        """Render the block list to ``out_path``. Raises
        ``ImportError`` if Pillow isn't installed.

        ``base_path`` is the directory used to resolve ``.include``
        image directives' relative ``src`` paths. Cycle detection
        runs through a visited-paths set scoped to this render
        call (single-level includes only — see GRAMMAR.md).
        """
        from PIL import Image, ImageDraw

        bg = "white"
        fg = "black"

        img = Image.new("RGB", (self._caps.width, self._caps.height), bg)
        draw = ImageDraw.Draw(img)

        body_font = self._load_default_font(size=14)
        italic_font = self._load_default_font(size=14, italic=True)
        heading_fonts: dict[int, Any] = {
            1: self._load_default_font(size=28, bold=True),
            2: self._load_default_font(size=22, bold=True),
            3: self._load_default_font(size=18, bold=True),
        }

        visited: set[str] = set()
        for block in blocks:
            self._draw_block(
                draw=draw,
                img=img,
                block=block,
                body_font=body_font,
                italic_font=italic_font,
                heading_fonts=heading_fonts,
                fg=fg,
                base_path=base_path,
                visited=visited,
            )

        img.save(out_path)

    @staticmethod
    def _load_default_font(size: int, bold: bool = False, italic: bool = False) -> Any:
        """Best-effort font load. Falls back to PIL's
        ``ImageFont.load_default()`` if no system font is found."""
        from PIL import ImageFont

        # Try a few common variants; not all hosts have all fonts.
        candidates: list[str] = []
        if bold and italic:
            candidates += ["DejaVuSansMono-BoldOblique.ttf", "Arial Bold Italic.ttf"]
        elif bold:
            candidates += ["DejaVuSansMono-Bold.ttf", "Arial Bold.ttf"]
        elif italic:
            candidates += [
                "DejaVuSansMono-Oblique.ttf",
                "DejaVuSans-Oblique.ttf",
                "Arial Italic.ttf",
            ]
        else:
            candidates += ["DejaVuSansMono.ttf", "Arial.ttf"]
        candidates += ["Helvetica.ttc"]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def _draw_block(
        self,
        draw: Any,
        img: Any,
        block: "LayoutBlock",
        body_font: Any,
        italic_font: Any,
        heading_fonts: "dict[int, Any]",
        fg: str,
        base_path: "str | None",
        visited: "set[str]",
    ) -> None:
        node = block.payload
        x, y = block.x, block.y
        w = block.w

        if isinstance(node, a.Heading):
            font = heading_fonts.get(node.level, body_font)
            draw.text((x, y), _flatten_inline(node.content), font=font, fill=fg)
            return
        if isinstance(node, a.Paragraph):
            self._draw_inline_run(
                draw=draw,
                img=img,
                nodes=node.content,
                x=x,
                y=y,
                w=w,
                body_font=body_font,
                italic_font=italic_font,
                fg=fg,
                base_path=base_path,
                visited=visited,
                block_h=block.h,
            )
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
        if isinstance(node, a.Image):
            self._draw_image_directive(
                draw=draw,
                img=img,
                node=node,
                block=block,
                body_font=body_font,
                italic_font=italic_font,
                fg=fg,
                base_path=base_path,
                visited=visited,
            )
            return
        if isinstance(node, a.Container):
            # Container is a slot marker; its child blocks already
            # appear separately in the flat block list.
            return
        # Unknown — debug-outline the slot.
        if self._caps.color_mode == COLOR_MONO:
            outline = fg
        else:
            outline = "gray"
        draw.rectangle([x, y, x + w, y + block.h], outline=outline, width=1)
        draw.text((x + 4, y + 4), block.kind, font=body_font, fill=fg)

    def _draw_inline_run(
        self,
        draw: Any,
        img: Any,
        nodes: "list[Any]",
        x: int,
        y: int,
        w: int,
        body_font: Any,
        italic_font: Any,
        fg: str,
        base_path: "str | None",
        visited: "set[str]",
        block_h: int,
    ) -> None:
        """Paint a paragraph's inline children. Image directives
        nested in paragraphs (e.g. ``Scan: ![QR](url){.qrcode}``)
        get dispatched the same way standalone images do.
        """
        # Render any image directives one-by-one inline; flatten
        # the rest as text.
        text_parts: list[str] = []
        for n in nodes:
            if isinstance(n, a.Image):
                # Flush text before the image.
                if text_parts:
                    draw.text((x, y), "".join(text_parts), font=body_font, fill=fg)
                    text_parts = []
                # Render image into a sub-rect at the same slot.
                from exoclaw_screen.layout import LayoutBlock

                sub = LayoutBlock(x=x, y=y, w=w, h=block_h, kind="image", attrs=n.attrs, payload=n)
                self._draw_image_directive(
                    draw=draw,
                    img=img,
                    node=n,
                    block=sub,
                    body_font=body_font,
                    italic_font=italic_font,
                    fg=fg,
                    base_path=base_path,
                    visited=visited,
                )
            else:
                text_parts.append(_flatten_inline([n]))
        if text_parts:
            draw.text((x, y), "".join(text_parts), font=body_font, fill=fg)

    def _draw_image_directive(
        self,
        draw: Any,
        img: Any,
        node: a.Image,
        block: "LayoutBlock",
        body_font: Any,
        italic_font: Any,
        fg: str,
        base_path: "str | None",
        visited: "set[str]",
    ) -> None:
        """Dispatch on the image's IAL ``.class``:

        - ``.include`` → recursively parse + render the referenced
          markdown file at the block's slot.
        - ``.qrcode`` → encode ``src`` into a QR code (gated on
          ``qrcode`` package; falls back to italic URL text).
        - default (no recognised class) → italic alt-text fallback.
        """
        classes = node.attrs.get("class") or []

        if "include" in classes:
            self._render_include(
                draw=draw,
                img=img,
                node=node,
                block=block,
                body_font=body_font,
                italic_font=italic_font,
                fg=fg,
                base_path=base_path,
                visited=visited,
            )
            return

        if "qrcode" in classes:
            self._render_qrcode(
                draw=draw,
                img=img,
                node=node,
                block=block,
                italic_font=italic_font,
                fg=fg,
            )
            return

        # Default: try to load + paste a real raster image from
        # ``src`` (resolved against ``base_path`` for relatives).
        # Aspect-preserving thumbnail into the block's (w, h) slot.
        # On any failure (missing file, unsupported format, IO
        # error) fall through to the italic alt-text rendering —
        # the agent still sees something rather than a blank
        # space, and the alt text usually identifies what was
        # supposed to be there.
        if self._render_raster_image(
            img=img,
            node=node,
            block=block,
            base_path=base_path,
        ):
            return
        text = node.alt or node.src
        draw.text((block.x, block.y), text, font=italic_font, fill=fg)

    def _render_raster_image(
        self,
        img: Any,
        node: a.Image,
        block: "LayoutBlock",
        base_path: "str | None",
    ) -> bool:
        """Load + paste a raster image from ``node.src`` into the
        block's slot. Returns ``True`` if the image was painted,
        ``False`` if the caller should fall back (e.g. file not
        found, format unsupported).

        Relative ``src`` is resolved against the layout's
        ``base_path`` — agents reference workspace-relative paths
        like ``cat.jpg`` and the renderer resolves to
        ``{workspace}/cat.jpg``. Absolute paths pass through
        unchanged.
        """
        from PIL import Image as _PILImage

        src = node.src
        if not src:
            return False
        # ``src`` may be a URL — we don't fetch over the wire from
        # inside the renderer (separation of concerns; the agent
        # uses ``web_fetch`` to download into the workspace, then
        # references the local path here).
        if src.startswith(("http://", "https://")):
            return False
        if base_path is not None and not os.path.isabs(src):
            full = os.path.normpath(os.path.join(base_path, src))
        else:
            full = src
        if not os.path.exists(full):
            return False
        try:
            raster = _PILImage.open(full)
            raster.load()
        except Exception:  # noqa: BLE001 — any PIL failure → alt fallback
            return False
        # Aspect-preserving fit into the slot. Scale to fill as
        # much of the (w, h) box as possible without distortion.
        # Unlike ``thumbnail`` (which only scales down), this
        # also scales up — a 600x400 source into an 800x480 slot
        # becomes 720x480 (height-constrained), centered
        # horizontally.
        from PIL import Image as _PILImage

        sw, sh = raster.size
        bw, bh = max(1, block.w), max(1, block.h)
        scale = min(bw / sw, bh / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        if (nw, nh) != (sw, sh):
            resample = getattr(_PILImage, "LANCZOS", None) or getattr(
                _PILImage.Resampling, "LANCZOS", 1
            )
            raster = raster.resize((nw, nh), resample)
        ox = block.x + max(0, (bw - nw) // 2)
        oy = block.y + max(0, (bh - nh) // 2)
        # ``paste`` accepts an RGBA source onto an RGB canvas via
        # the alpha mask if present — fall back to plain paste
        # otherwise. ``getbands`` lets us check without forcing
        # a conversion that'd flatten transparency for opaque
        # sources.
        bands = raster.getbands()
        if "A" in bands:
            img.paste(raster, (ox, oy), raster)
        else:
            img.paste(raster, (ox, oy))
        return True

    def _render_include(
        self,
        draw: Any,
        img: Any,
        node: a.Image,
        block: "LayoutBlock",
        body_font: Any,
        italic_font: Any,
        fg: str,
        base_path: "str | None",
        visited: "set[str]",
    ) -> None:
        """Resolve ``node.src`` against ``base_path``, parse the
        referenced markdown file, lay out into the block's slot,
        and recursively paint into ``img``.

        Single-level only (per GRAMMAR.md): if a nested include
        is encountered we render its alt text instead. Cycle
        detection via the ``visited`` set.
        """
        from exoclaw_screen.layout import lay_out
        from exoclaw_screen.parser import parse
        from exoclaw_screen.protocol import DisplayCapabilities

        src = node.src
        if not src:
            return
        # Resolve path.
        if base_path is not None and not os.path.isabs(src):
            full = os.path.normpath(os.path.join(base_path, src))
        else:
            full = src
        # Cycle detection.
        if full in visited:
            draw.text(
                (block.x, block.y),
                "[cycle: {}]".format(node.alt or node.src),
                font=italic_font,
                fill=fg,
            )
            return
        try:
            with open(full, "r") as fh:
                source = fh.read()
        except (OSError, IOError):
            draw.text(
                (block.x, block.y),
                "[missing: {}]".format(node.alt or node.src),
                font=italic_font,
                fill=fg,
            )
            return

        sub_visited = set(visited)
        sub_visited.add(full)

        # Lay out into the slot — fabricate caps that match the
        # outer panel but constrained to the slot dimensions.
        sub_caps = DisplayCapabilities(
            width=block.w,
            height=block.h or self._caps.height,
            color_mode=self._caps.color_mode,
            refresh_class=self._caps.refresh_class,
            char_cols=self._caps.char_cols,
            char_rows=self._caps.char_rows,
            supports_partial=self._caps.supports_partial,
            screen_path=full,
        )
        sub_doc = parse(source)
        sub_blocks = lay_out(sub_doc, sub_caps, base_path=os.path.dirname(full))
        # Translate sub-blocks into the outer slot.
        for sb in sub_blocks:
            # Skip nested includes — single level only.
            if isinstance(sb.payload, a.Image):
                cls = sb.payload.attrs.get("class") or []
                if "include" in cls:
                    draw.text(
                        (block.x + sb.x, block.y + sb.y),
                        "[nested-include: {}]".format(sb.payload.alt or sb.payload.src),
                        font=italic_font,
                        fill=fg,
                    )
                    continue
            sb.x += block.x
            sb.y += block.y
            self._draw_block(
                draw=draw,
                img=img,
                block=sb,
                body_font=body_font,
                italic_font=italic_font,
                heading_fonts={
                    1: self._load_default_font(size=28, bold=True),
                    2: self._load_default_font(size=22, bold=True),
                    3: self._load_default_font(size=18, bold=True),
                },
                fg=fg,
                base_path=os.path.dirname(full),
                visited=sub_visited,
            )

    def _render_qrcode(
        self,
        draw: Any,
        img: Any,
        node: a.Image,
        block: "LayoutBlock",
        italic_font: Any,
        fg: str,
    ) -> None:
        """Encode ``node.src`` into a QR code and paste into the
        block's slot.

        Gated on the optional ``qrcode`` Python package — when not
        installed (chip MicroPython has no QR encoder yet), falls
        back to rendering the URL as italic text. This is the v0
        contract (per GRAMMAR.md): QR is a best-effort renderer
        feature.
        """
        try:
            import qrcode
        except ImportError:
            # Fallback: italic URL text.
            draw.text(
                (block.x, block.y),
                node.src or node.alt,
                font=italic_font,
                fill=fg,
            )
            return

        size_px = _parse_qr_size(node.attrs.get("size"))
        if size_px is None:
            # Default: smaller of slot w/h, or 200.
            size_px = min(block.w, block.h or 200) or 200
        try:
            qr = qrcode.QRCode(border=1, box_size=4)
            qr.add_data(node.src or "")
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
            qr_img = qr_img.resize((size_px, size_px))
            img.paste(qr_img, (block.x, block.y))
        except Exception:
            # Defensive: any error in the QR pipeline shouldn't
            # crash the whole render — fall back to text.
            draw.text(
                (block.x, block.y),
                node.src or node.alt,
                font=italic_font,
                fill=fg,
            )


def _parse_qr_size(val: Any) -> "int | None":
    """Parse ``size=200`` IAL value into pixels. ``None`` if missing
    or malformed."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if not isinstance(val, str):
        return None
    s = val.strip()
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


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
