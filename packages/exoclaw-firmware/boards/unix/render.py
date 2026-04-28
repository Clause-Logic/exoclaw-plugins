"""CPython-only rasterisation CLI.

Invoked by the unix board's ``HostPreviewDisplay`` via
``os.system``. Parses markdown → lays out → renders via Pillow
→ composites HUD (status pip + captions bar) → writes PNG.

CLI:
    python3 -P render.py \\
        <md_path> <out_path> <width> <height> <color_mode> \\
        <refresh_class> <char_cols> <char_rows> <supports_partial> \\
        [<base_path> [<status> [<caption>]]]

All positional — no JSON, no shlex quoting needed on the MP side.
"""

from __future__ import annotations

import os
import sys
import textwrap

# Remove cwd from sys.path so MP-stub modules in ``.stage/``
# (typing.py, datetime.py) don't shadow real stdlib.
_cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", ".", _cwd)]

from exoclaw_screen.layout import lay_out
from exoclaw_screen.parser import parse
from exoclaw_screen.protocol import DisplayCapabilities
from exoclaw_screen.renderer.pillow import PillowRenderer


def _draw_hud(img: "Any", width: int, height: int, status: str, caption: str) -> None:
    """Composite the HUD overlay onto the rendered panel image."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)

    # Status pip — top-right corner.
    if status:
        try:
            pip_font = ImageFont.truetype("Helvetica.ttc", 11)
        except (OSError, IOError):
            pip_font = ImageFont.load_default()

        pip_colors = {
            "ready": (0, 180, 70),
            "listening": (200, 50, 50),
            "transcribing": (200, 150, 0),
            "thinking": (200, 150, 0),
        }
        color = pip_colors.get(status, (120, 120, 120))
        dot_r = 5
        dot_x = width - 12
        dot_y = 12
        draw.ellipse(
            (dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r),
            fill=color,
        )
        bbox = pip_font.getbbox(status)
        tw = bbox[2] - bbox[0]
        draw.text((dot_x - dot_r - tw - 4, dot_y - 6), status, fill=color, font=pip_font)

    # Captions bar — bottom of panel.
    if caption:
        bar_h = 48
        bar_y = height - bar_h
        # Semi-transparent dark bar. Pillow doesn't do alpha
        # compositing on RGB easily — just draw a solid dark rect.
        draw.rectangle((0, bar_y, width, height), fill=(30, 30, 30))
        try:
            cap_font = ImageFont.truetype("Helvetica.ttc", 13)
        except (OSError, IOError):
            cap_font = ImageFont.load_default()
        lines = textwrap.wrap(caption, width=90)[:2]
        for i, line in enumerate(lines):
            draw.text((10, bar_y + 6 + i * 18), line, fill=(240, 240, 240), font=cap_font)


def main(argv: list[str]) -> int:
    if len(argv) < 10:
        sys.stderr.write(
            "usage: render.py md_path out_path width height "
            "color_mode refresh_class char_cols char_rows "
            "supports_partial [base_path [status [caption]]]\n"
        )
        return 2
    md_path = argv[1]
    out_path = argv[2]
    caps = DisplayCapabilities(
        width=int(argv[3]),
        height=int(argv[4]),
        color_mode=argv[5],
        refresh_class=argv[6],
        char_cols=int(argv[7]),
        char_rows=int(argv[8]),
        supports_partial=argv[9] == "1",
    )
    base_path: str | None = argv[10] if len(argv) > 10 and argv[10] else None
    status: str = argv[11] if len(argv) > 11 else ""
    caption: str = argv[12] if len(argv) > 12 else ""

    with open(md_path, "r") as f:
        md = f.read()
    doc = parse(md)
    blocks = lay_out(doc, caps)

    # Render main canvas then composite HUD.
    from PIL import Image

    renderer = PillowRenderer(caps)
    bg = "white"
    img = Image.new("RGB", (caps.width, caps.height), bg)

    # Use the renderer's internal draw method if available,
    # otherwise render to file and re-open. The public API is
    # render_to_png which writes directly; we need the PIL Image
    # object to composite HUD before saving.
    renderer.render_to_png(blocks, out_path, base_path=base_path)

    if status or caption:
        img = Image.open(out_path)
        _draw_hud(img, caps.width, caps.height, status, caption)
        img.save(out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
