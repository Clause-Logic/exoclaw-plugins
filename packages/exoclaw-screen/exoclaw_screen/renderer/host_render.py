"""CPython-only rasterisation CLI.

Invoked by the unix board's ``HostPreviewDisplay`` via
``os.system`` so MicroPython unix-port (which can't load Pillow,
a CPython C extension) can still produce a real PNG when the
agent calls ``repaint_screen``. The MP Display writes the
markdown to disk + spawns this CLI; this CLI imports Pillow,
parses + lays out + renders, exits.

CLI:
    python3 -m exoclaw_screen.renderer.host_render \\
        <md_path> <out_path> <width> <height> <color_mode> \\
        <refresh_class> <char_cols> <char_rows> <supports_partial> \\
        [<base_path>]

``supports_partial`` is ``1`` or ``0``. ``base_path`` is optional
and used for ``.include`` image directives' relative-path
resolution.

Positional args (no JSON) so the MP caller doesn't need shlex
quoting machinery — every arg is a primitive scalar that
shell-quotes cleanly with simple ``"..."`` wrapping.
"""

from __future__ import annotations

import sys

from exoclaw_screen.layout import lay_out
from exoclaw_screen.parser import parse
from exoclaw_screen.protocol import DisplayCapabilities
from exoclaw_screen.renderer.pillow import PillowRenderer


def main(argv: list[str]) -> int:
    if len(argv) < 10:
        sys.stderr.write(
            "usage: host_render md_path out_path width height "
            "color_mode refresh_class char_cols char_rows "
            "supports_partial [base_path]\n"
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
    base_path: str | None = argv[10] if len(argv) > 10 else None

    with open(md_path, "r") as f:
        md = f.read()
    doc = parse(md)
    blocks = lay_out(doc, caps)
    PillowRenderer(caps).render_to_png(blocks, out_path, base_path=base_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
