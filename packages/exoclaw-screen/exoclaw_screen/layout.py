"""Layout engine — AST + ``DisplayCapabilities`` → flat block list.

Walks the parsed ``Document`` tree, computes pixel rects for each
block against the panel viewport (resolution + char budget), and
emits a flat list of positioned ``LayoutBlock`` records that the
renderer consumes.

Renderer-agnostic. Pillow on host, LVGL on chip, and DOM in
pyscript all consume the same flat list shape.

Box model:

- Containers (``.row`` / ``.col`` / ``.grid``) recurse with a
  shrunk viewport. ``.row`` distributes width, ``.col`` height,
  ``.grid cols=N`` flows children into N equal cells.
- Sizing attrs ``w`` / ``h`` are absolute pixels OR percentage of
  parent (``50%``). Omitted = equal split among siblings.
- ``gap=N`` adds N-pixel spacing between siblings.

This module ships in v0 with a minimal layout pass — it produces
a flat block list assuming a single top-level ``.col`` (no
container support yet). Container-aware layout lands in v0.1
once we've validated the parser + Pillow renderer round-trip.

Block list output shape:

``LayoutBlock(x, y, w, h, kind, attrs, payload)`` — one per
visible region. ``kind`` is the AST node class name lowercased
(``"heading"``, ``"paragraph"``, etc.); ``payload`` is the
node's content (already wrapped to fit ``w``).
"""

from __future__ import annotations

from typing import Any

from exoclaw_screen import ast as a
from exoclaw_screen.protocol import DisplayCapabilities


class LayoutBlock:
    """One positioned region the renderer paints. Plain class for
    cross-runtime parity (no @dataclass — see ``ast.py``)."""

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        kind: str,
        attrs: "dict[str, Any] | None" = None,
        payload: Any = None,
    ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.kind = kind
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}
        self.payload = payload


def lay_out(doc: a.Document, capabilities: DisplayCapabilities) -> "list[LayoutBlock]":
    """V0 layout pass — minimum viable.

    Produces a flat top-to-bottom block list. Each top-level child
    of the Document gets one LayoutBlock with full panel width and
    a hand-estimated height (``_estimate_height``).

    Container support and proper char-budget wrapping land in v0.1.
    """
    blocks: list[LayoutBlock] = []
    cursor_y = 0
    pad = 4  # px between blocks at the panel edge
    for node in doc.children:
        h = _estimate_height(node, capabilities)
        kind = type(node).__name__.lower()
        attrs = getattr(node, "attrs", {}) or {}
        blocks.append(
            LayoutBlock(
                x=0,
                y=cursor_y,
                w=capabilities.width,
                h=h,
                kind=kind,
                attrs=attrs,
                payload=node,
            )
        )
        cursor_y += h + pad
    return blocks


def _estimate_height(node: Any, caps: DisplayCapabilities) -> int:
    """Stub height-estimator for v0. Treats every block as one
    row of body-text height. Renderers can request a re-layout
    after rendering for accurate measure in v0.1."""
    row_h = max(1, caps.height // max(1, caps.char_rows))
    if isinstance(node, a.Heading):
        # Headings get more vertical real estate proportional to
        # level — h1 biggest, h6 smallest.
        scale = max(1, 4 - node.level)
        return row_h * scale
    if isinstance(node, a.HorizontalRule):
        return row_h // 2
    if isinstance(node, a.CodeBlock):
        return row_h * max(1, node.text.count("\n") + 1)
    if isinstance(node, a.ListBlock):
        return row_h * max(1, len(node.items))
    if isinstance(node, a.Container):
        return row_h * max(1, len(node.children))
    return row_h
