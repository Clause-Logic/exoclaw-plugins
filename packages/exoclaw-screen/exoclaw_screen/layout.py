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

Block list output shape:

``LayoutBlock(x, y, w, h, kind, attrs, payload)`` — one per
visible region. ``kind`` is the AST node class name lowercased
(``"heading"``, ``"paragraph"``, etc.); ``payload`` is the
node's content.

Container blocks themselves emit a slot — and recursively emit
their children's slots inside that area. Renderers can treat
container blocks as a no-op; they only need to paint the leaf
blocks that follow.
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


def lay_out(
    doc: a.Document,
    capabilities: DisplayCapabilities,
    base_path: "str | None" = None,
) -> "list[LayoutBlock]":
    """Walk the document, produce positioned ``LayoutBlock``s.

    ``base_path`` is the directory used to resolve ``.include``
    image directives' relative ``src`` paths. ``None`` means
    "don't resolve includes — render them as plain image stubs".
    Pillow renderer passes this through from the ``screen.md``
    parent dir. Cycle detection happens at render time via a
    visited-paths set.
    """
    blocks: list[LayoutBlock] = []
    cursor_y = 0
    pad = 4  # px between top-level blocks at the panel edge.
    for node in doc.children:
        h = _layout_node(
            node,
            x=0,
            y=cursor_y,
            w=capabilities.width,
            h=None,
            caps=capabilities,
            blocks=blocks,
            base_path=base_path,
        )
        cursor_y += h + pad
    return blocks


def _layout_node(
    node: Any,
    x: int,
    y: int,
    w: int,
    h: "int | None",
    caps: DisplayCapabilities,
    blocks: "list[LayoutBlock]",
    base_path: "str | None",
    slot_sized: bool = False,
) -> int:
    """Layout a single node into ``blocks`` at ``(x, y)`` with
    width ``w``. Returns the height the node consumed.

    ``h`` is the slot's available height (None means "use
    estimate"). Containers may override this for their children.

    ``slot_sized`` signals that a parent layout pass already
    resolved this node's ``w`` / ``h`` IAL attrs into ``w`` and
    ``h`` — so a container child should NOT re-resolve them
    against its slot. Without this flag, a 25% col inside a
    row would double-apply: row computes 25% of 800 = 200, then
    the col would see slot w=200 with attr w=25% and shrink to 50.
    """
    attrs = getattr(node, "attrs", {}) or {}
    kind = type(node).__name__.lower()

    if isinstance(node, a.Container):
        return _layout_container(
            node,
            x=x,
            y=y,
            w=w,
            h=h,
            caps=caps,
            blocks=blocks,
            base_path=base_path,
            slot_sized=slot_sized,
        )

    # Non-container node: emit a single block. Use h if given,
    # otherwise estimate.
    height = h if h is not None else _estimate_height(node, caps)
    blocks.append(
        LayoutBlock(
            x=x,
            y=y,
            w=w,
            h=height,
            kind=kind,
            attrs=attrs,
            payload=node,
        )
    )
    return height


def _layout_container(
    node: a.Container,
    x: int,
    y: int,
    w: int,
    h: "int | None",
    caps: DisplayCapabilities,
    blocks: "list[LayoutBlock]",
    base_path: "str | None",
    slot_sized: bool = False,
) -> int:
    """Lay out a ``.row`` / ``.col`` / ``.grid`` container.

    Resolves child ``w`` / ``h`` against the container's box,
    applies ``gap`` between siblings. ``.grid cols=N`` flows
    children into N equal-width cells.

    ``slot_sized`` indicates the parent layout already resolved
    this container's ``w`` / ``h`` IAL attrs into the slot
    arguments — so we trust them as-is and don't re-apply.
    """
    attrs = node.attrs or {}
    gap = _parse_int_attr(attrs, "gap", 0)

    box_w = w
    box_h = h
    if not slot_sized:
        # Root call (or parent that didn't claim to size us):
        # honour our own ``w`` / ``h`` attrs against the slot.
        if "w" in attrs:
            own_w = _resolve_size(attrs, "w", w, None)
            if own_w is not None:
                box_w = own_w
        if "h" in attrs:
            own_h = _resolve_size(attrs, "h", h if h is not None else caps.height, None)
            if own_h is not None:
                box_h = own_h
    # If the container has no resolved height after the above,
    # we'll grow to fit children. Track that.
    auto_height = box_h is None

    children = node.children
    n_children = len(children)
    if n_children == 0:
        # Empty container — emit a slot anyway for renderer
        # debugging and return 0 height.
        blocks.append(
            LayoutBlock(
                x=x,
                y=y,
                w=box_w if box_w is not None else w,
                h=0,
                kind="container",
                attrs=attrs,
                payload=node,
            )
        )
        return 0

    # Emit a marker block for the container itself so renderers
    # can debug-outline it. Height fills in after children done.
    container_block = LayoutBlock(
        x=x,
        y=y,
        w=box_w if box_w is not None else w,
        h=0,
        kind="container",
        attrs=attrs,
        payload=node,
    )
    blocks.append(container_block)

    if node.kind == "row":
        # Distribute width left-to-right.
        consumed_h = _layout_row(
            children,
            x=x,
            y=y,
            box_w=box_w if box_w is not None else w,
            box_h=box_h,
            gap=gap,
            caps=caps,
            blocks=blocks,
            base_path=base_path,
        )
    elif node.kind == "grid":
        consumed_h = _layout_grid(
            node,
            x=x,
            y=y,
            box_w=box_w if box_w is not None else w,
            box_h=box_h,
            gap=gap,
            caps=caps,
            blocks=blocks,
            base_path=base_path,
        )
    else:
        # ``col`` and unknown kinds: stack top-to-bottom.
        consumed_h = _layout_col(
            children,
            x=x,
            y=y,
            box_w=box_w if box_w is not None else w,
            box_h=box_h,
            gap=gap,
            caps=caps,
            blocks=blocks,
            base_path=base_path,
        )

    final_h = consumed_h if auto_height else (box_h or consumed_h)
    container_block.h = final_h
    return final_h


def _layout_row(
    children: "list[Any]",
    x: int,
    y: int,
    box_w: int,
    box_h: "int | None",
    gap: int,
    caps: DisplayCapabilities,
    blocks: "list[LayoutBlock]",
    base_path: "str | None",
) -> int:
    """Lay out children left-to-right inside ``box_w`` × ``box_h``."""
    n = len(children)
    if n == 0:
        return 0
    total_gap = gap * (n - 1) if n > 1 else 0
    inner_w = max(0, box_w - total_gap)

    # Resolve each child's explicit width; remaining width gets
    # split among siblings without a ``w``.
    explicit_widths: list[int | None] = []
    for child in children:
        c_attrs = getattr(child, "attrs", {}) or {}
        cw = _resolve_size(c_attrs, "w", inner_w, None)
        explicit_widths.append(cw)

    explicit_total = sum(w for w in explicit_widths if w is not None)
    auto_count = sum(1 for w in explicit_widths if w is None)
    auto_w = max(0, (inner_w - explicit_total) // max(1, auto_count)) if auto_count else 0

    cursor_x = x
    max_h_used = 0
    for i, child in enumerate(children):
        c_attrs = getattr(child, "attrs", {}) or {}
        cw = explicit_widths[i] if explicit_widths[i] is not None else auto_w
        ch = _resolve_size(c_attrs, "h", box_h if box_h is not None else 0, box_h)
        # Recurse with the child's slot. ``slot_sized=True`` so a
        # nested container child doesn't double-resolve its own
        # ``w``/``h`` against the slot we just gave it.
        consumed = _layout_node(
            child,
            x=cursor_x,
            y=y,
            w=cw or 0,
            h=ch,
            caps=caps,
            blocks=blocks,
            base_path=base_path,
            slot_sized=True,
        )
        if consumed > max_h_used:
            max_h_used = consumed
        cursor_x += (cw or 0) + (gap if i < n - 1 else 0)
    return box_h if box_h is not None else max_h_used


def _layout_col(
    children: "list[Any]",
    x: int,
    y: int,
    box_w: int,
    box_h: "int | None",
    gap: int,
    caps: DisplayCapabilities,
    blocks: "list[LayoutBlock]",
    base_path: "str | None",
) -> int:
    """Lay out children top-to-bottom inside ``box_w`` × ``box_h``."""
    n = len(children)
    if n == 0:
        return 0
    total_gap = gap * (n - 1) if n > 1 else 0

    # If we have a fixed box_h, distribute among children with
    # explicit heights getting their share, the rest splitting
    # what's left. If no box_h, just stack heights.
    explicit_heights: list[int | None] = []
    for child in children:
        c_attrs = getattr(child, "attrs", {}) or {}
        ch = _resolve_size(c_attrs, "h", box_h if box_h is not None else 0, None)
        explicit_heights.append(ch)

    if box_h is not None:
        inner_h = max(0, box_h - total_gap)
        explicit_total = sum(h for h in explicit_heights if h is not None)
        auto_count = sum(1 for h in explicit_heights if h is None)
        auto_h = max(0, (inner_h - explicit_total) // max(1, auto_count)) if auto_count else 0
    else:
        auto_h = 0  # unused; we'll use estimates

    cursor_y = y
    for i, child in enumerate(children):
        c_attrs = getattr(child, "attrs", {}) or {}
        cw = _resolve_size(c_attrs, "w", box_w, box_w)
        if box_h is not None:
            ch = explicit_heights[i] if explicit_heights[i] is not None else auto_h
        else:
            ch = explicit_heights[i]  # may be None → estimate
        consumed = _layout_node(
            child,
            x=x,
            y=cursor_y,
            w=cw if cw is not None else box_w,
            h=ch,
            caps=caps,
            blocks=blocks,
            base_path=base_path,
            slot_sized=True,
        )
        cursor_y += consumed + (gap if i < n - 1 else 0)
    return cursor_y - y


def _layout_grid(
    node: a.Container,
    x: int,
    y: int,
    box_w: int,
    box_h: "int | None",
    gap: int,
    caps: DisplayCapabilities,
    blocks: "list[LayoutBlock]",
    base_path: "str | None",
) -> int:
    """Lay out children in an N-column grid.

    ``cols=N`` from the container's attrs determines the column
    count. Without ``cols``, falls back to col-style stacking.
    """
    cols = _parse_int_attr(node.attrs or {}, "cols", 0)
    if cols <= 0:
        return _layout_col(
            node.children,
            x=x,
            y=y,
            box_w=box_w,
            box_h=box_h,
            gap=gap,
            caps=caps,
            blocks=blocks,
            base_path=base_path,
        )
    n = len(node.children)
    if n == 0:
        return 0
    total_h_gap = gap * (cols - 1) if cols > 1 else 0
    cell_w = max(0, (box_w - total_h_gap) // cols)
    rows = (n + cols - 1) // cols
    # Estimate per-row height as max of children in that row.
    cursor_y = y
    for r in range(rows):
        row_children = node.children[r * cols : (r + 1) * cols]
        row_max_h = 0
        for c, child in enumerate(row_children):
            cx = x + c * (cell_w + gap)
            consumed = _layout_node(
                child,
                x=cx,
                y=cursor_y,
                w=cell_w,
                h=None,
                caps=caps,
                blocks=blocks,
                base_path=base_path,
                slot_sized=True,
            )
            if consumed > row_max_h:
                row_max_h = consumed
        cursor_y += row_max_h + (gap if r < rows - 1 else 0)
    return cursor_y - y


# ── Sizing helpers ───────────────────────────────────────────────


def _resolve_size(
    attrs: "dict[str, Any]", key: str, parent_extent: int, default: "int | None"
) -> "int | None":
    """Resolve a size attr (``w`` / ``h``) into pixels.

    - ``"200"`` → ``200`` (absolute pixels)
    - ``"50%"`` → percent of ``parent_extent``
    - missing key → ``default``
    - malformed → ``default``
    """
    val = attrs.get(key)
    if val is None:
        return default
    if not isinstance(val, str):
        return default
    s = val.strip()
    if not s:
        return default
    if s.endswith("%"):
        try:
            pct = int(s[:-1])
        except (ValueError, TypeError):
            return default
        return max(0, (parent_extent * pct) // 100)
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def _parse_int_attr(attrs: "dict[str, Any]", key: str, default: int) -> int:
    """Parse a bare integer attr (``gap=10`` / ``cols=3``)."""
    val = attrs.get(key)
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if not isinstance(val, str):
        return default
    try:
        return int(val.strip())
    except (ValueError, TypeError):
        return default


def _estimate_height(node: Any, caps: DisplayCapabilities) -> int:
    """Stub height-estimator. Treats every block as one row of
    body-text height. Renderers can request a re-layout after
    rendering for accurate measure in v0.1."""
    row_h = max(1, caps.height // max(1, caps.char_rows))
    if isinstance(node, a.Heading):
        # Headings get more vertical real estate proportional to
        # level — h1 biggest, h6 smallest.
        scale = max(1, 4 - node.level)
        return row_h * scale
    if isinstance(node, a.HorizontalRule):
        # Clamp to ≥1 — when ``row_h == 1`` (e.g. tiny OLED with
        # huge ``char_rows``), ``row_h // 2`` is 0 and zero-height
        # blocks would overlap their successor + crash renderers
        # that assume positive heights.
        return max(1, row_h // 2)
    if isinstance(node, a.CodeBlock):
        return row_h * max(1, node.text.count("\n") + 1)
    if isinstance(node, a.ListBlock):
        return row_h * max(1, len(node.items))
    if isinstance(node, a.Container):
        return row_h * max(1, len(node.children))
    return row_h
