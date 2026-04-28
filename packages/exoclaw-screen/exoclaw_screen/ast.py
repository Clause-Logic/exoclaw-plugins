"""AST node classes for the parsed screen markdown.

Node trees are produced by ``parser.parse(source)`` and consumed
by ``layout.lay_out(doc, capabilities)``.

Plain classes with ``__init__`` rather than ``@dataclass`` for
cross-runtime parity — MicroPython 1.27 doesn't populate
``__annotations__`` for variable annotations, so the runtime
``@dataclass`` decorator on MP synthesises an empty ``__init__``.
Same pattern as ``CronJob`` / ``CronSchedule`` /
``BatchSnapshot`` elsewhere in this workspace.

Two layers of node:

- **Block nodes** — own a region of the layout. Headings,
  paragraphs, lists, blockquotes, code blocks, horizontal rules,
  containers, images-as-directives.
- **Inline nodes** — content INSIDE a block. Text, bold, italic,
  inline code, links, hard breaks.

A block's ``content`` field is a list of inline nodes (or list of
block nodes for containers / blockquotes).

``attrs`` is the IAL attribute dict produced by
``parser.parse_ial``. Class shorthand ``.foo`` lands under the
``"class"`` key as an ordered list (e.g. ``{"class": ["title",
"section"]}``). Bare ``key=value`` attributes land as their own
keys with string values. A bare key with no ``=`` lands as
``{"key": True}``. Always present (empty dict if no IAL).
"""

from __future__ import annotations

from typing import Any

# ── Block kinds ──────────────────────────────────────────────────


class Document:
    """Root node. ``children`` is a list of top-level block nodes."""

    def __init__(self, children: "list[Any] | None" = None) -> None:
        self.children: list[Any] = children if children is not None else []


class Container:
    """``::: {.row}`` / ``::: {.col}`` / ``::: {.grid cols=N}`` block.

    ``kind`` is one of ``"row"``, ``"col"``, ``"grid"``, or any
    other class the agent emits — the parser doesn't gatekeep,
    the layout engine ignores unknown kinds.
    """

    def __init__(
        self,
        kind: str,
        attrs: "dict[str, Any] | None" = None,
        children: "list[Any] | None" = None,
    ) -> None:
        self.kind = kind
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}
        self.children: list[Any] = children if children is not None else []


class Heading:
    """``# heading`` — ``level`` is 1–6 (we only RENDER 1–3 in v0
    but the parser preserves the level so we don't lossy-strip
    h4/h5/h6 if the agent emits them)."""

    def __init__(
        self,
        level: int,
        content: "list[Any] | None" = None,
        attrs: "dict[str, Any] | None" = None,
    ) -> None:
        self.level = level
        self.content: list[Any] = content if content is not None else []
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}


class Paragraph:
    """A run of text that doesn't fit any other block kind."""

    def __init__(
        self,
        content: "list[Any] | None" = None,
        attrs: "dict[str, Any] | None" = None,
    ) -> None:
        self.content: list[Any] = content if content is not None else []
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}


class ListBlock:
    """``- a / - b`` (unordered) or ``1. a / 2. b`` (ordered).

    ``items`` is a list of ``ListItem``. Single-level only in v0;
    nested-list source is parsed flat (the indented item just
    becomes a sibling text run).
    """

    def __init__(
        self,
        ordered: bool,
        items: "list[Any] | None" = None,
        attrs: "dict[str, Any] | None" = None,
    ) -> None:
        self.ordered = ordered
        self.items: list[Any] = items if items is not None else []
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}


class ListItem:
    """A single item in a ``ListBlock``. ``content`` is inline."""

    def __init__(self, content: "list[Any] | None" = None) -> None:
        self.content: list[Any] = content if content is not None else []


class Blockquote:
    """``> ...`` line-prefix block. ``content`` is a list of block
    nodes (paragraphs etc.) — multi-paragraph quotes parse correctly
    even though we render only the first paragraph in v0."""

    def __init__(
        self,
        content: "list[Any] | None" = None,
        attrs: "dict[str, Any] | None" = None,
    ) -> None:
        self.content: list[Any] = content if content is not None else []
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}


class CodeBlock:
    """``\\`\\`\\``` fenced code. ``lang`` is the optional info
    string (e.g. ``python``); ``text`` is the raw code without the
    fence delimiters."""

    def __init__(
        self,
        text: str,
        lang: str = "",
        attrs: "dict[str, Any] | None" = None,
    ) -> None:
        self.text = text
        self.lang = lang
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}


class HorizontalRule:
    """``---`` on its own line."""

    def __init__(self, attrs: "dict[str, Any] | None" = None) -> None:
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}


class Image:
    """``![alt](src){.class attr=val}``. The ``.class`` IAL
    determines whether this is rendered as a plain image (default,
    falls back to italic alt on chip), a directive (``.qrcode``,
    ``.include``), or something else the renderer registry
    recognises."""

    def __init__(
        self,
        src: str,
        alt: str = "",
        attrs: "dict[str, Any] | None" = None,
    ) -> None:
        self.src = src
        self.alt = alt
        self.attrs: dict[str, Any] = attrs if attrs is not None else {}


# ── Inline kinds ─────────────────────────────────────────────────


class Text:
    """Plain text run."""

    def __init__(self, text: str) -> None:
        self.text = text


class Bold:
    """``**text**``. Children are inline nodes (so
    ``**bold _italic_**`` nests correctly)."""

    def __init__(self, children: "list[Any] | None" = None) -> None:
        self.children: list[Any] = children if children is not None else []


class Italic:
    """``_text_`` or ``*text*``. Children are inline nodes."""

    def __init__(self, children: "list[Any] | None" = None) -> None:
        self.children: list[Any] = children if children is not None else []


class InlineCode:
    """`` `text` `` — backtick-delimited inline code."""

    def __init__(self, text: str) -> None:
        self.text = text


class Link:
    """``[text](url)`` or ``[text](url "title")``. ``text`` is a
    list of inline nodes (so ``[**bold link**](x)`` nests)."""

    def __init__(
        self,
        text: "list[Any] | None" = None,
        url: str = "",
        title: str = "",
    ) -> None:
        self.text: list[Any] = text if text is not None else []
        self.url = url
        self.title = title


class HardBreak:
    """Two trailing spaces + newline inside a paragraph — explicit
    line break that doesn't end the paragraph."""
