"""HTML→Markdown converter — turndown-style output, no ``re`` module.

The chip's MicroPython ``re`` engine rejects most realistic patterns
(no negative lookahead, shallow group depth) so this module is built
on a hand-rolled state-machine tokenizer + a postorder tree walker.

Output choices follow turndown's ``commonmark-rules`` defaults:

- ``headingStyle = setext`` (h1=``===``, h2=``---``, h3-h6=``###…``)
- ``codeBlockStyle = indented`` (4-space prefix, no fences)
- ``bulletListMarker = '*'``  ⇒ ``*   ``
- ``emDelimiter = '_'``  ⇒ ``_em_``
- ``strongDelimiter = '**'``  ⇒ ``**strong**``
- ``linkStyle = inlined``  ⇒ ``[text](url)``
- ``hr = '* * *'``
- ``br = '  '``  (two trailing spaces + newline)

Anything off-defaults (fenced code, reference links, ATX headings,
custom bullets, ``preformattedCode``) is intentionally not rendered
to match — those fixtures live in ``KNOWN_FAILURES`` in the harness.

The converter is **streaming**. ``StreamingMarkdownConverter`` is the
canonical entry point — feed HTML chunks via ``feed(chunk)`` and the
sink callback receives markdown fragments as each top-level block
finishes parsing. ``convert(html_str) -> str`` is a thin wrapper that
collects sink output into a list and returns the join.

Per-block emission means peak memory is bounded by "the largest
single top-level block currently being assembled" rather than "the
whole document." For typical pages that's a few KB — chip-safe even
on a 5 MB HTML body.

License-clean note: the algorithm here mirrors turndown's flow at
the level of "this rule emits this string for this tag" — not a
copy-paste port. ``html2text`` (GPL-3.0) was not consulted.
"""

from __future__ import annotations

from typing import Callable

# ``html.unescape`` exists on CPython but isn't reliably present on
# the chip MP test runner — micropython-lib's ``html`` package isn't
# always frozen. We inline a small named-entity table + a numeric
# entity decoder below to stay stdlib-on-CPython, no-deps-on-MP.
try:
    from html import unescape as _stdlib_unescape

    def unescape(s: str) -> str:
        return _stdlib_unescape(s)
except ImportError:  # pragma: no cover (cpython has it)

    def unescape(s: str) -> str:
        return _entity_unescape(s)


# ── HTML tokenizer ──────────────────────────────────────────────
#
# A pure char-by-char state machine that emits ``(kind, payload)``
# tokens:
#
#   ("start", (tag, attrs, self_closing))
#   ("end",   tag)
#   ("text",  text)
#   ("comment", text)        — emitted but ignored by the tree
#   ("doctype", text)        — same; ignored
#
# No ``re``, no ``html.parser``. The tokenizer calls ``unescape``
# (above) on attribute values + text runs to decode entities.


# Common named entities — pragmatic subset (the full HTML5 list is
# ~2300 entries; we ship the common ones and let everything else
# pass through). Decoded inline by ``_entity_unescape`` on MP and
# by stdlib ``html.unescape`` on CPython.
_NAMED_ENTITIES = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": "\xa0",
    "copy": "\xa9",
    "reg": "\xae",
    "trade": "™",
    "hellip": "…",
    "mdash": "—",
    "ndash": "–",
    "lsquo": "‘",
    "rsquo": "’",
    "ldquo": "“",
    "rdquo": "”",
    "laquo": "\xab",
    "raquo": "\xbb",
    "para": "\xb6",
    "sect": "\xa7",
    "deg": "\xb0",
    "plusmn": "\xb1",
    "times": "\xd7",
    "divide": "\xf7",
    "middot": "\xb7",
    "bull": "•",
}


def _entity_unescape(s: str) -> str:
    """Pure-Python entity decoder: handles named (``&amp;``,
    ``&nbsp;``…), decimal (``&#39;``), and hex (``&#x27;``)
    references. Unknown entities are left as-is, matching browser
    behavior + ``html.unescape``."""
    if "&" not in s:
        return s
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c != "&":
            out.append(c)
            i += 1
            continue
        # Find the terminating ``;`` within ~32 chars; bail out if
        # absent (left as literal ``&``).
        end = -1
        limit = min(n, i + 32)
        for j in range(i + 1, limit):
            if s[j] == ";":
                end = j
                break
            if s[j] in " \t\r\n<&":
                break
        if end == -1:
            out.append("&")
            i += 1
            continue
        body = s[i + 1 : end]
        decoded: str | None = None
        if body.startswith("#"):
            try:
                if body[1:2] in ("x", "X"):
                    decoded = chr(int(body[2:], 16))
                else:
                    decoded = chr(int(body[1:]))
            except (ValueError, OverflowError):
                decoded = None
        else:
            decoded = _NAMED_ENTITIES.get(body)
        if decoded is None:
            out.append("&")
            i += 1
            continue
        out.append(decoded)
        i = end + 1
    return "".join(out)


_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "command",
    "embed",
    "hr",
    "img",
    "input",
    "keygen",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

# Tags whose content should be dropped wholesale (not rendered as
# text). Plus comments handled separately.
_DROP_TAGS = {"script", "style", "noscript", "head", "title"}

# Block-level tags — copied from turndown's ``blockElements``. Used
# during whitespace collapse + during rendering join logic.
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "audio",
    "blockquote",
    "body",
    "canvas",
    "center",
    "dd",
    "dir",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hgroup",
    "hr",
    "html",
    "isindex",
    "li",
    "main",
    "menu",
    "nav",
    "noframes",
    "noscript",
    "ol",
    "output",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}

# Tags that count as "meaningful when blank" — turndown keeps them
# even if their text content is whitespace only.
_MEANINGFUL_WHEN_BLANK = {
    "a",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "th",
    "td",
    "iframe",
    "script",
    "audio",
    "video",
}

# Tags that we treat as **transparent at the document root** —
# they don't count as a "top-level block" for emission purposes.
# When the parser sees ``<html>`` or ``<body>`` opened at root, it
# enters "inside transparent root" mode; anything that closes back
# to inside-transparent-root is the actual top-level block.
_ROOT_TRANSPARENT = {"html", "body"}


def _is_ascii_letter(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def _is_tag_name_char(c: str) -> bool:
    return _is_ascii_letter(c) or c.isdigit() or c in "-_:"


# ── Tree model ──────────────────────────────────────────────────


class _Text:
    __slots__ = ("data", "is_code")

    def __init__(self, data: str, is_code: bool = False) -> None:
        self.data = data
        self.is_code = is_code

    @property
    def kind(self) -> str:
        return "text"


class _Element:
    __slots__ = ("tag", "attrs", "children", "is_code")

    def __init__(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        self.tag = tag
        self.attrs = attrs
        self.children: list[object] = []
        self.is_code = False

    @property
    def kind(self) -> str:
        return "element"

    def get(self, name: str, default: str = "") -> str:
        for k, v in self.attrs:
            if k == name:
                return v
        return default


def _mark_code(node: object, in_code: bool) -> None:
    if isinstance(node, _Text):
        node.is_code = in_code
        return
    assert isinstance(node, _Element)
    next_in = in_code or node.tag == "code"
    if isinstance(node, _Element) and node.tag == "code":
        node.is_code = True
    for ch in node.children:
        _mark_code(ch, next_in)


def _strip_dropped(node: _Element) -> None:
    keep: list[object] = []
    for ch in node.children:
        if isinstance(ch, _Element):
            if ch.tag in _DROP_TAGS:
                continue
            _strip_dropped(ch)
        keep.append(ch)
    node.children = keep


# ── Whitespace collapse ─────────────────────────────────────────
#
# Mirrors turndown's ``collapse-whitespace`` pass: collapse runs of
# ASCII whitespace ``[ \t\r\n]`` to single spaces inside non-pre
# subtrees, drop leading-on-block-edge spaces, drop trailing space
# at the very end.


_ASCII_WS = " \t\r\n"


def _collapse_runs(s: str) -> str:
    """Collapse runs of ASCII WS to a single space. Non-ASCII
    whitespace (\xa0, etc.) passes through verbatim — matching
    turndown's ``[ \\r\\n\\t]+`` regex."""
    out = []
    in_ws = False
    for c in s:
        if c in _ASCII_WS:
            if not in_ws:
                out.append(" ")
                in_ws = True
        else:
            out.append(c)
            in_ws = False
    return "".join(out)


def _is_block(el: _Element) -> bool:
    return el.tag in _BLOCK_TAGS


def _is_pre(el: _Element) -> bool:
    return el.tag == "pre"


def _is_void(el: _Element) -> bool:
    return el.tag in _VOID_TAGS


class _CollapseState:
    """Mutable state for the whitespace-collapse pass. Lives in
    its own class so ty's narrowing across nested closures works
    cleanly (a ``dict`` would force a wide ``object`` type)."""

    __slots__ = ("prev", "keep_leading")

    def __init__(self) -> None:
        self.prev: _Text | None = None
        self.keep_leading: bool = False


def _collapse_ws(root: _Element) -> None:
    """Apply turndown-style whitespace collapse. Operates in place."""
    state = _CollapseState()

    def walk(el: _Element) -> None:
        if _is_pre(el):
            return
        i = 0
        while i < len(el.children):
            ch = el.children[i]
            if isinstance(ch, _Text):
                text = _collapse_runs(ch.data)
                prev = state.prev
                if (prev is None or prev.data.endswith(" ")) and not state.keep_leading:
                    if text.startswith(" "):
                        text = text[1:]
                if text == "":
                    # Drop empty text node.
                    el.children.pop(i)
                    continue
                ch.data = text
                state.prev = ch
                i += 1
            elif isinstance(ch, _Element):
                if _is_block(ch) or ch.tag == "br":
                    # Trim trailing space on prev — the block break
                    # separates ``prev`` from anything that comes
                    # after it.
                    if state.prev is not None and state.prev.data.endswith(" "):
                        state.prev.data = state.prev.data[:-1]
                    state.prev = None
                    state.keep_leading = False
                    walk(ch)
                    # Same logic when LEAVING the block — the
                    # ``prev`` set inside the block (e.g. li's last
                    # text) has its trailing space stripped at the
                    # boundary.
                    if state.prev is not None and state.prev.data.endswith(" "):
                        state.prev.data = state.prev.data[:-1]
                    state.prev = None
                    state.keep_leading = False
                elif _is_void(ch) or _is_pre(ch):
                    state.prev = None
                    state.keep_leading = True
                    walk(ch)
                else:
                    state.keep_leading = False
                    walk(ch)
                i += 1
            else:
                i += 1

    walk(root)
    if state.prev is not None and state.prev.data.endswith(" "):
        state.prev.data = state.prev.data[:-1]
        if state.prev.data == "":
            # Walk the tree and remove this empty text node.
            _drop_text(root, state.prev)


def _drop_text(el: _Element, target: _Text) -> bool:
    for i, ch in enumerate(el.children):
        if ch is target:
            del el.children[i]
            return True
        if isinstance(ch, _Element):
            if _drop_text(ch, target):
                return True
    return False


# ── Flanking whitespace rule ────────────────────────────────────
#
# turndown moves leading/trailing ASCII whitespace OUT of inline
# elements so the markdown delimiter (e.g. ``_``) sits adjacent to
# a non-space character, which is what CommonMark requires.


def _text_content(node: object) -> str:
    if isinstance(node, _Text):
        return node.data
    assert isinstance(node, _Element)
    parts: list[str] = []
    for ch in node.children:
        parts.append(_text_content(ch))
    return "".join(parts)


def _is_ws(c: str) -> bool:
    """Match Python's regex ``\\s`` semantics for our purposes:
    ASCII whitespace + ``\\xa0`` (NBSP) + a couple common Unicode
    whitespace. We treat anything that ``str.isspace()`` reports
    as whitespace — same as JS' ``\\s`` which includes nbsp."""
    return c.isspace()


def _edge_spans(s: str) -> tuple[str, str, str, str]:
    """Decompose ``s`` into edge whitespace spans matching
    turndown's ``edgeWhitespace`` regex.

    Returns ``(leading_ascii, leading_nonascii, trailing_nonascii,
    trailing_ascii)``. Concatenations:

    * ``leading`` (full)  = ``leading_ascii + leading_nonascii``
    * ``trailing`` (full) = ``trailing_nonascii + trailing_ascii``

    For whitespace-only input, the WHOLE string ends up in the
    leading triple (turndown's m[1]); trailing is empty (m[4]).
    """
    n = len(s)
    if n == 0:
        return "", "", "", ""
    # Forward: ASCII WS run, then any whitespace run.
    i = 0
    while i < n and s[i] in _ASCII_WS:
        i += 1
    j = i
    while j < n and _is_ws(s[j]):
        j += 1
    leading_ascii = s[:i]
    leading_nonascii = s[i:j]
    if j == n:
        # Whitespace-only — turndown's regex puts it ALL in leading,
        # trailing is empty.
        return leading_ascii, leading_nonascii, "", ""
    # Backward: ASCII WS run, then any whitespace run.
    k = n
    while k > j and s[k - 1] in _ASCII_WS:
        k -= 1
    m = k
    while m > j and _is_ws(s[m - 1]):
        m -= 1
    trailing_nonascii = s[m:k]
    trailing_ascii = s[k:n]
    return leading_ascii, leading_nonascii, trailing_nonascii, trailing_ascii


def _is_flanked(el: _Element, parent: _Element, side: str) -> bool:
    """Sibling on ``side`` ends/starts with ASCII space?"""
    siblings = parent.children
    try:
        idx = siblings.index(el)
    except ValueError:
        return False
    if side == "left":
        if idx == 0:
            return False
        sib = siblings[idx - 1]
        if isinstance(sib, _Text):
            return sib.data.endswith(" ")
        if isinstance(sib, _Element) and not _is_block(sib):
            return _text_content(sib).endswith(" ")
        return False
    if idx + 1 >= len(siblings):
        return False
    sib = siblings[idx + 1]
    if isinstance(sib, _Text):
        return sib.data.startswith(" ")
    if isinstance(sib, _Element) and not _is_block(sib):
        return _text_content(sib).startswith(" ")
    return False


def _flanking(el: _Element, parent: _Element) -> tuple[str, str]:
    """``(leading, trailing)`` whitespace to surround ``el``'s
    rendered output with. Matches turndown's behaviour: ASCII WS
    on the inner side gets dropped if the sibling already provides
    an ASCII space; non-ASCII WS (e.g. ``\\xa0``) is always kept
    so it can surface adjacent to the markdown delimiter."""
    if _is_block(el):
        return "", ""
    text = _text_content(el)
    la, lna, tna, ta = _edge_spans(text)
    leading = la + lna
    trailing = tna + ta
    if la and _is_flanked(el, parent, "left"):
        leading = lna
    if ta and _is_flanked(el, parent, "right"):
        trailing = tna
    # ASCII runs collapse to a single space; non-ASCII chars are
    # preserved verbatim (turndown's flanking surfaces ``\xa0``
    # adjacent to the delimiter so escapes/emphasis still parse).
    return _collapse_flank(leading), _collapse_flank(trailing)


def _collapse_flank(s: str) -> str:
    """Collapse a leading/trailing whitespace span: ASCII runs ⇒
    one space; non-ASCII whitespace chars (``\\xa0``, etc.)
    preserved verbatim. Output stays adjacent in source order."""
    out: list[str] = []
    in_ascii = False
    for c in s:
        if c in _ASCII_WS:
            if not in_ascii:
                out.append(" ")
                in_ascii = True
        else:
            out.append(c)
            in_ascii = False
    return "".join(out)


# ── Markdown escape ─────────────────────────────────────────────
#
# Turndown's escape table (``utilities.escapeMarkdown``) — order
# matters because earlier escapes affect later regex matches. We
# process them with explicit string scanning to dodge ``re``.


def _escape_md(s: str) -> str:
    if not s:
        return s
    # 1. Backslash → ``\\``.
    s = s.replace("\\", "\\\\")
    # 2. ``*`` → ``\*``.
    s = s.replace("*", "\\*")
    # 3. ``^-`` → ``\-``: only the leading ``-`` of the string. We
    #    scan only the start.
    if s.startswith("-"):
        s = "\\" + s
    # 4. ``^+ `` → ``\+ ``: leading ``+ ``.
    if s.startswith("+ "):
        s = "\\" + s
    # 5. ``^(=+)`` → ``\$1``: leading run of ``=``.
    if s.startswith("="):
        # Find run length.
        i = 0
        while i < len(s) and s[i] == "=":
            i += 1
        s = "\\" + s
    # 6. ``^(#{1,6}) `` → ``\\$1 ``: leading 1–6 ``#`` followed by space.
    if s.startswith("#"):
        i = 0
        while i < len(s) and s[i] == "#" and i < 6:
            i += 1
        if i >= 1 and i < len(s) and s[i] == " ":
            s = "\\" + s
    # 7. backtick → ``\``` ``.
    s = s.replace("`", "\\`")
    # 8. ``^~~~`` → ``\~~~``: leading three tildes.
    if s.startswith("~~~"):
        s = "\\" + s
    # 9. ``[`` → ``\[``.
    s = s.replace("[", "\\[")
    # 10. ``]`` → ``\]``.
    s = s.replace("]", "\\]")
    # 11. ``^>`` → ``\>``: only the leading ``>``.
    if s.startswith(">"):
        s = "\\" + s
    # 12. ``_`` → ``\_``.
    s = s.replace("_", "\\_")
    # 13. ``^(\d+)\. `` → ``$1\\. `` — leading digits then dot+space.
    if s and s[0].isdigit():
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        if i + 1 < len(s) and s[i] == "." and s[i + 1] == " ":
            s = s[:i] + "\\" + s[i:]
    return s


# ── Renderer ─────────────────────────────────────────────────────
#
# Postorder walk. Each node returns its rendered string, surrounded
# by a directive about block/inline framing; an outer ``join``
# helper merges adjacent outputs with the right number of newlines
# (turndown's ``join`` algorithm).


def _trim_leading_newlines(s: str) -> str:
    i = 0
    while i < len(s) and s[i] == "\n":
        i += 1
    return s[i:]


def _trim_trailing_newlines(s: str) -> str:
    i = len(s)
    while i > 0 and s[i - 1] == "\n":
        i -= 1
    return s[:i]


def _trim_newlines(s: str) -> str:
    return _trim_trailing_newlines(_trim_leading_newlines(s))


def _join(left: str, right: str) -> str:
    """Glue two emitted chunks with the right number of newlines.

    Mirrors turndown's ``join`` — pick the larger of the
    trailing-newline count from ``left`` and the leading-newline
    count from ``right``, capped at 2."""
    s1 = _trim_trailing_newlines(left)
    s2 = _trim_leading_newlines(right)
    nls = max(len(left) - len(s1), len(right) - len(s2))
    if nls > 2:
        nls = 2
    sep = "\n\n"[:nls]
    return s1 + sep + s2


def _render_children(el: _Element, ctx: dict) -> str:
    out = ""
    for ch in el.children:
        out = _join(out, _render(ch, el, ctx))
    return out


def _is_blank(el: _Element) -> bool:
    """``True`` if the element contributes no meaningful content."""
    if _is_void(el):
        return False
    if el.tag in _MEANINGFUL_WHEN_BLANK:
        return False
    text = _text_content(el)
    if text.strip(" \t\r\n"):
        return False
    # Also blank if no nested void / meaningful descendants.
    return not _has_void_or_meaningful(el)


def _has_void_or_meaningful(el: _Element) -> bool:
    for ch in el.children:
        if isinstance(ch, _Element):
            if ch.tag in _VOID_TAGS:
                return True
            if ch.tag in _MEANINGFUL_WHEN_BLANK:
                return True
            if _has_void_or_meaningful(ch):
                return True
    return False


def _list_index(li: _Element, parent: _Element) -> int:
    """0-based index of ``<li>`` among element children of parent."""
    idx = 0
    for ch in parent.children:
        if isinstance(ch, _Element) and ch.tag == "li":
            if ch is li:
                return idx
            idx += 1
    return idx


def _render(node: object, parent: _Element | None, ctx: dict) -> str:  # noqa: PLR0911, PLR0912, PLR0915
    if isinstance(node, _Text):
        if node.is_code:
            return node.data
        return _escape_md(node.data)
    assert isinstance(node, _Element)
    el = node
    tag = el.tag

    # Blank elements: turndown's ``blankRule`` ⇒ ``\n\n`` for
    # block, ``''`` for inline — but ``replacementForNode`` STILL
    # surrounds the result with the element's flanking whitespace.
    # That's how ``<p>Foo<span> </span>Bar</p>`` keeps a space
    # between Foo and Bar even though the span is "blank".
    if _is_blank(el) and tag not in ("pre", "code", "br"):
        if _is_block(el):
            return "\n\n"
        leading, trailing = _flanking(el, parent) if parent else ("", "")
        return leading + trailing

    # Pure passthrough containers: ``<html>`` and ``<body>`` are
    # synthetic; ``<#root>`` is our wrapper.
    if tag in ("html", "body", "#root"):
        return _render_children(el, ctx)

    # Headings.
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(tag[1])
        content = _render_children(el, ctx).strip()
        if level < 3:
            underline = ("=" if level == 1 else "-") * len(content)
            return "\n\n" + content + "\n" + underline + "\n\n"
        return "\n\n" + ("#" * level) + " " + content + "\n\n"

    if tag == "p":
        content = _render_children(el, ctx)
        return "\n\n" + content + "\n\n"

    if tag == "br":
        return "  \n"

    if tag == "hr":
        return "\n\n* * *\n\n"

    if tag in ("strong", "b"):
        content = _render_children(el, ctx)
        if not content.strip():
            return ""
        leading, trailing = _flanking(el, parent) if parent else ("", "")
        return leading + "**" + content.strip() + "**" + trailing

    if tag in ("em", "i"):
        content = _render_children(el, ctx)
        if not content.strip():
            return ""
        leading, trailing = _flanking(el, parent) if parent else ("", "")
        return leading + "_" + content.strip() + "_" + trailing

    if tag in ("s", "strike", "del"):
        content = _render_children(el, ctx)
        if not content.strip():
            return ""
        leading, trailing = _flanking(el, parent) if parent else ("", "")
        return leading + "~" + content.strip() + "~" + trailing

    if tag == "a":
        href = el.get("href", "")
        title = el.get("title", "")
        content = _render_children(el, ctx)
        if not href:
            # Anchors without ``href`` are passthrough (e.g.
            # ``<a id="foo">…</a>``). Still apply flanking.
            leading, trailing = _flanking(el, parent) if parent else ("", "")
            if leading or trailing:
                content = content.strip()
            return leading + content + trailing
        leading, trailing = _flanking(el, parent) if parent else ("", "")
        if leading or trailing:
            content = content.strip()
        href_esc = _escape_link_dest(href)
        if title:
            title_esc = _escape_link_title(title)
            inner = "[" + content + "](" + href_esc + ' "' + title_esc + '")'
        else:
            inner = "[" + content + "](" + href_esc + ")"
        return leading + inner + trailing

    if tag == "img":
        alt = _escape_md(el.get("alt", ""))
        src = el.get("src", "")
        title = el.get("title", "")
        if not src:
            return ""
        if title:
            return "![" + alt + "](" + src + ' "' + title + '")'
        return "![" + alt + "](" + src + ")"

    if tag == "code":
        # Skip if inside ``<pre>`` (handled by ``pre`` rule).
        if parent is not None and parent.tag == "pre":
            return _text_content(el)  # raw — pre handler reads it
        text = _text_content(el)
        if not text:
            return ""
        # Replace any ``\r\n``/``\r``/``\n`` with single space.
        cleaned: list[str] = []
        for c in text:
            if c == "\r" or c == "\n":
                cleaned.append(" ")
            else:
                cleaned.append(c)
        text = "".join(cleaned)
        # Pick delimiter: shortest run of backticks not present in
        # text. Start with one; grow if needed.
        runs = _backtick_runs(text)
        delim = "`"
        while delim in runs:
            delim = delim + "`"
        # Pad with a space if text starts/ends with a backtick or is
        # all-spaces around a non-space.
        needs_pad = False
        if text.startswith("`") or text.endswith("`"):
            needs_pad = True
        elif len(text) >= 2 and text.startswith(" ") and text.endswith(" ") and text.strip(" "):
            # Matches turndown's ``^ .*?[^ ].* $`` — if surrounded
            # by spaces but contains a non-space inside.
            needs_pad = True
        pad = " " if needs_pad else ""
        return delim + pad + text + pad + delim

    if tag == "pre":
        # Indented code block (default option) — only fires when
        # ``<pre>`` has a ``<code>`` first child. Without one, fall
        # through to the default block treatment so plain
        # ``<pre>~~~ foo</pre>`` etc. gets escape-processed text.
        first_child = None
        for ch in el.children:
            if isinstance(ch, _Element):
                first_child = ch
                break
        if first_child is not None and first_child.tag == "code":
            text = _text_content(first_child)
            if not text:
                return "\n\n    \n\n"
            if text.endswith("\n"):
                text = text[:-1]
            prefixed_lines = [("    " + line) if line else "    " for line in text.split("\n")]
            return "\n\n" + "\n".join(prefixed_lines) + "\n\n"
        # Otherwise fall through to default block path below.

    if tag == "blockquote":
        content = _render_children(el, ctx)
        content = _trim_newlines(content)
        # Prefix every line with ``> ``.
        out_lines: list[str] = []
        for line in content.split("\n"):
            out_lines.append("> " + line)
        return "\n\n" + "\n".join(out_lines) + "\n\n"

    if tag in ("ul", "ol"):
        # Block-level wrapping rule from turndown ``rules.list``:
        # if the list is the last element of an ``<li>``, just
        # ``\n + content`` (no double break before/after).
        content = _render_children(el, ctx)
        if parent is not None and parent.tag == "li" and _last_element_child(parent) is el:
            return "\n" + content
        return "\n\n" + content + "\n\n"

    if tag == "li":
        # Determine bullet/number.
        bullet = "*   "
        if parent is not None and parent.tag == "ol":
            start_attr = parent.get("start")
            try:
                start_num = int(start_attr) if start_attr else 1
            except ValueError:
                start_num = 1
            idx = _list_index(el, parent)
            num = start_num + idx
            bullet = str(num) + ".  "
        content = _render_children(el, ctx)
        # turndown's listItem: ``isParagraph = /\n$/.test(content)``.
        # The original content (before trim) ending in ``\n`` means
        # the LI contained a block (paragraph, blockquote, list).
        is_paragraph = content.endswith("\n")
        content = _trim_newlines(content)
        if is_paragraph:
            content = content + "\n"
        # Indent continuation lines by len(prefix).
        indent = " " * len(bullet)
        indented_lines: list[str] = []
        for i, line in enumerate(content.split("\n")):
            if i == 0:
                indented_lines.append(line)
            else:
                indented_lines.append(indent + line if line else indent)
        content = "\n".join(indented_lines)
        # ``+ '\n' if not last sibling`` — turndown adds ``\n``
        # after each li except the last.
        if _has_next_li(el, parent):
            return bullet + content + "\n"
        return bullet + content

    # Default fallback for unknown tags: pass children through. If
    # block-level, wrap with newlines.
    content = _render_children(el, ctx)
    if _is_block(el):
        return "\n\n" + content + "\n\n"
    return content


def _has_next_li(el: _Element, parent: _Element | None) -> bool:
    if parent is None:
        return False
    seen = False
    for ch in parent.children:
        if seen and isinstance(ch, _Element):
            return True
        if ch is el:
            seen = True
    return False


def _last_element_child(el: _Element) -> _Element | None:
    last: _Element | None = None
    for ch in el.children:
        if isinstance(ch, _Element):
            last = ch
    return last


def _backtick_runs(s: str) -> set:
    """All runs of consecutive backticks present in ``s``.

    e.g. ``"a```b`c"`` ⇒ ``{"`", "```"}``."""
    runs: set = set()
    n = len(s)
    i = 0
    while i < n:
        if s[i] == "`":
            j = i
            while j < n and s[j] == "`":
                j += 1
            runs.add(s[i:j])
            i = j
        else:
            i += 1
    return runs


def _escape_link_dest(s: str) -> str:
    """Escape parens in href; if URL contains spaces, wrap in
    ``< >`` (turndown ``escapeLinkDestination``)."""
    if " " in s:
        return "<" + s + ">"
    out = []
    for c in s:
        if c == "(":
            out.append("\\(")
        elif c == ")":
            out.append("\\)")
        else:
            out.append(c)
    return "".join(out)


def _escape_link_title(s: str) -> str:
    """Escape ``"`` in link titles. Newlines are preserved as
    single ``\\n`` (turndown's ``cleanAttribute`` collapses runs of
    ``\\n+\\s*\\n+`` to a single ``\\n`` but otherwise keeps them).
    """
    s = _clean_attr(s)
    out: list[str] = []
    for c in s:
        if c == '"':
            out.append('\\"')
        else:
            out.append(c)
    return "".join(out)


def _clean_attr(s: str) -> str:
    """Turndown's ``cleanAttribute``: replace runs matching
    ``/(\\n+\\s*)+/g`` with a single ``\\n``. The match starts at
    a ``\\n`` and extends greedily through any whitespace, picking
    up further runs of ``\\n+\\s*``. Leading spaces BEFORE the
    first newline are preserved."""
    if "\n" not in s:
        return s
    out: list[str] = []
    n = len(s)
    i = 0
    while i < n:
        c = s[i]
        if c == "\n":
            # Greedy match of (\n+\s*)+. We're already at \n.
            j = i
            while j < n:
                if s[j] != "\n":
                    break
                # \n+ run.
                while j < n and s[j] == "\n":
                    j += 1
                # \s* run.
                while j < n and s[j].isspace() and s[j] != "\n":
                    j += 1
                # Loop continues only if next char is again \n.
            out.append("\n")
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


# ── Streaming tokenizer ──────────────────────────────────────────
#
# The tokenizer holds a buffer of unconsumed input and an index
# into that buffer. Each ``feed_chunk(chunk)`` call appends to the
# buffer, then runs ``_step()`` until either:
#   - we hit an incomplete token (need more input) → return,
#     leaving partial state in the buffer for the next feed
#   - we produce a token → call the consumer callback
#
# On ``finalize()``, run ``_step(eof=True)`` which flushes any
# trailing text run as a token (no incomplete-handling — we do our
# best with what's there).


class _TokenizerError(Exception):
    """Internal — never raised outward."""


class _StreamingTokenizer:
    """Incremental HTML tokenizer. Holds partial-token buffer
    across ``feed`` calls. Emits ``(kind, payload)`` via the
    callback passed to ``__init__``.

    The strategy is "incremental scan": we keep a buffer of input
    we haven't tokenised yet, plus an index of how far we've
    scanned WITHIN that buffer. On each feed we try to extend
    scanning. If a token boundary needs more input (e.g. unfinished
    ``<!--`` comment, an unfinished tag name), we stop, the unread
    portion stays in the buffer, and the next feed extends it.

    On ``finalize()`` we treat partial tokens as best-effort
    completions: any incomplete tag/comment is dropped, any partial
    text run is emitted as-is.
    """

    __slots__ = ("_buf", "_emit", "_in_rawtext", "_rawtext_tag")

    def __init__(self, emit: Callable[[str, object], None]) -> None:
        self._buf: str = ""
        self._emit = emit
        # When inside ``<script>`` or ``<style>`` we're in raw-text
        # mode: drop content until matching ``</tag>``.
        self._in_rawtext: bool = False
        self._rawtext_tag: str = ""

    def feed(self, chunk: str) -> None:
        if chunk:
            self._buf = self._buf + chunk
        self._scan(eof=False)

    def finalize(self) -> None:
        self._scan(eof=True)
        # Anything left in buffer at EOF is dropped (matches
        # turndown's behaviour for unterminated tokens).
        self._buf = ""

    def _scan(self, eof: bool) -> None:  # noqa: PLR0912, PLR0915
        """Try to consume tokens from ``self._buf`` until we hit
        an incomplete token (need more input) or run out of
        buffer.

        Each successful token is emitted via ``self._emit`` and
        the consumed prefix is sliced off the buffer. We loop
        rather than emitting once-per-call so each feed drains
        as much as possible.
        """
        while True:
            buf = self._buf
            n = len(buf)
            if n == 0:
                return

            # ── Raw-text mode ────────────────────────────────
            # Inside ``<script>`` / ``<style>``: scan for the
            # matching ``</tag>`` and drop everything in between.
            if self._in_rawtext:
                close = "</" + self._rawtext_tag
                # Find close case-insensitively. We do this by
                # lowering only the prefix at each potential ``<``
                # — most pages don't have inner ``<`` so the cost
                # is dominated by the final close-tag scan.
                pos = 0
                lower_buf = buf.lower()
                idx = lower_buf.find(close, pos)
                if idx == -1:
                    if eof:
                        # Drop the rest.
                        self._buf = ""
                        return
                    # Need more input. But we can drop everything
                    # up to the LAST possible ``<`` start that
                    # could begin the close — keeps the buffer
                    # bounded. Simplest safe drop: keep the last
                    # ``len(close)`` chars in case a ``<`` is
                    # split.
                    keep = len(close)
                    if n > keep:
                        self._buf = buf[n - keep :]
                    return
                # Found close. Find the ``>`` after it.
                gt = buf.find(">", idx + len(close))
                if gt == -1:
                    if eof:
                        self._buf = ""
                        return
                    return
                self._emit("end", self._rawtext_tag)
                self._in_rawtext = False
                self._rawtext_tag = ""
                self._buf = buf[gt + 1 :]
                continue

            c = buf[0]
            if c == "<" and n >= 2:
                nxt = buf[1]
                # Comment ``<!-- ... -->`` or doctype ``<!DOCTYPE …>``
                if nxt == "!":
                    if buf.startswith("<!--"):
                        end = buf.find("-->", 4)
                        if end == -1:
                            if eof:
                                # Unterminated comment — drop the
                                # rest.
                                self._buf = ""
                                return
                            return
                        self._emit("comment", buf[4:end])
                        self._buf = buf[end + 3 :]
                        continue
                    # Doctype / CDATA — treat as comment (drop).
                    end = buf.find(">", 2)
                    if end == -1:
                        if eof:
                            self._buf = ""
                            return
                        return
                    self._emit("doctype", buf[2:end])
                    self._buf = buf[end + 1 :]
                    continue
                # Processing instruction-ish.
                if nxt == "?":
                    end = buf.find(">", 2)
                    if end == -1:
                        if eof:
                            self._buf = ""
                            return
                        return
                    self._emit("comment", buf[2:end])
                    self._buf = buf[end + 1 :]
                    continue
                # End tag ``</foo>``
                if nxt == "/":
                    if not self._can_complete_end_tag(buf, eof):
                        return
                    j = 2
                    start = j
                    while j < n and _is_tag_name_char(buf[j]):
                        j += 1
                    tag = buf[start:j].lower()
                    while j < n and buf[j] != ">":
                        j += 1
                    if j >= n:
                        # Already checked _can_complete_end_tag,
                        # so this branch is unreachable on the
                        # non-EOF path. EOF: drop.
                        if eof:
                            self._buf = ""
                        return
                    self._emit("end", tag)
                    self._buf = buf[j + 1 :]
                    continue
                # Start tag ``<foo …>``
                if _is_ascii_letter(nxt):
                    if not self._can_complete_start_tag(buf, eof):
                        return
                    consumed = self._consume_start_tag(buf)
                    if consumed is None:
                        # Couldn't parse — fall through to literal.
                        # Treat the ``<`` as text.
                        self._emit_text_run(buf, 1)
                        self._buf = buf[1:]
                        continue
                    new_buf, tag, attrs, self_closing = consumed
                    if tag in _DROP_TAGS and tag in ("script", "style"):
                        # Enter raw-text mode. Emit start, then on
                        # next iteration we'll scan for close.
                        self._emit("start", (tag, attrs, self_closing))
                        self._buf = new_buf
                        if not self_closing:
                            self._in_rawtext = True
                            self._rawtext_tag = tag
                        continue
                    if tag in _VOID_TAGS:
                        self_closing = True
                    self._emit("start", (tag, attrs, self_closing))
                    self._buf = new_buf
                    continue
                # ``<X`` where X is not letter/!?/`/` — literal.
                self._emit_text_run(buf, 1)
                self._buf = buf[1:]
                continue
            elif c == "<" and n == 1:
                if eof:
                    # Stray ``<`` at very end — emit as text.
                    self._emit("text", "<")
                    self._buf = ""
                    return
                # Need more input.
                return

            # Text run.
            i = 0
            while i < n and buf[i] != "<":
                i += 1
            if i == 0:
                # Should be unreachable — buf[0] != '<' was the
                # entry condition.
                return
            if i == n and not eof:
                # Text reaches end of buffer; could still grow.
                # Emit a "safe" prefix (everything up to i, if
                # the last char isn't ``&`` mid-entity).
                #
                # Why split: if we emit the whole buffer now, an
                # entity like ``&amp;`` split across feeds becomes
                # ``&amp`` (literal) + ``;`` next chunk — wrong.
                # We keep the trailing ``&...`` (up to 32 chars)
                # buffered.
                safe = self._safe_text_split(buf)
                if safe == 0:
                    return
                self._emit("text", unescape(buf[:safe]))
                self._buf = buf[safe:]
                continue
            # Either we hit ``<`` or we're at EOF.
            self._emit("text", unescape(buf[:i]))
            self._buf = buf[i:]
            continue

    def _emit_text_run(self, buf: str, n_chars: int) -> None:
        self._emit("text", unescape(buf[:n_chars]))

    def _safe_text_split(self, buf: str) -> int:
        """Pick the largest prefix of ``buf`` that's safe to emit
        without splitting an entity. Returns the prefix length."""
        n = len(buf)
        if n == 0:
            return 0
        # If the last 32 chars contain ``&`` without subsequent
        # ``;``, hold from that ``&`` onward.
        scan_from = max(0, n - 32)
        for i in range(n - 1, scan_from - 1, -1):
            if buf[i] == "&":
                # Is there a ``;`` after it?
                term = buf.find(";", i + 1)
                if term == -1:
                    # Unterminated entity at the tail — split here.
                    return i
                # Found ``;`` — entity is complete, safe up to end.
                return n
        return n

    def _can_complete_end_tag(self, buf: str, eof: bool) -> bool:
        """Do we have the closing ``>`` for an end tag?"""
        gt = buf.find(">", 2)
        if gt != -1:
            return True
        return eof

    def _can_complete_start_tag(self, buf: str, eof: bool) -> bool:
        """Do we have a closing ``>`` for a start tag, accounting
        for quoted attribute values that might contain ``>``?"""
        n = len(buf)
        j = 1
        # Skip tag name.
        while j < n and _is_tag_name_char(buf[j]):
            j += 1
        # Walk attrs, stop at unquoted ``>``.
        while j < n:
            c = buf[j]
            if c == ">":
                return True
            if c == "/" and j + 1 < n and buf[j + 1] == ">":
                return True
            if c in ('"', "'"):
                # Find matching close quote.
                close = buf.find(c, j + 1)
                if close == -1:
                    return eof
                j = close + 1
                continue
            j += 1
        # Hit end of buffer without ``>``.
        return eof

    def _consume_start_tag(  # noqa: PLR0912
        self, buf: str
    ) -> "tuple[str, str, list[tuple[str, str]], bool] | None":
        """Parse a start tag from ``buf``. Returns
        ``(remaining_buf, tag, attrs, self_closing)`` or ``None``
        if the parse failed. Caller has already verified the tag
        is complete (closing ``>`` is present)."""
        n = len(buf)
        j = 1
        start = j
        while j < n and _is_tag_name_char(buf[j]):
            j += 1
        tag = buf[start:j].lower()
        if not tag:
            return None
        attrs: list[tuple[str, str]] = []
        self_closing = False
        while j < n:
            while j < n and buf[j] in " \t\r\n":
                j += 1
            if j >= n:
                break
            cj = buf[j]
            if cj == ">":
                break
            if cj == "/":
                self_closing = True
                j += 1
                continue
            name_start = j
            while j < n and buf[j] not in "=> \t\r\n/":
                j += 1
            name = buf[name_start:j].lower()
            if not name:
                j += 1
                continue
            while j < n and buf[j] in " \t\r\n":
                j += 1
            value = ""
            if j < n and buf[j] == "=":
                j += 1
                while j < n and buf[j] in " \t\r\n":
                    j += 1
                if j < n and buf[j] in ('"', "'"):
                    quote = buf[j]
                    j += 1
                    v_start = j
                    while j < n and buf[j] != quote:
                        j += 1
                    value = buf[v_start:j]
                    if j < n:
                        j += 1
                else:
                    v_start = j
                    while j < n and buf[j] not in "> \t\r\n":
                        j += 1
                    value = buf[v_start:j]
            attrs.append((name, unescape(value)))
        if j < n and buf[j] == ">":
            j += 1
        return buf[j:], tag, attrs, self_closing


# ── Streaming tree builder + emitter ─────────────────────────────


class StreamingMarkdownConverter:
    """Stream HTML chunks in, emit Markdown fragments via
    ``sink``. The converter holds at most one top-level block's
    subtree in memory at any time — once a top-level block closes,
    its tree is rendered and the nodes freed.

    ``sink`` is called with strings. The caller bounds output by
    raising or setting a flag from ``sink`` when the buffer is
    full; the converter doesn't manage output bounding itself, but
    if ``sink`` raises we stop emitting (the exception propagates
    up through ``feed`` / ``close``).

    Usage::

        out: list[str] = []
        c = StreamingMarkdownConverter(sink=out.append)
        c.feed("<p>hello")
        c.feed(" world</p>")
        c.close()
        markdown = "".join(out)
    """

    __slots__ = (
        "_sink",
        "_root",
        "_stack",
        "_tokenizer",
        "_emitted_count",
        "_last_emit_trailing_nl",
        "_closed",
    )

    def __init__(self, sink: Callable[[str], None]) -> None:
        self._sink = sink
        self._root: _Element = _Element("#root", [])
        # Stack of currently-open elements. ``self._root`` is
        # always at index 0.
        self._stack: list[_Element] = [self._root]
        self._tokenizer: _StreamingTokenizer = _StreamingTokenizer(self._on_token)
        # Per-block emission state — match turndown's join cap
        # of 2 newlines between adjacent emits.
        self._emitted_count: int = 0
        self._last_emit_trailing_nl: int = 0
        self._closed: bool = False

    def feed(self, chunk: str) -> None:
        """Feed an arbitrary HTML chunk. May produce zero or more
        ``sink`` calls before returning. Chunks may split a tag
        mid-token — internal state holds the partial token buffer
        until the next chunk completes it."""
        if self._closed:
            raise RuntimeError("StreamingMarkdownConverter: feed() after close()")
        self._tokenizer.feed(chunk)

    def close(self) -> None:
        """Signal end of input. Flushes any pending output via
        ``sink``."""
        if self._closed:
            return
        self._closed = True
        self._tokenizer.finalize()
        # If any open elements remain (unclosed tags), flush them
        # by treating each top-level child of root as a complete
        # block.
        self._flush_remaining()

    # ── Token consumer ──

    def _on_token(self, kind: str, payload: object) -> None:  # noqa: PLR0912
        """Tree-builder callback. Mirrors ``_build_tree`` from the
        non-streaming path, but flushes batches of top-level
        children at safe block boundaries (paragraph / heading /
        hr / etc. close events) so memory is bounded.

        Batching is essential — emitting per-individual-child broke
        sibling whitespace context: ``_render_children`` decides
        flanking whitespace based on what's adjacent in the same
        synthetic root, and a per-child render sees no siblings.
        Batching preserves that context and emits in source order.
        """
        stack = self._stack
        top = stack[-1]
        if kind == "text":
            text = payload
            assert isinstance(text, str)
            if text == "":
                return
            top.children.append(_Text(text))
            return
        if kind == "comment" or kind == "doctype":
            return
        if kind == "start":
            tag, attrs, self_closing = payload  # type: ignore[misc]
            assert isinstance(tag, str)
            # ``<html>`` / ``<body>`` are transparent at root —
            # they don't get treated as blocks themselves; their
            # children become the top-level batch.
            if tag in _ROOT_TRANSPARENT and top is self._root:
                el = _Element(tag, attrs)
                self._root.children.append(el)
                if not self_closing:
                    stack.append(el)
                return
            el = _Element(tag, attrs)
            top.children.append(el)
            if tag in _DROP_TAGS:
                if not self_closing:
                    stack.append(el)
                return
            if not (self_closing or tag in _VOID_TAGS):
                stack.append(el)
                return
            # Self-closing or void at top level. ``<hr/>`` is a
            # block-tag void; flush any preceding batch + the
            # void element together.
            if self._is_at_top_level() and tag in _BLOCK_TAGS:
                self._flush_pending_blocks()
            return
        if kind == "end":
            tag = payload
            assert isinstance(tag, str)
            # Pop until matching open. If never found, ignore
            # (stray close tag).
            popped_any = False
            for idx in range(len(stack) - 1, 0, -1):
                if stack[idx].tag == tag:
                    del stack[idx:]
                    popped_any = True
                    break
            if not popped_any:
                return
            # Flush only when (a) we're back at the top level AND
            # (b) the element we just closed was a block-level tag.
            # Inline-tag closures at the top level (e.g. ``</em>``
            # at root for a doc that's just inline content) DON'T
            # trigger a flush — the inline content stays buffered
            # until a block-tag close lands OR ``close()`` is
            # called. This keeps sibling whitespace context intact.
            if self._is_at_top_level() and tag in _BLOCK_TAGS:
                self._flush_pending_blocks()
            return

    # ── Emission ──

    def _is_at_top_level(self) -> bool:
        """True if the stack is back to root-level (or to inside a
        transparent ``<html>`` / ``<body>`` at root). After closing
        a top-level block, we want to emit it."""
        stack = self._stack
        if len(stack) == 1:
            return True
        if len(stack) == 2 and stack[1].tag in _ROOT_TRANSPARENT:
            return True
        if len(stack) == 3 and stack[1].tag == "html" and stack[2].tag == "body":
            return True
        return False

    def _top_level_container(self) -> _Element:
        """The element whose ``children`` list holds the current
        top-level blocks. Usually ``self._root``, but if the doc
        has ``<html><body>`` we descend to the body."""
        stack = self._stack
        # Walk the stack from the bottom and find the deepest
        # transparent root.
        container = self._root
        if len(stack) >= 2 and stack[1].tag in _ROOT_TRANSPARENT:
            # Find the html/body that's still open.
            container = stack[1]
            if len(stack) >= 3 and stack[2].tag in _ROOT_TRANSPARENT:
                container = stack[2]
            return container
        # Stack might be just [root] but container may have an
        # html/body child that's already closed — in that case
        # children of root are NOT top-level blocks; the
        # html/body's children are. Walk down to the deepest
        # transparent child.
        while True:
            advanced = False
            for ch in container.children:
                if (
                    isinstance(ch, _Element)
                    and ch.tag in _ROOT_TRANSPARENT
                    and not _has_been_emitted(ch)
                ):
                    # Already-emitted markers don't exist; we just
                    # detect by "is still in tree". Keep simple:
                    # if html/body is in tree AND its children
                    # haven't been wiped, treat it as the
                    # container.
                    if ch.children:
                        container = ch
                        advanced = True
                        break
            if not advanced:
                break
        return container

    def _flush_top_level_text(self) -> None:
        """If the top-level container has pending text-only
        children (no element), emit them as one block. Used when
        a new ``<html>`` or ``<body>`` opens to clear any pre-doc
        text."""
        # In practice this is rare; bail if no children.
        container = self._root
        if not container.children:
            return
        # Don't strip — leave them; the next block emission will
        # handle them as part of root.
        return

    def _flush_pending_blocks(self) -> None:
        """Render all pending top-level children in source order
        as a single batch, sink the markdown, and clear them from
        the tree to free memory.

        Batching preserves the sibling whitespace context that
        ``_render_children`` / ``_collapse_ws`` rely on — they
        decide flanking whitespace per-element by looking at
        neighbours within the same root. Splitting siblings into
        independent renders breaks that and produces wrong-order
        / wrong-flanking output (the bug that motivated this
        rewrite).

        Memory ceiling = "what's accumulated since the last
        block-close-flush." For typical pages that's one
        paragraph + its preceding inline runs, ≤ a few KB.
        """
        container = self._top_level_container()
        if not container.children:
            return
        # Build a synthetic root that owns the pending children.
        # Use ``list(...)`` to copy the children reference list —
        # we'll clear the live container's children after building
        # synthetic so the renderer can mutate freely.
        synthetic = _Element("#root", [])
        synthetic.children = list(container.children)
        # Free the live container's children — the synthetic root
        # owns them now.
        container.children = []
        # Mark ``is_code`` on text nodes inside ``<code>`` —
        # turndown uses this to skip markdown-escaping.
        _mark_code(synthetic, False)
        # Drop subtrees whose tag is in ``_DROP_TAGS``.
        _strip_dropped(synthetic)
        # Whitespace collapse on this subtree.
        _collapse_ws(synthetic)
        # Render and emit.
        rendered = _render_children(synthetic, {})
        self._emit_block(rendered)

    def _emit_block(self, rendered: str) -> None:
        """Sink ``rendered`` with proper inter-block newline
        handling. Mirrors turndown's ``join`` algorithm: cap 2
        newlines between adjacent blocks; lstrip leading newlines
        on the first emit; rstrip trailing whitespace on close."""
        if rendered == "":
            return
        # Count leading newlines.
        n = len(rendered)
        i = 0
        while i < n and rendered[i] == "\n":
            i += 1
        leading_nl = i
        # Count trailing newlines.
        j = n
        while j > 0 and rendered[j - 1] == "\n":
            j -= 1
        trailing_nl = n - j
        # Strip trailing whitespace (spaces / tabs) from body —
        # turndown's list / blockquote renderers leave a trailing
        # ``\n    `` indent on the LAST item of the block, expecting
        # the document-level final-strip to drop it. Streaming has
        # no document-level final-strip; we strip per-emit instead.
        # Interior whitespace is unaffected (only the suffix is
        # touched) so multi-line block content (blockquotes, code
        # blocks) keeps its inner structure.
        body = rendered[i:j].rstrip()
        if not body:
            # All-newlines emit (e.g. blank block element). Track
            # the separator demand for the next emit.
            spacing = leading_nl + trailing_nl
            if spacing > 2:
                spacing = 2
            if self._emitted_count > 0 and spacing > self._last_emit_trailing_nl:
                self._last_emit_trailing_nl = spacing
            return
        if self._emitted_count == 0:
            # First emit — drop leading newlines (final lstrip).
            self._sink(body)
        else:
            sep_nls = max(self._last_emit_trailing_nl, leading_nl)
            if sep_nls > 2:
                sep_nls = 2
            if sep_nls > 0:
                self._sink("\n" * sep_nls)
            self._sink(body)
        self._last_emit_trailing_nl = trailing_nl
        self._emitted_count += 1

    def _flush_remaining(self) -> None:
        """At ``close()``, batch-render any still-pending top-level
        children (e.g. unclosed ``<p>foo`` at end-of-doc, OR a doc
        whose root content is entirely inline with no block
        boundaries to trigger an earlier flush). Reuses
        ``_flush_pending_blocks`` so the close-time render path
        and the mid-stream render path share the same sibling-
        whitespace-context behaviour."""
        self._flush_pending_blocks()


def _has_been_emitted(_el: _Element) -> bool:
    """Stub — we don't currently mark emitted nodes; emission
    pops them from the tree, so anything still in tree is by
    definition unemitted. Kept as a hook for future refactors."""
    return False


# ── Public API ──────────────────────────────────────────────────


class MarkdownConverter:
    """Stateless wrapper around :func:`convert`. Provided for users
    who prefer an object handle (e.g. to subclass or to swap in a
    mock for testing). The function form is the canonical entry."""

    def convert(self, html: str) -> str:
        return convert(html)


def convert(html: str) -> str:
    """Convert an HTML string to Markdown using turndown's default
    options. Returns ``""`` for empty / falsy input.

    Backwards-compat wrapper around :class:`StreamingMarkdownConverter`
    — feeds the whole string at once and joins the sink output."""
    if not html:
        return ""
    out: list[str] = []
    c = StreamingMarkdownConverter(sink=out.append)
    c.feed(html)
    c.close()
    return "".join(out)


__all__ = ["MarkdownConverter", "StreamingMarkdownConverter", "convert"]
