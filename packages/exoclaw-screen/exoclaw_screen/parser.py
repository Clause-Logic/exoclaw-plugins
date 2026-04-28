"""Markdown + IAL + Pandoc fenced-div parser for screen.md.

Pure-Python state machine — no ``re``. MicroPython 1.27's regex
engine rejects negative lookahead and limits group depth, so the
patterns in CPython's ``html.parser`` (and the html2text deps that
need it) blow up on chip. Same lesson applied here: parsing is
linear character-by-character with explicit state tracking.

Two layers:

1. **Block parser** — line-oriented. Walks ``source.splitlines()``
   and emits block AST nodes. Handles ``:::`` fenced-div nesting,
   blockquotes, code blocks, lists, headings, paragraphs.

2. **Inline parser** — character-oriented. Walks within a block's
   text content and emits inline nodes. Handles bold/italic/code
   delimiter pairing, link bracket-and-paren matching, hard breaks,
   image directives.

The IAL helper (``parse_trailing_ial``) recognises a trailing
``{.class attr=value}`` block on a line and pulls it off so the
block parser sees clean content. Pandoc-style — supports ``.class``
class shorthand and bare ``key=value`` attributes (no quoted
values in v0).

The grammar is documented in ``SKILL.md``; this file's job is to
mirror it correctly. Out-of-scope tokens (tables, nested lists,
reference links, strikethrough, HTML pass-through) are tolerated
and rendered as plain text rather than raising.
"""

from __future__ import annotations

from typing import Any

from exoclaw_screen import ast as a

# ── IAL ──────────────────────────────────────────────────────────


def parse_ial(text: str) -> "dict[str, Any]":
    """Parse a single IAL block ``{.class attr=value}`` (no braces
    in input) into an attrs dict.

    Class shorthand keys land under ``"class"`` as a list (order
    preserved). ``key=value`` attributes land as their own keys with
    string values. Empty input returns an empty dict.

    Quoted values (``key="val with spaces"``) are NOT supported in
    v0 — values are bare tokens terminated by whitespace.
    """
    out: dict[str, Any] = {}
    classes: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        # Skip whitespace.
        while i < n and text[i] in " \t":
            i += 1
        if i >= n:
            break
        if text[i] == ".":
            # Class shorthand — read until whitespace.
            i += 1
            start = i
            while i < n and text[i] not in " \t":
                i += 1
            cls = text[start:i]
            if cls:
                classes.append(cls)
        else:
            # ``key`` or ``key=value`` token.
            start = i
            while i < n and text[i] not in " \t=":
                i += 1
            key = text[start:i]
            if i < n and text[i] == "=":
                i += 1
                vstart = i
                while i < n and text[i] not in " \t":
                    i += 1
                # Empty-key guard — input like ``"=foo"`` would
                # otherwise produce ``{"": "foo"}``. Skip silently;
                # malformed IAL shouldn't leak through to renderers.
                if key:
                    out[key] = text[vstart:i]
            else:
                # Bare key — store as ``True`` so caller can test
                # presence with ``"key" in attrs``.
                if key:
                    out[key] = True
    if classes:
        out["class"] = classes
    return out


def parse_trailing_ial(line: str) -> "tuple[str, dict[str, Any]]":
    """Strip a trailing ``{.class attr=value}`` IAL block from
    ``line`` and return ``(line_without_ial, attrs)``.

    If the line doesn't end with a balanced ``{...}``, returns
    ``(line, {})``. Brace-matching walks backwards from the right —
    handles nested braces inside the IAL only at the top level
    (v0 doesn't expect nested braces).
    """
    s = line.rstrip()
    if not s.endswith("}"):
        return line, {}
    # Walk back from the closing ``}`` to find the matching ``{``.
    depth = 0
    j = len(s) - 1
    found = -1
    while j >= 0:
        c = s[j]
        if c == "}":
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0:
                found = j
                break
        j -= 1
    if found == -1:
        return line, {}
    inner = s[found + 1 : -1]
    attrs = parse_ial(inner)
    if not attrs:
        # ``{}`` empty — leave the line alone.
        return line, {}
    # Strip the IAL plus any leading whitespace before the ``{``.
    head = s[:found].rstrip()
    return head, attrs


def _trailing_ial_belongs_to_image(line: str) -> bool:
    """Detect if a line's trailing ``{...}`` is the IAL of a
    closing image directive ``![alt](url){.qrcode}`` rather than
    a block-level IAL.

    Walks back from the trailing ``}`` to find the matching ``{``;
    if the char immediately before that ``{`` is ``)`` AND somewhere
    earlier on the same line is ``![``, the IAL belongs to the image.
    Used by paragraph + blockquote IAL stripping to avoid stealing
    the image directive's class.
    """
    s = line.rstrip()
    if not s.endswith("}"):
        return False
    depth = 0
    j = len(s) - 1
    found = -1
    while j >= 0:
        c = s[j]
        if c == "}":
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0:
                found = j
                break
        j -= 1
    if found <= 0:
        return False
    # Char immediately before ``{`` (skip trailing whitespace
    # between ``)`` and ``{`` — most images write them flush).
    k = found - 1
    while k >= 0 and s[k] in " \t":
        k -= 1
    if k < 0 or s[k] != ")":
        return False
    # Look earlier for an ``![`` opening on the same line.
    return "![" in s[:k]


# ── Inline parser ────────────────────────────────────────────────


def _parse_inline(src: str) -> "list[Any]":
    """Convert a block's text content into a list of inline AST
    nodes. Walks character-by-character; pairs delimiters by stack
    discipline (no regex)."""
    out: list[Any] = []
    buf: list[str] = []
    i = 0
    n = len(src)

    def flush_text() -> None:
        if buf:
            out.append(a.Text("".join(buf)))
            buf.clear()

    while i < n:
        c = src[i]

        # Hard break: two trailing spaces before a newline.
        if c == " " and i + 2 < n and src[i + 1] == " " and src[i + 2] == "\n":
            flush_text()
            out.append(a.HardBreak())
            i += 3
            continue

        # Escaped char — pass next char through verbatim.
        if c == "\\" and i + 1 < n:
            buf.append(src[i + 1])
            i += 2
            continue

        # Inline code: backticks. Match opening run length, scan
        # for closing run of same length.
        if c == "`":
            run = 0
            while i + run < n and src[i + run] == "`":
                run += 1
            close = src.find("`" * run, i + run)
            if close != -1 and close > i + run:
                # Don't accidentally match an opening run that's
                # part of a longer run at the close site.
                if close + run < n and src[close + run] == "`":
                    # Close site has more backticks than expected —
                    # not the matching close.
                    pass
                else:
                    flush_text()
                    out.append(a.InlineCode(src[i + run : close]))
                    i = close + run
                    continue
            # No close found — treat backticks as literal text.
            buf.append(c * run)
            i += run
            continue

        # Bold (``**``).
        if c == "*" and i + 1 < n and src[i + 1] == "*":
            close = src.find("**", i + 2)
            if close != -1:
                flush_text()
                out.append(a.Bold(_parse_inline(src[i + 2 : close])))
                i = close + 2
                continue
            # No close — literal.
            buf.append("**")
            i += 2
            continue

        # Italic via ``_``.
        if c == "_":
            close = src.find("_", i + 1)
            if close != -1 and close > i + 1:
                flush_text()
                out.append(a.Italic(_parse_inline(src[i + 1 : close])))
                i = close + 1
                continue
            buf.append(c)
            i += 1
            continue

        # Italic via ``*`` (not ``**``, that's bold).
        if c == "*":
            close = src.find("*", i + 1)
            if close != -1 and close > i + 1:
                # Don't consume across a ``**`` token.
                if close + 1 < n and src[close + 1] == "*":
                    pass
                else:
                    flush_text()
                    out.append(a.Italic(_parse_inline(src[i + 1 : close])))
                    i = close + 1
                    continue
            buf.append(c)
            i += 1
            continue

        # Image: ``![alt](src){...}``.
        if c == "!" and i + 1 < n and src[i + 1] == "[":
            close_alt = src.find("]", i + 2)
            if close_alt != -1 and close_alt + 1 < n and src[close_alt + 1] == "(":
                close_paren = src.find(")", close_alt + 2)
                if close_paren != -1:
                    flush_text()
                    alt = src[i + 2 : close_alt]
                    paren_inner = src[close_alt + 2 : close_paren]
                    src_url, _title = _split_url_title(paren_inner)
                    j = close_paren + 1
                    # Optional trailing ``{...}`` IAL on the image.
                    attrs: dict[str, Any] = {}
                    if j < n and src[j] == "{":
                        # Find matching close brace.
                        depth = 1
                        k = j + 1
                        while k < n and depth > 0:
                            if src[k] == "{":
                                depth += 1
                            elif src[k] == "}":
                                depth -= 1
                            k += 1
                        if depth == 0:
                            attrs = parse_ial(src[j + 1 : k - 1])
                            j = k
                    out.append(a.Image(src=src_url, alt=alt, attrs=attrs))
                    i = j
                    continue
            buf.append(c)
            i += 1
            continue

        # Link: ``[text](url)``.
        if c == "[":
            close_text = src.find("]", i + 1)
            if close_text != -1 and close_text + 1 < n and src[close_text + 1] == "(":
                close_paren = src.find(")", close_text + 2)
                if close_paren != -1:
                    flush_text()
                    inner = src[i + 1 : close_text]
                    paren_inner = src[close_text + 2 : close_paren]
                    url, title = _split_url_title(paren_inner)
                    out.append(a.Link(text=_parse_inline(inner), url=url, title=title))
                    i = close_paren + 1
                    continue
            buf.append(c)
            i += 1
            continue

        # Plain character.
        buf.append(c)
        i += 1

    flush_text()
    return out


def _split_url_title(inner: str) -> "tuple[str, str]":
    """Split the contents of a ``(url "title")`` link/image paren
    into (url, title). Title is the optional double-quoted suffix
    after whitespace. Returns ``("", "")`` for empty input.
    """
    s = inner.strip()
    if not s:
        return "", ""
    # Find first unquoted whitespace.
    i = 0
    n = len(s)
    while i < n and s[i] not in " \t":
        i += 1
    if i >= n:
        return s, ""
    url = s[:i]
    rest = s[i:].strip()
    # Title is optional, double-quoted.
    if rest.startswith('"') and rest.endswith('"') and len(rest) >= 2:
        return url, rest[1:-1]
    return url, ""


# ── Block parser ─────────────────────────────────────────────────


class _LineCursor:
    """Helper that walks lines with ``peek`` / ``advance`` semantics
    and supports a small lookahead for fenced-block detection."""

    def __init__(self, lines: "list[str]") -> None:
        self._lines = lines
        self._i = 0

    def at_eof(self) -> bool:
        return self._i >= len(self._lines)

    def peek(self) -> str:
        if self._i >= len(self._lines):
            return ""
        return self._lines[self._i]

    def advance(self) -> str:
        line = self.peek()
        self._i += 1
        return line


def _is_fence_open(line: str) -> "tuple[bool, dict[str, Any]]":
    """Detect ``::: {...}`` fenced-div opening line. Returns
    ``(True, attrs)`` if matched; ``(False, {})`` otherwise."""
    s = line.strip()
    if not s.startswith(":::"):
        return False, {}
    rest = s[3:].strip()
    if not rest:
        return False, {}
    if rest.startswith("{") and rest.endswith("}"):
        attrs = parse_ial(rest[1:-1])
        if attrs:
            return True, attrs
    return False, {}


def _is_fence_close(line: str) -> bool:
    return line.strip() == ":::"


def _is_hr(line: str) -> bool:
    """``---`` on its own line. Markdown also allows ``***`` /
    ``___``; we accept those too for grammar tolerance."""
    s = line.strip()
    if len(s) < 3:
        return False
    if s[0] not in "-*_":
        return False
    return all(c == s[0] for c in s)


def _container_kind_from_attrs(attrs: "dict[str, Any]") -> str:
    """Pick a container kind from IAL classes — first recognised
    class wins; falls back to ``"col"`` (default vertical layout)."""
    classes = attrs.get("class") or []
    for cls in classes:
        if cls in ("row", "col", "grid"):
            return cls
    return "col"


def _is_standalone_ial_line(line: str) -> "tuple[bool, dict[str, Any]]":
    """Return ``(True, attrs)`` if ``line`` (after strip) is just an
    IAL block ``{...}`` and nothing else. Used to detect the
    list-as-whole IAL form where ``{.cols=2}`` lives on its own line
    immediately above the first ``- item``.
    """
    s = line.strip()
    if not (s.startswith("{") and s.endswith("}") and len(s) >= 2):
        return False, {}
    attrs = parse_ial(s[1:-1])
    if not attrs:
        return False, {}
    return True, attrs


def _parse_blocks(cur: _LineCursor, depth: int = 0) -> "list[Any]":
    """Parse a block sequence, returning when the cursor hits EOF
    OR a ``:::`` close fence at the current depth.
    """
    out: list[Any] = []
    # ``pending_list_attrs`` carries a standalone-IAL line's attrs
    # forward to the next list block (and ONLY a list block — if a
    # different block comes between, the IAL is dropped on the floor
    # to keep the grammar tight). See list-as-whole IAL rule:
    # ordering is "IAL line, optional blank lines, then list".
    pending_list_attrs: dict[str, Any] = {}
    while not cur.at_eof():
        line = cur.peek()
        # ``:::`` close fence — return to caller (parent fence).
        if depth > 0 and _is_fence_close(line):
            cur.advance()
            return out
        # Skip blank lines.
        if not line.strip():
            cur.advance()
            continue
        # Standalone IAL line: ``{.cols=2 align=left}`` on its own.
        # If the NEXT non-blank line opens a list, we attach to the
        # list. If it doesn't, drop the line (don't emit a paragraph
        # of literal ``{...}`` text — that's noise the agent didn't
        # mean).
        is_ial, ial_attrs = _is_standalone_ial_line(line)
        if is_ial:
            # Peek ahead past blanks for the next non-blank line.
            saved_i = cur._i
            cur.advance()
            while not cur.at_eof() and not cur.peek().strip():
                cur.advance()
            if not cur.at_eof() and _detect_list_marker(cur.peek()) is not None:
                pending_list_attrs = ial_attrs
                # Don't reset cursor — we've consumed the IAL line
                # plus any blanks before the list.
                continue
            # Not followed by a list — restore cursor, treat the
            # line as a regular paragraph (fall through). This
            # preserves "an unrelated paragraph that ends with
            # ``{...}`` shouldn't have its IAL stolen by a list
            # that follows" — here the IAL stands alone.
            cur._i = saved_i
            # Fall through to paragraph collection below.
        # ``::: {.row}`` open fence — recurse.
        is_open, attrs = _is_fence_open(line)
        if is_open:
            cur.advance()
            kind = _container_kind_from_attrs(attrs)
            children = _parse_blocks(cur, depth + 1)
            out.append(a.Container(kind=kind, attrs=attrs, children=children))
            pending_list_attrs = {}
            continue
        # Heading: ``# `` to ``###### ``.
        if line.startswith("#"):
            level = 0
            while level < len(line) and line[level] == "#":
                level += 1
            if 1 <= level <= 6 and level < len(line) and line[level] == " ":
                cur.advance()
                rest = line[level + 1 :]
                rest, attrs = parse_trailing_ial(rest)
                out.append(a.Heading(level=level, content=_parse_inline(rest), attrs=attrs))
                pending_list_attrs = {}
                continue
        # Horizontal rule.
        if _is_hr(line):
            cur.advance()
            out.append(a.HorizontalRule())
            pending_list_attrs = {}
            continue
        # Fenced code block: ``\`\`\`...\`\`\```.
        if line.startswith("```"):
            cur.advance()
            info = line[3:].strip()
            # Info string: ``lang`` token (first whitespace-delimited
            # word), then optional trailing ``{...}`` IAL.
            cb_attrs: dict[str, Any] = {}
            if info.endswith("}"):
                stripped_info, cb_attrs = parse_trailing_ial(info)
                if cb_attrs:
                    info = stripped_info
            # First whitespace-delimited word is the lang.
            lang = info.strip()
            # If there's still extra cruft after a space (no IAL but
            # multiple words), only the first is the lang.
            for sp in (" ", "\t"):
                idx = lang.find(sp)
                if idx != -1:
                    lang = lang[:idx]
                    break
            buf: list[str] = []
            while not cur.at_eof():
                inner = cur.peek()
                if inner.startswith("```"):
                    cur.advance()
                    break
                buf.append(cur.advance())
            out.append(a.CodeBlock(text="\n".join(buf), lang=lang, attrs=cb_attrs))
            pending_list_attrs = {}
            continue
        # Blockquote: ``>`` line prefix (one or more contiguous lines).
        if line.startswith(">"):
            quote_lines: list[str] = []
            while not cur.at_eof() and cur.peek().startswith(">"):
                ln = cur.advance()
                # Strip the leading ``>`` plus optional space.
                if ln.startswith("> "):
                    quote_lines.append(ln[2:])
                else:
                    quote_lines.append(ln[1:])
            # Blockquote-level IAL: same rule as paragraph — trailing
            # ``{...}`` on the last non-blank line, unless it's an
            # image directive's IAL.
            bq_attrs: dict[str, Any] = {}
            last_idx = len(quote_lines) - 1
            while last_idx >= 0 and not quote_lines[last_idx].strip():
                last_idx -= 1
            if last_idx >= 0:
                last_line = quote_lines[last_idx]
                if not _trailing_ial_belongs_to_image(last_line):
                    stripped, bq_attrs = parse_trailing_ial(last_line)
                    if bq_attrs:
                        quote_lines[last_idx] = stripped
            quote_text = "\n".join(quote_lines)
            sub_cursor = _LineCursor(quote_text.splitlines())
            children = _parse_blocks(sub_cursor)
            out.append(a.Blockquote(content=children, attrs=bq_attrs))
            pending_list_attrs = {}
            continue
        # List: unordered (``- `` / ``* ``) or ordered (``1. ``).
        list_match = _detect_list_marker(line)
        if list_match is not None:
            ordered = list_match
            items: list[a.ListItem] = []
            while not cur.at_eof():
                ln = cur.peek()
                marker = _detect_list_marker(ln)
                if marker is None or marker != ordered:
                    break
                cur.advance()
                content_text = _strip_list_marker(ln, ordered)
                items.append(a.ListItem(content=_parse_inline(content_text)))
            out.append(a.ListBlock(ordered=ordered, items=items, attrs=pending_list_attrs))
            pending_list_attrs = {}
            continue
        # Paragraph: collect contiguous non-blank, non-special lines.
        para_lines: list[str] = []
        while not cur.at_eof():
            ln = cur.peek()
            if not ln.strip():
                break
            if _line_starts_block(ln):
                break
            para_lines.append(cur.advance())
        if para_lines:
            # Paragraph-level IAL: strip a trailing ``{...}`` from
            # the paragraph's last line UNLESS that line ends with
            # an image-directive close ``)`` immediately before the
            # IAL — in which case the IAL belongs to the image.
            # See ``_trailing_ial_belongs_to_image`` for the rule.
            attrs: dict[str, Any] = {}
            last = para_lines[-1]
            if not _trailing_ial_belongs_to_image(last):
                stripped, attrs = parse_trailing_ial(last)
                if attrs:
                    para_lines[-1] = stripped
            text = "\n".join(para_lines)
            out.append(a.Paragraph(content=_parse_inline(text), attrs=attrs))
            pending_list_attrs = {}
    return out


def _detect_list_marker(line: str) -> "bool | None":
    """``- `` / ``* `` → False (unordered), ``N. `` → True
    (ordered), anything else → None."""
    if line.startswith(("- ", "* ")):
        return False
    # Ordered: ``digits + "." + " "`` at start.
    i = 0
    while i < len(line) and line[i].isdigit():
        i += 1
    if i > 0 and i + 1 < len(line) and line[i] == "." and line[i + 1] == " ":
        return True
    return None


def _strip_list_marker(line: str, ordered: bool) -> str:
    """Drop the list-marker prefix from a list-item line."""
    if not ordered:
        return line[2:]
    i = 0
    while i < len(line) and line[i].isdigit():
        i += 1
    # Skip the ``"."`` and the space.
    return line[i + 2 :]


def _line_starts_block(line: str) -> bool:
    """True if a line opens some block (heading / hr / fence / list
    / blockquote / fenced code) — used by paragraph collection to
    stop greedy line accumulation. ``#``-prefixed lines only count
    as a block-start if they actually form a valid heading
    (1–6 hashes followed by space); ``####### foo`` is NOT a
    heading and SHOULD flow into paragraph collection."""
    if line.startswith("#"):
        # Count the leading-hash run, then check the heading-shape
        # constraint (1–6 hashes + space). Anything else is plain
        # text — fall through to paragraph.
        level = 0
        while level < len(line) and line[level] == "#":
            level += 1
        if 1 <= level <= 6 and level < len(line) and line[level] == " ":
            return True
        return False
    if line.startswith(">"):
        return True
    if line.startswith("```"):
        return True
    if line.startswith(":::"):
        return True
    if _is_hr(line):
        return True
    if _detect_list_marker(line) is not None:
        return True
    return False


# ── Public entry point ───────────────────────────────────────────


def parse(source: str) -> a.Document:
    """Parse a markdown screen source into a ``Document`` AST."""
    lines = source.splitlines()
    cur = _LineCursor(lines)
    children = _parse_blocks(cur)
    return a.Document(children=children)
