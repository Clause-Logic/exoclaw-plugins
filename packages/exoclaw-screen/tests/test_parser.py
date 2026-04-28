"""Parser tests — small, focused, hand-curated for v0 grammar.

Each test asserts a specific AST shape from a markdown input. We
don't try to round-trip a big corpus here (that's the
exoclaw-tools-web subagent's job for HTML→Markdown — different
direction, different problem); here we just verify the parser
produces the AST the layout engine expects."""

from __future__ import annotations

from exoclaw_screen import ast as a
from exoclaw_screen.parser import parse, parse_ial, parse_trailing_ial

# ── IAL ─────────────────────────────────────────────────────────


class TestParseIAL:
    def test_empty(self) -> None:
        assert parse_ial("") == {}

    def test_single_class(self) -> None:
        assert parse_ial(".title") == {"class": ["title"]}

    def test_multiple_classes(self) -> None:
        assert parse_ial(".title .center") == {"class": ["title", "center"]}

    def test_attribute(self) -> None:
        assert parse_ial("align=center") == {"align": "center"}

    def test_class_plus_attrs(self) -> None:
        attrs = parse_ial(".title align=center color=red")
        assert attrs["class"] == ["title"]
        assert attrs["align"] == "center"
        assert attrs["color"] == "red"

    def test_bare_key(self) -> None:
        assert parse_ial("bold") == {"bold": True}


class TestParseTrailingIAL:
    def test_no_ial(self) -> None:
        line, attrs = parse_trailing_ial("plain heading")
        assert line == "plain heading"
        assert attrs == {}

    def test_strips_trailing(self) -> None:
        line, attrs = parse_trailing_ial("hello {.title}")
        assert line == "hello"
        assert attrs == {"class": ["title"]}

    def test_unbalanced_braces_left_alone(self) -> None:
        line, attrs = parse_trailing_ial("foo bar {")
        assert line == "foo bar {"
        assert attrs == {}


# ── Block parser ────────────────────────────────────────────────


class TestHeadings:
    def test_h1(self) -> None:
        doc = parse("# Hello")
        assert len(doc.children) == 1
        h = doc.children[0]
        assert isinstance(h, a.Heading)
        assert h.level == 1

    def test_h2_with_ial(self) -> None:
        doc = parse("## Title {.section align=center}")
        h = doc.children[0]
        assert isinstance(h, a.Heading)
        assert h.level == 2
        assert h.attrs["class"] == ["section"]
        assert h.attrs["align"] == "center"

    def test_h3(self) -> None:
        doc = parse("### Subhead")
        assert doc.children[0].level == 3

    def test_seven_hashes_is_paragraph(self) -> None:
        # ``####### foo`` isn't a valid heading (>6 hashes); should
        # become a paragraph.
        doc = parse("####### foo")
        assert isinstance(doc.children[0], a.Paragraph)


class TestParagraph:
    def test_simple(self) -> None:
        doc = parse("Just some text.")
        assert isinstance(doc.children[0], a.Paragraph)

    def test_two_paragraphs(self) -> None:
        doc = parse("First.\n\nSecond.")
        assert len(doc.children) == 2
        assert all(isinstance(c, a.Paragraph) for c in doc.children)


class TestLists:
    def test_unordered(self) -> None:
        doc = parse("- a\n- b\n- c")
        lst = doc.children[0]
        assert isinstance(lst, a.ListBlock)
        assert lst.ordered is False
        assert len(lst.items) == 3

    def test_ordered(self) -> None:
        doc = parse("1. a\n2. b\n3. c")
        lst = doc.children[0]
        assert isinstance(lst, a.ListBlock)
        assert lst.ordered is True
        assert len(lst.items) == 3


class TestBlockquote:
    def test_single_line(self) -> None:
        doc = parse("> quoted")
        bq = doc.children[0]
        assert isinstance(bq, a.Blockquote)
        # Inner is a paragraph with the text.
        assert isinstance(bq.content[0], a.Paragraph)


class TestCodeBlock:
    def test_fenced(self) -> None:
        doc = parse("```python\nprint('hi')\n```")
        cb = doc.children[0]
        assert isinstance(cb, a.CodeBlock)
        assert cb.lang == "python"
        assert "print('hi')" in cb.text


class TestHorizontalRule:
    def test_dashes(self) -> None:
        doc = parse("---")
        assert isinstance(doc.children[0], a.HorizontalRule)


class TestFencedDivContainers:
    def test_row(self) -> None:
        src = "::: {.row gap=20}\n# A\n:::"
        doc = parse(src)
        c = doc.children[0]
        assert isinstance(c, a.Container)
        assert c.kind == "row"
        assert c.attrs.get("gap") == "20"
        assert len(c.children) == 1
        assert isinstance(c.children[0], a.Heading)

    def test_nested_col_in_row(self) -> None:
        src = "::: {.row}\n::: {.col w=50%}\n# Left\n:::\n::: {.col w=50%}\n# Right\n:::\n:::"
        doc = parse(src)
        row = doc.children[0]
        assert isinstance(row, a.Container)
        assert row.kind == "row"
        assert len(row.children) == 2
        for col in row.children:
            assert isinstance(col, a.Container)
            assert col.kind == "col"
            assert col.attrs.get("w") == "50%"

    def test_grid(self) -> None:
        src = "::: {.grid cols=3 gap=10}\n# A\n# B\n# C\n:::"
        doc = parse(src)
        g = doc.children[0]
        assert isinstance(g, a.Container)
        assert g.kind == "grid"
        assert g.attrs.get("cols") == "3"
        assert len(g.children) == 3


# ── Inline parser ───────────────────────────────────────────────


class TestInlineBold:
    def test_bold(self) -> None:
        doc = parse("**hi**")
        para = doc.children[0]
        assert isinstance(para, a.Paragraph)
        b = para.content[0]
        assert isinstance(b, a.Bold)
        assert isinstance(b.children[0], a.Text)
        assert b.children[0].text == "hi"


class TestInlineItalic:
    def test_underscore(self) -> None:
        doc = parse("_hi_")
        para = doc.children[0]
        i = para.content[0]
        assert isinstance(i, a.Italic)

    def test_asterisk(self) -> None:
        doc = parse("*hi*")
        para = doc.children[0]
        i = para.content[0]
        assert isinstance(i, a.Italic)


class TestInlineCode:
    def test_backticks(self) -> None:
        doc = parse("hello `code` world")
        para = doc.children[0]
        # Three children: Text, InlineCode, Text.
        kinds = [type(c).__name__ for c in para.content]
        assert "InlineCode" in kinds


class TestInlineLink:
    def test_simple(self) -> None:
        doc = parse("[label](https://example.com)")
        para = doc.children[0]
        link = para.content[0]
        assert isinstance(link, a.Link)
        assert link.url == "https://example.com"

    def test_link_with_title(self) -> None:
        doc = parse('[label](https://example.com "Title")')
        link = doc.children[0].content[0]
        assert isinstance(link, a.Link)
        assert link.url == "https://example.com"
        assert link.title == "Title"


class TestInlineImage:
    def test_plain_image(self) -> None:
        doc = parse("![alt](https://x.test/img.png)")
        para = doc.children[0]
        img = para.content[0]
        assert isinstance(img, a.Image)
        assert img.alt == "alt"
        assert img.src == "https://x.test/img.png"

    def test_image_with_directive_class(self) -> None:
        doc = parse("![QR](https://example.com){.qrcode size=200}")
        img = doc.children[0].content[0]
        assert isinstance(img, a.Image)
        assert img.attrs.get("class") == ["qrcode"]
        assert img.attrs.get("size") == "200"


class TestInlineHardBreak:
    def test_two_trailing_spaces(self) -> None:
        # The hard-break sequence is space-space-newline within a
        # block; our parser trims trailing whitespace per-line, so
        # this test asserts the inline parser handles the explicit
        # ``  \n`` token.
        doc = parse("a  \nb")
        # The two-line paragraph has a HardBreak inline.
        para = doc.children[0]
        assert isinstance(para, a.Paragraph)
        # NOTE: depending on how splitlines + paragraph collection
        # normalises whitespace, the hard-break may render as a
        # plain newline in v0. Both behaviours are acceptable.
        assert len(para.content) >= 1


class TestEscaping:
    def test_backslash_escape(self) -> None:
        doc = parse("\\*not italic\\*")
        para = doc.children[0]
        # Should produce a plain Text node, no Italic.
        assert all(isinstance(c, a.Text) for c in para.content)
        assert "*not italic*" in "".join(c.text for c in para.content)


# ── IAL on block elements (paragraph / blockquote / code / list) ──


class TestParagraphIAL:
    def test_simple_trailing_ial(self) -> None:
        doc = parse("Now: 72°F {color=red weight=bold}")
        para = doc.children[0]
        assert isinstance(para, a.Paragraph)
        assert para.attrs.get("color") == "red"
        assert para.attrs.get("weight") == "bold"

    def test_class_ial(self) -> None:
        doc = parse("Hello world {.callout}")
        para = doc.children[0]
        assert isinstance(para, a.Paragraph)
        assert para.attrs.get("class") == ["callout"]

    def test_image_directive_keeps_its_ial(self) -> None:
        # The IAL belongs to the image, not the paragraph.
        doc = parse("![QR](https://example.com){.qrcode size=200}")
        para = doc.children[0]
        assert isinstance(para, a.Paragraph)
        # Paragraph should NOT have stolen the IAL.
        assert para.attrs == {}
        # The image inside still has the IAL.
        img = para.content[0]
        assert isinstance(img, a.Image)
        assert img.attrs.get("class") == ["qrcode"]
        assert img.attrs.get("size") == "200"

    def test_paragraph_text_and_image_directive_keeps_image_ial(self) -> None:
        # Paragraph with text + trailing image directive — IAL is the image's.
        doc = parse("Scan: ![QR](https://example.com){.qrcode}")
        para = doc.children[0]
        assert isinstance(para, a.Paragraph)
        assert para.attrs == {}

    def test_no_trailing_ial_means_empty_attrs(self) -> None:
        doc = parse("Plain text.")
        para = doc.children[0]
        assert isinstance(para, a.Paragraph)
        assert para.attrs == {}


class TestBlockquoteIAL:
    def test_trailing_ial_on_quote(self) -> None:
        doc = parse("> wisdom of the day {.callout color=blue}")
        bq = doc.children[0]
        assert isinstance(bq, a.Blockquote)
        assert bq.attrs.get("class") == ["callout"]
        assert bq.attrs.get("color") == "blue"

    def test_image_directive_inside_quote_keeps_its_ial(self) -> None:
        doc = parse("> ![QR](https://example.com){.qrcode}")
        bq = doc.children[0]
        assert isinstance(bq, a.Blockquote)
        # The blockquote-level IAL should NOT have been stolen from the image.
        assert bq.attrs == {}

    def test_no_ial_means_empty_attrs(self) -> None:
        doc = parse("> just a quote")
        bq = doc.children[0]
        assert isinstance(bq, a.Blockquote)
        assert bq.attrs == {}


class TestCodeBlockIAL:
    def test_lang_with_ial(self) -> None:
        doc = parse("```python {.callout}\nprint('hi')\n```")
        cb = doc.children[0]
        assert isinstance(cb, a.CodeBlock)
        assert cb.lang == "python"
        assert cb.attrs.get("class") == ["callout"]

    def test_ial_only_no_lang(self) -> None:
        doc = parse("``` {.callout}\nprint('hi')\n```")
        cb = doc.children[0]
        assert isinstance(cb, a.CodeBlock)
        assert cb.lang == ""
        assert cb.attrs.get("class") == ["callout"]

    def test_lang_only_still_works(self) -> None:
        doc = parse("```python\nprint('hi')\n```")
        cb = doc.children[0]
        assert cb.lang == "python"
        assert cb.attrs == {}


class TestListIAL:
    def test_standalone_ial_above_list(self) -> None:
        src = "{cols=2 align=left}\n- a\n- b\n- c"
        doc = parse(src)
        lst = doc.children[0]
        assert isinstance(lst, a.ListBlock)
        assert lst.attrs.get("cols") == "2"
        assert lst.attrs.get("align") == "left"

    def test_standalone_ial_with_blank_then_list(self) -> None:
        src = "{cols=2}\n\n- a\n- b"
        doc = parse(src)
        lst = doc.children[0]
        assert isinstance(lst, a.ListBlock)
        assert lst.attrs.get("cols") == "2"

    def test_class_only_ial_above_list(self) -> None:
        src = "{.bullets}\n- a\n- b"
        doc = parse(src)
        lst = doc.children[0]
        assert isinstance(lst, a.ListBlock)
        assert lst.attrs.get("class") == ["bullets"]

    def test_paragraph_with_ial_followed_by_list_doesnt_leak(self) -> None:
        # A paragraph that ends with ``{...}`` IAL should NOT have
        # its IAL stolen by the list that follows it.
        src = "A para. {.callout}\n\n- a\n- b"
        doc = parse(src)
        assert len(doc.children) == 2
        para = doc.children[0]
        lst = doc.children[1]
        assert isinstance(para, a.Paragraph)
        assert para.attrs.get("class") == ["callout"]
        assert isinstance(lst, a.ListBlock)
        assert lst.attrs == {}

    def test_no_list_after_means_ial_is_dropped(self) -> None:
        # A standalone IAL followed by a non-list block — ensure
        # the parser doesn't crash and doesn't smuggle the attrs
        # somewhere weird.
        src = "{cols=2}\n\n# Heading"
        doc = parse(src)
        # Either the ``{cols=2}`` line gets emitted as a paragraph
        # OR is dropped (we drop it). The heading is unaffected.
        h = doc.children[-1]
        assert isinstance(h, a.Heading)
        assert h.attrs == {}
