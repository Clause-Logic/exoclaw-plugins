# exoclaw-screen v0 grammar

Authoritative reference for the markdown subset + IAL + Pandoc
fenced divs that ``parser.py`` is supposed to handle. This file
is the spec; ``SKILL.md`` is the agent-facing summary; the parser
implements against this.

## Block elements

| Markdown | Meaning |
|---|---|
| `# Title` ... `### h3` | Headings 1–3 (h4–h6 are useless on a 7.5" panel) |
| Plain text + blank line | Paragraph |
| `- item` / `* item` | Unordered list (single-level, no nesting in v0) |
| `1. item` | Ordered list (single-level) |
| `> quoted` | Blockquote (line-prefix; multi-line OK) |
| ` ```...``` ` | Fenced code block |
| `---` | Horizontal rule |
| `![alt](src){.class}` (own line) | Image / directive |

## Inline elements

| Syntax | Meaning |
|---|---|
| `**text**` | Bold |
| `_text_` or `*text*` | Italic |
| `` `text` `` | Inline code |
| `[text](url)` | Link (renders as text with URL in muted style; e-ink can't be clicked) |
| Two trailing spaces + `\n` | Hard break inside a paragraph |

## IAL — attribute lists

Trailing `{...}` on a block element. Supports:

- `{.class}` — one or more semantic classes
- `{key=value}` — explicit attribute (unquoted values; v0 doesn't support `key="quoted val"`)
- Mixed: `{.title align=center color=red}`

**Attaches to:** headings, paragraphs, blockquotes, code blocks,
images. **On a list**, attaches to the list as a whole (not
individual items).

## Layout containers — Pandoc fenced divs

```markdown
# Home Dashboard {.title}

::: {.row gap=20}
::: {.col w=50%}
## Weather
**Now**: 72°F
**High**: 78°F · **Low**: 64°F
:::
::: {.col w=50%}
## Calendar
- 9am — Standup
- 11am — Design review
:::
:::

---

::: {.grid cols=3 gap=10}
::: {.cell}
**Temp** · 72°F
:::
::: {.cell}
**Humid** · 45%
:::
::: {.cell}
**CO₂** · 410ppm
:::
:::
```

| Class | Children laid out | Use for |
|---|---|---|
| `.row` | left-to-right | side-by-side widgets |
| `.col` | top-to-bottom | stacked widgets (also default at root) |
| `.grid cols=N` | N-cell equal grid, flowed | uniform widget grids |

### Sizing on containers and children

- `w=200` — absolute pixels
- `w=50%` — percentage of parent
- `h=...` — same for height
- `gap=10` — spacing between children (px)
- Omitted `w` / `h` → equal split among siblings

No `auto`, no flex-grow, no min/max in v0. Overflow is the
agent's problem (SKILL.md tells it the budget).

## Directives — via image syntax

V0 ships exactly two directives (renderer-side dispatch on
`.class`):

| Form | Effect |
|---|---|
| `![alt](path.md){.include}` | Recursively parse + inline-render `path.md`. Single level only in v0 (no nested includes); detect cycles. |
| `![alt](src){.qrcode size=200}` | Treat `src` as the data to encode, render QR code. Optional in v0 if no QR library is available; falls back to rendering the URL as text. |

Plain `![alt](src)` (no directive class) → render as italic alt
text in v0. Image fetch/render is post-v0.

## Out of scope for v0

Things the parser will TOLERATE (not crash on) but won't render
specially — they pass through as plain text or get dropped:

- Tables (`| col |`)
- Nested lists
- Reference-style links (`[txt][id]`)
- Strikethrough (`~~text~~`)
- Definition lists, footnotes, task lists
- HTML pass-through (`<div>...</div>`)
- Multi-paragraph blockquotes (single block only)
- Mixed `***bold-italic***`
- Emphasis-marker variant flips (markdown's `_em_` vs `*em*` —
  pick one canonical, accept either as input)

## Summary

- **Inline**: bold, italic, inline code, links, hard break.
- **Block**: headings 1–3, paragraphs, lists (single-level,
  ordered + unordered), blockquote, fenced code, hr,
  image-as-directive.
- **Containers**: `:::` fenced divs with `.row` / `.col` /
  `.grid` classes + `w` / `h` / `gap` attrs.
- **Directives**: `.include`, `.qrcode` (optional), plain image
  → italic alt fallback.
- **IAL on any block or container.**
