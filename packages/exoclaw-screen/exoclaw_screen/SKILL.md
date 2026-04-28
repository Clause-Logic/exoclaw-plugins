---
name: screen
description: Control the device's screen by editing screen.md and calling repaint_screen
---

# Screen — File-backed Display

The device has a screen. You control what's on it by editing the
file `screen.md` (using the standard file tools — `read_file`,
`edit_file`, `write_file`) and then calling `repaint_screen()`.

The screen holds whatever was last rendered until the next
`repaint_screen()` call. Persistence is automatic — reboot the
device and the screen comes back to whatever was on disk.

## How to update the screen

**Change one value:**

1. `edit_file("screen.md", old="72°F", new="74°F")`
2. `repaint_screen()`

**Replace the whole layout:**

1. `write_file("screen.md", new_markdown_content)`
2. `repaint_screen()`

**See what's on screen now:**

`read_file("screen.md")`

## Layout grammar — markdown + IAL + fenced divs

Standard Markdown with two extensions:

- **IAL attribute lists** — a trailing `{.class attr=value}` block
  attaches metadata to a heading, paragraph, list, blockquote, code
  block, image, or container.
- **Fenced divs** (`:::`) — Pandoc syntax for layout containers.

### Block elements

```markdown
# Heading 1
## Heading 2
### Heading 3

A paragraph of text. **Bold**, _italic_, and `inline code`.
[Links](https://example.com) render as the link text.

- Bullet list
- Single level
- No nesting in v0

1. Ordered list
2. Same
3. No nesting

> Blockquote, single paragraph in v0.

` ` ` python
code block
` ` `

---

![alt](https://example.com/img.png)
```

### Inline elements

- `**bold**`
- `_italic_` or `*italic*`
- `` `inline code` ``
- `[link text](url)`
- Two trailing spaces + newline → hard break inside a paragraph

### IAL — attribute lists

Append `{.class attr=value}` to a block to tune its rendering:

```markdown
# Title {.title align=center}

**Now**: 72°F {color=red weight=bold}

_Updated 10:23am_ {.footer align=right size=small}
```

Class shorthand: `.foo`. Bare attributes: `key=value`. Combined:
`{.title align=center color=red}`. No quoted values in v0.

IAL attaches to:

- Headings, paragraphs, blockquotes, code blocks, images —
  trailing `{...}` on the block.
- Code blocks — `\`\`\`python {.callout}` (lang token first, then
  optional IAL).
- **Lists** — IAL attaches to the list as a whole, not items. Put
  `{.cols=2 align=left}` on its own line *immediately above* the
  first list marker:

  ```markdown
  {.bullets}
  - a
  - b
  - c
  ```

  Blank line between IAL and list is allowed; any other block
  between them drops the IAL.

### Layout containers — fenced divs

`:::` with an IAL opens a container. `:::` alone closes it.
Three primitives:

- `.row` — children laid out left-to-right.
- `.col` — top-to-bottom (also the default at root).
- `.grid cols=N` — N-cell equal grid.

Sizing on containers and children:

- `w=200` — absolute pixels.
- `w=50%` — percent of parent.
- `h=...` — same for height.
- `gap=10` — spacing between children (px).

Example dashboard:

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

### Directives — image syntax

Three image modes, dispatched on the IAL class:

```markdown
![cat](cat.jpg){h=300}                            # plain raster
![QR](https://example.com/dashboard){.qrcode size=200}
![weather](weather.md){.include}
```

- **Plain image, no class** — load a real raster file (JPEG /
  PNG / GIF / WebP / BMP) from `src` and paste into the layout
  slot, aspect-preserving fit. `src` is resolved as
  workspace-relative (e.g. `cat.jpg` → `{workspace}/cat.jpg`).
  HTTP URLs are not fetched at render time — use `web_fetch`
  with `save_to=<path>` first to land the bytes in the
  workspace, then reference the local path here.
  Use `{w=N h=N}` to size the slot; without sizing the image
  collapses to one row of body-text height. The image is
  centred inside the slot if it's smaller than the box.
- `.qrcode` — encode `src` as a QR code (gated on the
  `qrcode` package; falls back to italic URL text on chip).
- `.include` — recursively inline another markdown file.
  Single level only in v0; cycle detection on path.

## Out of scope for v0 (parsed but not rendered specially)

Tables (`| col |`), nested lists, reference-style links
(`[txt][id]`), strikethrough (`~~text~~`), footnotes, task lists,
HTML pass-through (`<div>`), multi-paragraph blockquotes, mixed
`***bold-italic***`. Don't rely on them rendering — your layout
will look better if you stick to the supported subset above.
