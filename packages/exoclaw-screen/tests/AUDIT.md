# exoclaw-screen v0 grammar — parser/layout/renderer audit

Snapshot taken while bringing parser to full GRAMMAR.md compliance.
Every row that was CUT or PARTIAL is now OK.

| Spec                                      | Before  | After |
| ----------------------------------------- | ------- | ----- |
| Heading IAL                               | OK      | OK    |
| Paragraph IAL                             | CUT     | OK    |
| Blockquote IAL                            | CUT     | OK    |
| Code block IAL                            | CUT     | OK    |
| Image IAL                                 | OK      | OK    |
| List-as-whole IAL                         | CUT     | OK    |
| Container IAL                             | OK      | OK    |
| Container `w` / `h` / `gap` honored       | PARTIAL | OK    |
| `.include` directive renderer dispatch    | PARTIAL | OK    |
| `.qrcode` directive renderer dispatch     | PARTIAL | OK    |
| Plain image italic-alt fallback           | PARTIAL | OK    |

## Notes

### Paragraph + Blockquote IAL — image collision detection

Both branches strip a trailing `{...}` from their last non-blank
line UNLESS that line ends with an image-directive close
(`){.qrcode}` etc.) — detected via `_trailing_ial_belongs_to_image`
in `parser.py`. The rule: walk back from the trailing `}` to the
matching `{`; if the char immediately before the `{` is `)` AND
the line earlier contains `![`, the IAL belongs to the image.

### Code block IAL

The fence's info string is parsed as `lang {ial?}`. Lang is the
first whitespace-delimited word; trailing `{...}` becomes the
code block's `attrs`.

### List-as-whole IAL

A standalone `{...}` line (just an IAL block, nothing else) that
is followed (after zero or more blank lines) by a list marker
attaches its attrs to the upcoming `ListBlock`. If the next
non-blank line is NOT a list, the standalone IAL is dropped.
This is implemented as a `pending_list_attrs` accumulator in
`_parse_blocks` that gets cleared by every other block branch.

A paragraph that ends with `{...}` IAL followed by a list does
NOT leak — the paragraph keeps its IAL, the list gets empty
attrs.

### Container `w` / `h` / `gap`

`_layout_container` honors `w` and `h` against the slot it's
given. A new `slot_sized` flag tells nested containers their
slot was already resolved by the parent, preventing
double-application of percentages.

`.row` distributes width left-to-right, `.col` distributes height
top-to-bottom (when `h` is given), `.grid cols=N` flows children
into N equal cells. Children with explicit `w` (or `h`) get their
share first; the remainder splits among siblings without one.

### `.include` directive

`PillowRenderer.render_to_png` accepts `base_path` and threads
it through. Includes resolve `src` against `base_path`; cycle
detection via a `visited: set[str]` snapshot. Single-level
only — nested `.include` inside an included file renders an
italic stub `[nested-include: ...]`.

### `.qrcode` directive

Uses the optional `qrcode` Python package, declared in
`[host-preview]` extras alongside Pillow. Gated behind a
`try: import qrcode` — if missing, falls back to italic URL
text. Same fallback covers chip MicroPython where no QR
encoder ships yet.

### Plain image italic alt

`_draw_image_directive` paints `node.alt or node.src` using an
italic font when no recognised `.class` is on the image. The
italic font is best-effort loaded (`DejaVuSansMono-Oblique.ttf`
etc.); falls back to `ImageFont.load_default()` if no
italic font is on the host.
