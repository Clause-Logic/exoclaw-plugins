# exoclaw-tools-web

Web tools for [exoclaw](https://github.com/Clause-Logic/exoclaw)
agents — pure-Python, cross-runtime (CPython + MicroPython).

## Tools

- **`web_fetch`** — GET a URL and return its content as Markdown.
  Uses `exoclaw.http.HTTPClient` (cross-runtime) for the fetch
  and a stdlib-only HTML→Markdown state-machine converter for
  the conversion. Bounded output (~32 KB on chip, ~128 KB on
  CPython).
- **`web_search`** — answer a query via an OpenAI-compatible LLM
  with a web-search plugin enabled (e.g. OpenRouter's
  `{"plugins": [{"id": "web"}]}` extra-body). Drops `litellm`
  in favour of routing through the same `OpenAIStreamingProvider`
  the agent already speaks to.

## Converter

`exoclaw_tools_web.convert(html_str) -> str` is the
HTML→Markdown converter. State machine, no `re`, no
`html.parser`. Passes 113/131 of turndown's regression-test
fixtures on default options (the 18 known fails all use
turndown options we don't mirror — referenced links, atx
headings, custom code-block fences, etc.).

## License

MIT.

The vendored test fixtures under `tests/fixtures/` are derived
from the [turndown](https://github.com/mixmark-io/turndown) test
suite (MIT 2017 Dom Christie) — see `tests/fixtures/NOTICE`.
