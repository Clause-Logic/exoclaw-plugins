---
name: web
description: Fetch URLs as Markdown and search the web for grounded answers
---

# Web Tools

Two tools for reaching the web:

## `web_fetch`

Fetch a URL and get its content back as **Markdown** (cleaned up
from raw HTML — no script tags, no nav menus, no ad markup).

```json
{"url": "https://example.com/article"}
```

Use when:

- The user gives you a link and asks "what does this say".
- You have a URL from `web_search` results and want the full
  article body.
- You need to read a doc page, a GitHub README, etc.

The output is capped (~32 KB on chip, ~128 KB on host). Long
pages return the head with a `(truncated …)` notice. The chip
streams the page through the converter so memory stays bounded
regardless of source page size.

## `web_search`

Search the web and get a grounded answer with citations.

```json
{"query": "openai pricing per million tokens"}
```

Use when:

- You don't already know the answer and the question is
  factual / current.
- The user asks about recent events, prices, weather, latest
  docs, or anything that might have changed since training.

The answer already has source citations — you don't need to
follow them with `web_fetch` unless the user asks to read the
original.

## When to use which

- **Search first** if you don't know where the answer lives.
- **Fetch** when you have a specific URL and want its content
  verbatim.

Don't fetch a search engine page — use `web_search` instead.
