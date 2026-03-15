---
name: etl
description: Agent-controlled ETL using batch, reduce, and llm_call tools
---

# ETL Toolkit

Use `batch`, `reduce`, and `llm_call` for data pipelines. All output goes to disk — use `read_file` to inspect results.

## Tools

- **batch** — fan-out a tool across N inputs concurrently. Returns `{output_path, count}`.
- **reduce** — merge multiple files into one. Supports dedup, chunking, and tree-reduce (`until` + `then`).
- **llm_call** — single-shot LLM call with Jinja2 templates. `{{ var }}`, `{{ file('/path') }}`. Optional `schema` for structured output.

## Patterns

### Scrape + Extract
```
batch(tool="web_fetch", items=[{url: "..."}, ...])
batch(tool="llm_call", items=[
  {prompt: "Extract data:\n{{ file(input_path) }}", vars: {input_path: "..."}, model: "haiku", schema: {...}},
  ...
])
reduce(dir="/tmp/batch_llm_call.../")
```

### Tree-reduce (summarize large datasets)
```
batch(tool="read_file", items=[{path: "..."}, ...])
reduce(dir="...", until=1, chunk_size=20, then={tool: "llm_call", params: {prompt: "Summarize:\n{{ file(input_path) }}", model: "haiku"}})
```

### Filter + Deduplicate
```
reduce(files=[...], dedup="url")
reduce(dir="...", key="results", chunk_size=50)
```

## Tips

- **Don't read_file batch output** — it dumps raw content into your context and wastes tokens. Instead, pipe it to `llm_call` via `{{ file(path) }}` which sends it to the cheap model directly.
- If you must inspect output, peek with `read_file(path, offset=0, limit=10)`.
- Use `{{ file('/path') }}` in llm_call prompts to keep content off the agent context.
- Use `schema` in `llm_call` for structured JSON output.
- Use cheap models (haiku/nano) for extraction/filtering, expensive models for final synthesis.
- Don't use batch for a single item — call the tool directly.
