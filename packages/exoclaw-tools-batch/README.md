# exoclaw-tools-batch

Map and reduce primitives for agent-controlled ETL pipelines. Run tools against multiple inputs concurrently, then merge results — all without per-item LLM calls.

## Tools

### BatchTool (map)

Fan out a registered tool across multiple inputs concurrently. Results written to disk to keep agent context clean.

```python
from exoclaw_tools_batch import BatchTool, ReduceTool

batch = BatchTool(concurrency=10)
reduce = ReduceTool()
app = Exoclaw(tools=[batch, reduce, ...])
```

```json
{
  "tool": "batch",
  "items": [
    {"url": "https://example.com/1"},
    {"url": "https://example.com/2"}
  ],
  "concurrency": 5
}
```

Returns `{output_path: "/tmp/batch_web_fetch_abc.json", count: 2}`. Use `read_file` to inspect results.

### ReduceTool (reduce)

Merge multiple batch output files into one.

```json
{
  "dir": "/tmp/batch_llm_call_abc/",
  "key": "results",
  "dedup": "url"
}
```

Returns `{output_path: "/tmp/reduce_xyz.json", count: 150}`.

Options:
- `files` — explicit list of paths (alternative to `dir`)
- `key` — JSON key to extract from each file (default: `"results"`, empty string for root)
- `dedup` — deduplicate by field name
- `output` — explicit output path

## ETL pipeline example

```
1. batch(tool="web_fetch", items=[...93 feeds...])
   → /tmp/batch_web_fetch_abc.json

2. batch(tool="llm_call", items=[
     {prompt: "{{ file(path) }}\nExtract interesting URLs", model: "haiku", output: "/tmp/triage/0.json"},
     ...93 items...
   ])
   → 93 parallel cheap LLM calls

3. reduce(dir="/tmp/triage/", dedup="url")
   → /tmp/reduce_xyz.json (merged + deduped)

4. Agent reads merged results, queues to digest
```

2 batch calls + 1 reduce + 93 cheap model calls. No full agent loops per item.

## How it works

- `BatchTool` receives the `ToolRegistry` via duck-typed `set_registry()` at registration time
- Runs `registry.execute(tool, params)` for each item behind an asyncio semaphore
- Results written to temp files (configurable `output_dir`)
- `ReduceTool` merges JSON files by extracting at a key and concatenating

## Requirements

- `exoclaw >= 0.3.0` (with `set_registry` hook in AgentLoop)
