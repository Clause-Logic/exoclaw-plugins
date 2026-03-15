# exoclaw-tools-batch

Batch/map tool for exoclaw — run a tool against multiple inputs concurrently without LLM calls.

## Usage

```python
from exoclaw_tools_batch import BatchTool

batch = BatchTool(concurrency=10)
app = Exoclaw(tools=[batch, ...])
```

The LLM can then call:

```json
{
  "tool": "batch",
  "items": [
    {"url": "https://example.com/1"},
    {"url": "https://example.com/2"},
    {"url": "https://example.com/3"}
  ],
  "concurrency": 5
}
```

Each item is passed to the named tool's `execute()` directly (no LLM involved). Results are returned as a JSON array in the same order.

## How it works

1. `BatchTool` receives the `ToolRegistry` via duck-typed `set_registry()` at registration time
2. On execute, it runs `registry.execute(tool, params)` for each item behind an asyncio semaphore
3. Results are gathered, ordered, and returned as one coalesced JSON response

## Requirements

- `exoclaw >= 0.1.0` (with `set_registry` hook in AgentLoop)
