# exoclaw-tools-llm-call

Single-shot LLM call tool with Jinja2 templating for exoclaw.

No agent loop, no tools — just prompt in, text out. Use with `batch` for parallel cheap-model processing.

## Usage

```python
from exoclaw_tools_llm_call import LLMCallTool

llm_call = LLMCallTool(
    provider=provider,
    allowed_models=["haiku", "sonnet"],
    default_model="haiku",
)
app = Exoclaw(tools=[llm_call, ...])
```

The agent can then call:

```json
{
  "prompt": "Feed: {{ feed_name }}\n\n{{ file(data_path) }}\n\nExtract interesting URLs as JSON.",
  "vars": {"feed_name": "Simon Willison", "data_path": "/tmp/batch_abc/0.json"},
  "model": "haiku",
  "output": "/tmp/results/0.json"
}
```

## Template features

- `{{ var }}` — variable substitution from `vars` dict
- `{{ file('/path/to/file') }}` — inline file contents
- All Jinja2 features: filters, conditionals, loops

## Combined with batch

```
batch(tool="llm_call", items=[
  {prompt: "...", vars: {data_path: "/tmp/feeds/0.json"}, model: "haiku", output: "/tmp/out/0.json"},
  {prompt: "...", vars: {data_path: "/tmp/feeds/1.json"}, model: "haiku", output: "/tmp/out/1.json"},
  ...
])
```

93 cheap LLM calls in parallel. Main agent only sees the filtered output.
