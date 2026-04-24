# exoclaw-provider-openai

Direct-httpx OpenAI-compatible provider for exoclaw.

Uses httpx directly (no OpenAI SDK, no LiteLLM) so the request body can be
streamed as the JSON is built rather than materialized as one contiguous
string before the POST. On long conversations this drops peak per-turn RAM
from ≈3× prompt-size (list + JSON dump + httpx buffer) to ≈1× streaming
buffer — see `exoclaw/docs/memory-model.md` Step B.

## Shape

```python
from exoclaw_provider_openai import Deployment, OpenAIStreamingProvider

provider = OpenAIStreamingProvider(
    default_model="zai/glm-4.7",
    deployments={
        "zai/glm-4.7":           Deployment(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY),
        "minimax/minimax-m2.7":  Deployment(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY),
        "openai/gpt-5.4":        Deployment(base_url="https://api.openai.com/v1",    api_key=OPENAI_KEY),
        "zai-direct/glm-5.1":    Deployment(base_url="https://api.z.ai/api/coding/paas/v4", api_key=ZAI_KEY),
    },
    fallbacks={
        "zai/glm-4.7":   ["minimax/minimax-m2.7"],
        "zai/glm-5.1":   ["minimax/minimax-m2.7"],
    },
)
```

Each model name maps to exactly one deployment (base URL + API key + optional
extra headers). `fallbacks` is a per-model list — when the primary raises a
retryable error (429, 5xx, timeout) the provider walks the fallback list
until one succeeds or all are exhausted.

## Routing policy

- Each model → exactly one deployment. No load balancing across deployments
  for a single model.
- Fallbacks are strict per-model lists. No automatic cross-provider fallback.
- SSE response streaming is always on (the streaming-request-body path
  requires it as the codec; non-streaming responses aren't supported).
- TTFT timeout: if the first response byte doesn't arrive inside
  ``stream_ttft_timeout`` seconds, the request is abandoned and the fallback
  list is tried. Default 15 s.

## What's intentionally skipped vs. LiteLLM

- Multi-provider cost tracking / usage normalization (OpenRouter returns the
  OpenAI shape directly; openclaw doesn't use cost tracking).
- Automatic Anthropic cache_control tagging. If your deployment needs it,
  stamp it on the messages before handing them to `chat()` — this provider
  is OpenAI-schema only.
- Deep parameter remapping across providers. Everything here is the OpenAI
  chat-completions schema; it's the caller's responsibility to send messages
  in that shape.
