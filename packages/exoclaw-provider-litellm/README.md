# exoclaw-provider-litellm

LiteLLM-backed `LLMProvider` for exoclaw — supports OpenAI, Anthropic, OpenRouter, Gemini, and any other LiteLLM-compatible endpoint through a single interface.

## Install

```
pip install exoclaw-provider-litellm
```

## Usage

```python
from exoclaw_provider_litellm.provider import LiteLLMProvider

provider = LiteLLMProvider(
    api_key="sk-...",
    default_model="anthropic/claude-opus-4-5",
)

response = await provider.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    tools=[],
)
print(response.content)
```

For a custom gateway or OpenRouter, pass `api_base` alongside `api_key`. The provider normalises tool call IDs, sanitises empty content, and optionally logs full request/response details via `LLM_LOGGING=true`.
