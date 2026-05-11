# exoclaw-turn

Run a single exoclaw agent turn as a library call — no bus, no channels, no persistence.

```
pip install exoclaw-turn
```

## Why

Most of exoclaw is built around the bus-driven multi-turn model: messages come in on a channel, the agent loop processes them, replies go back out. That shape is right for bots and assistants, but overkill when you just want to **call an agent like a function** — get one bounded run, with tool dispatch, and a return value.

`run_turn` is that front door. It spins up an ephemeral `AgentLoop` with a throwaway `MessageBus` and an in-memory conversation, drives a single turn to completion, and returns the result. Everything the agent loop already does — compaction on context overflow, loop detection, plugin context collection, subagent chain tracking, tool dispatch — carries over unchanged.

## Usage

```python
from exoclaw_turn import run_turn
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_openrouter_search.tool import OpenRouterSearchTool

provider = LiteLLMProvider(default_model="claude-sonnet-4-6")

result = await run_turn(
    provider=provider,
    system="You are a research assistant. Be terse.",
    message="What's the population of Reykjavik?",
    tools=[OpenRouterSearchTool()],
)

print(result.text)           # final assistant reply
print(result.tool_calls)     # tools the model invoked, in order
print(result.messages)       # all new messages produced this turn
```

Anything implementing the `exoclaw.agent.tools.protocol.Tool` protocol works as a tool — every tool plugin in the exoclaw-plugins catalog drops in unchanged.

## What it inherits from `AgentLoop`

`run_turn` owns no agent behaviour of its own. It assembles the loop and reshapes the result. That means whatever lives in `AgentLoop` today is available through this front door:

- **Context-window compaction.** Long tool chains that overflow the model's context window trigger `Conversation.recover_from_overflow` if implemented — for the ephemeral conversation it isn't, so the original `ContextWindowExceededError` surfaces. Pass a longer-lived conversation if you need recovery.
- **Loop detection.** Pass `iteration_policy=...` (e.g. from `exoclaw-loop-detection`) to terminate based on tool-call patterns instead of a hard count.
- **Tool dispatch.** Tools execute through the same `ToolRegistry` the bus-driven path uses.
- **Subagent chain tracking.** If you call `run_turn` from inside another turn, the trace contextvars (`turn.id`, `turn.parent_id`, `turn.chain`) extend the parent's ancestry — same logging hierarchy as `exoclaw-subagent`.

## Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `provider` | `LLMProvider` | required | Any provider — LiteLLM, OpenAI, etc. |
| `message` | `str` | required | The user message that seeds the turn. |
| `system` | `str \| None` | `None` | System prompt. Plugin context from tools is appended. |
| `tools` | `list[Tool] \| None` | `None` | Anything implementing `Tool`. |
| `model` | `str \| None` | provider default | Override the provider's default for this turn. |
| `max_iterations` | `int` | `40` | Hard cap on tool-call iterations. |
| `temperature` | `float` | `0.1` | Forwarded to provider. |
| `max_tokens` | `int` | `4096` | Forwarded to provider. |
| `reasoning_effort` | `str \| None` | `None` | Forwarded to provider for reasoning models. |
| `iteration_policy` | `IterationPolicy \| None` | `None` | Replace the hard count with pattern-based termination. |
| `on_progress` | `Callable \| None` | `None` | Async progress callback. |

## What it doesn't do

- **No persistence.** State is dropped at end of turn. For multi-turn conversations, use the bus-driven `AgentLoop` directly, or build on top of `exoclaw-conversation`.
- **No streaming result.** The function awaits the full turn before returning. Use `on_progress` for incremental feedback if you need it.
- **No multimodal attachments.** The ephemeral conversation doesn't encode media into the user message content. For image/file inputs, use a bus-driven `AgentLoop` with a real `Conversation` implementation.
- **No channel.** This is library-shaped. For an interactive REPL, use `exoclaw-channel-cli`.
