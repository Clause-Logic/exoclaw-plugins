# exoclaw-plugins

Drop-in pieces for [exoclaw](https://github.com/Clause-Logic/exoclaw) — pick a provider, a memory store, the channels you want, and the tools you need.

---

## Working CLI agent in 30 seconds

```
pip install exoclaw-nanobot
exoclaw-nanobot
```

That installs a bundle (provider + conversation + tools + the CLI and heartbeat channels) and drops you into an interactive REPL. Edit `~/.nanobot/config.json` to point it at your LLM.

```python
import asyncio
from exoclaw_nanobot import create

async def main():
    bot = await create()  # reads ~/.nanobot/config.json + NANOBOT_* env vars
    await bot.run()

asyncio.run(main())
```

Need Slack, Telegram, Discord, etc.? Install the channel package alongside it and pass it via `extra_channels=[...]` — see [Slack example](#slack-bot-four-lines-of-config) below.

---

## Or pick your own stack

A working agent needs at least:

1. An **LLM provider** — talks to a model.
2. A **conversation** — remembers what's been said.
3. A **channel** — how messages come in and out.

Everything else is optional. Pick from the catalog below.

### LLM providers

| Package | What you get |
|---|---|
| [`exoclaw-provider-litellm`](packages/exoclaw-provider-litellm) | Anthropic, OpenAI, OpenRouter, Bedrock, Ollama, … via [LiteLLM] |
| [`exoclaw-provider-openai`](packages/exoclaw-provider-openai) | Direct OpenAI SDK |

[LiteLLM]: https://github.com/BerriAI/litellm

### Conversation memory

| Package | What you get |
|---|---|
| [`exoclaw-conversation`](packages/exoclaw-conversation) | File-backed sessions, JSONL history, LLM memory consolidation |

### Channels

| Package | What you get |
|---|---|
| [`exoclaw-channel-cli`](packages/exoclaw-channel-cli) | Interactive terminal REPL — great for local testing |
| [`exoclaw-channel-slack`](packages/exoclaw-channel-slack) | Slack — Socket Mode, Block Kit, file uploads |
| [`exoclaw-channel-telegram`](packages/exoclaw-channel-telegram) | Telegram — long-poll, inline keyboards, video |
| [`exoclaw-channel-discord`](packages/exoclaw-channel-discord) | Discord — threads, streaming via message edits |
| [`exoclaw-channel-email`](packages/exoclaw-channel-email) | Email — IMAP poll + SMTP send |
| [`exoclaw-channel-matrix`](packages/exoclaw-channel-matrix) | Matrix — E2E encryption, threads |
| [`exoclaw-channel-whatsapp`](packages/exoclaw-channel-whatsapp) | WhatsApp — manages a Node bridge sidecar |
| [`exoclaw-channel-heartbeat`](packages/exoclaw-channel-heartbeat) | Timed pings to trigger background agent tasks |
| [`exoclaw-channel-pipe`](packages/exoclaw-channel-pipe) | Stdin/stdout — wire the agent into Unix pipelines |

### Tools

| Package | What the agent can do |
|---|---|
| [`exoclaw-tools-workspace`](packages/exoclaw-tools-workspace) | Read, write, and run shell commands inside a workspace dir |
| [`exoclaw-tools-web`](packages/exoclaw-tools-web) | Web search and page fetching |
| [`exoclaw-tools-mcp`](packages/exoclaw-tools-mcp) | Connect MCP servers and use their tools |
| [`exoclaw-tools-cron`](packages/exoclaw-tools-cron) | Schedule reminders and recurring tasks |
| [`exoclaw-tools-message`](packages/exoclaw-tools-message) | Send messages to other channels mid-turn |
| [`exoclaw-tools-spawn`](packages/exoclaw-tools-spawn) | Spawn background subagents |
| [`exoclaw-tools-batch`](packages/exoclaw-tools-batch) | Run several tool calls in parallel |
| [`exoclaw-tools-llm-call`](packages/exoclaw-tools-llm-call) | Make a one-off LLM call without touching session state |
| [`exoclaw-tools-voice`](packages/exoclaw-tools-voice) | Audio transcription |

### Behavior plugins

| Package | What it changes |
|---|---|
| [`exoclaw-loop-detection`](packages/exoclaw-loop-detection) | Smarter stop conditions (repeat detection, ping-pong, circuit breaker) |
| [`exoclaw-turn-budget`](packages/exoclaw-turn-budget) | Per-turn token + tool budget enforcement |
| [`exoclaw-executor-dbos`](packages/exoclaw-executor-dbos) | Durable execution via [DBOS](https://www.dbos.dev) — checkpointed turns |
| [`exoclaw-subagent`](packages/exoclaw-subagent) | Nested AgentLoop execution for delegated work |

### Integrations

| Package | What it gives you |
|---|---|
| [`exoclaw-github`](packages/exoclaw-github) | GitHub Actions bot — replies to issues and PRs |
| [`exoclaw-screen`](packages/exoclaw-screen) | Screen capture / OCR for visual context |
| [`exoclaw-firmware`](packages/exoclaw-firmware) | MicroPython firmware images for ESP32-S3 |

---

## Where these packages live

We publish to two indexes:

1. **PyPI** — the default `pip install` source. Hosts core `exoclaw`, the bundle, providers, conversation, tools, and behavior plugins.
2. **[clause-logic.github.io/registry](https://clause-logic.github.io/registry/)** — our self-hosted PEP 503 index. Hosts everything PyPI does *plus* the six channel packages (slack/telegram/discord/email/matrix/whatsapp) — those hit PyPI's [new-project creation rate limit](https://github.com/pypi/support/issues/10572) on initial publish and only live here for now.

The release workflow attempts both indexes on every tag; once the channel rate-limit clears, they'll appear on PyPI too.

To prefer the registry for everything we publish, add this to your project's `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "clause-logic"
url = "https://clause-logic.github.io/registry/pypi/simple/"
```

uv checks clause-logic first for every package and falls through to PyPI for anything not there. Transitive deps like `pydantic` and `structlog` only live on PyPI — they're not republished here.

Pip users: add `--extra-index-url https://clause-logic.github.io/registry/pypi/simple/` to your install command (or set it in `pip.conf`). Note that with pip this adds the registry as an additional source rather than promoting it ahead of PyPI; that's fine for our case since the channel packages aren't on PyPI yet.

---

## Examples

### Slack bot, four lines of config

```python
import asyncio
from exoclaw import Exoclaw
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_channel_slack.channel import SlackChannel

async def main():
    provider = LiteLLMProvider(default_model="claude-sonnet-4-6")
    bot = Exoclaw(
        provider=provider,
        conversation=DefaultConversation.create(workspace="~/.slackbot", provider=provider),
        channels=[SlackChannel(config={"bot_token": "...", "app_token": "...", "allow_from": ["*"]})],
    )
    await bot.run()

asyncio.run(main())
```

### CLI agent with web search

```python
from exoclaw import Exoclaw
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_channel_cli.channel import CLIChannel
from exoclaw_tools_web.search import WebSearchTool

provider = LiteLLMProvider(default_model="claude-sonnet-4-6")
bot = Exoclaw(
    provider=provider,
    conversation=DefaultConversation.create(workspace="~/.cli-agent", provider=provider),
    channels=[CLIChannel()],
    tools=[WebSearchTool(api_key="...")],
)
```

---

## Development

[uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) — all packages live under `packages/`.

```bash
uv sync                # install all packages in editable mode
uv run pytest          # run all tests
mise run test          # ditto, plus formatting and linting
```

Channel packages under `packages/exoclaw-channel-{slack,telegram,discord,email,matrix,whatsapp}` are codemod-vendored from [HKUDS/nanobot](https://github.com/HKUDS/nanobot) (MIT). Source lives in each package's `vendor/` dir; the channel module is generated at build/test time. To pull in upstream fixes:

```bash
UPSTREAM=~/hkuds-nanobot bash packages/exoclaw-channel-codemod/sync.sh <name> --apply
```

See [`packages/exoclaw-channel-codemod`](packages/exoclaw-channel-codemod) for the codemod itself, and [`packages/exoclaw-nanobot-compat`](packages/exoclaw-nanobot-compat) for the runtime compat shim.

---

## License

MIT
