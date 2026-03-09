# exoclaw-plugins

Plugin packages for [exoclaw](https://github.com/exoclaw/exoclaw) — the protocol-only AI agent framework.

```
pip install exoclaw-nanobot   # full stack, drop-in replacement for nanobot
```

Or pick only what you need:

```
pip install exoclaw-provider-litellm
pip install exoclaw-conversation
pip install exoclaw-channel-cli
```

---

## Packages

| Package | PyPI | Description |
|---|---|---|
| `exoclaw-nanobot` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-nanobot)](https://pypi.org/project/exoclaw-nanobot/) | Full-stack bundle — config, wiring, `exoclaw-nanobot` CLI |
| `exoclaw-conversation` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-conversation)](https://pypi.org/project/exoclaw-conversation/) | File-backed sessions, JSONL history, LLM memory consolidation |
| `exoclaw-provider-litellm` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-provider-litellm)](https://pypi.org/project/exoclaw-provider-litellm/) | LiteLLM provider — Anthropic, OpenAI, OpenRouter, and more |
| `exoclaw-channel-cli` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-channel-cli)](https://pypi.org/project/exoclaw-channel-cli/) | Interactive terminal REPL channel |
| `exoclaw-channel-heartbeat` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-channel-heartbeat)](https://pypi.org/project/exoclaw-channel-heartbeat/) | Timed heartbeat service for background agent tasks |
| `exoclaw-tools-workspace` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-tools-workspace)](https://pypi.org/project/exoclaw-tools-workspace/) | File, shell, and web tools |
| `exoclaw-tools-cron` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-tools-cron)](https://pypi.org/project/exoclaw-tools-cron/) | Cron scheduler tool |
| `exoclaw-tools-message` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-tools-message)](https://pypi.org/project/exoclaw-tools-message/) | Send messages to channels from within a turn |
| `exoclaw-tools-spawn` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-tools-spawn)](https://pypi.org/project/exoclaw-tools-spawn/) | Spawn background subagents |
| `exoclaw-tools-mcp` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-tools-mcp)](https://pypi.org/project/exoclaw-tools-mcp/) | Connect MCP servers and register their tools |
| `exoclaw-subagent` | [![PyPI](https://img.shields.io/pypi/v/exoclaw-subagent)](https://pypi.org/project/exoclaw-subagent/) | SubagentManager — nested AgentLoop execution |

---

## Quick start

```python
import asyncio
from exoclaw_nanobot import create

async def main():
    bot = await create()  # reads ~/.nanobot/config.json, NANOBOT_* env vars
    await bot.run()

asyncio.run(main())
```

Or from the command line:

```
exoclaw-nanobot
```

---

## Development

This is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/). All packages live under `packages/`.

```bash
uv sync                          # install all packages in editable mode
uv run pytest                    # run all tests
uv run mypy packages/<pkg>/      # typecheck a package
```

To add a new package:

```bash
mkdir -p packages/exoclaw-my-package/exoclaw_my_package
# create pyproject.toml with hatchling build backend
# add to [tool.uv.sources] in root pyproject.toml if it depends on another workspace package
```

---

## License

MIT
