# exoclaw-nanobot

Full-stack exoclaw bundle — wires provider, conversation, all workspace/cron/message/spawn/MCP tools, subagent manager, CLI channel, and heartbeat into a single ready-to-run agent.

## Install

```
pip install exoclaw-nanobot
```

## CLI

```
exoclaw-nanobot
```

Reads config from `~/.nanobot/config.json` (or `NANOBOT_*` env vars). Drops into an interactive REPL.

## Programmatic usage

```python
import asyncio
from exoclaw_nanobot.app import create, ExoclawNanobot

async def main() -> None:
    bot: ExoclawNanobot = await create()
    await bot.run()

asyncio.run(main())
```

`create()` accepts an optional pre-built `Config` or `config_path`. It returns an `ExoclawNanobot` whose `run()` method starts the cron service, heartbeat, agent loop, and CLI REPL, and tears everything down cleanly on exit.

## Adding channels (Slack, Telegram, Discord, Email, Matrix, WhatsApp)

Each channel lives in its own package — install only what you need:

```
pip install 'exoclaw-nanobot[slack]'
pip install 'exoclaw-nanobot[slack,telegram,discord]'
pip install 'exoclaw-nanobot[all-channels]'
```

Then enable each channel in your config:

```json
{
  "channels": {
    "slack": {
      "enabled": true,
      "botToken": "xoxb-...",
      "appToken": "xapp-...",
      "allowFrom": ["U01ABC..."]
    },
    "telegram": {
      "enabled": true,
      "token": "123456:abcdef...",
      "allowFrom": ["123456789"]
    }
  }
}
```

`create()` reads `config.channels.<name>` for each section, instantiates the matching channel class, and starts it alongside the CLI. If a channel is `enabled: true` but its package isn't installed, startup fails with a clear pointer to the right `pip install` command.

Per-channel config fields live in `exoclaw_nanobot.config.schema` — `SlackConfig`, `TelegramConfig`, `DiscordConfig`, `EmailConfig`, `MatrixConfig`, `WhatsAppConfig`.

## Where these packages live

This bundle and every package it pulls in dual-publish to **PyPI** (the default `pip install` source) and to a self-hosted PEP 503 index at **[clause-logic.github.io/registry](https://clause-logic.github.io/registry/)**. The registry is authoritative — it always has the latest release. PyPI is a best-effort mirror that occasionally lags due to its [new-project creation rate limit](https://github.com/pypi/support/issues/10572).

To prefer the registry, add to your project's `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "clause-logic"
url = "https://clause-logic.github.io/registry/pypi/simple/"

[tool.uv.sources]
exoclaw-nanobot = { index = "clause-logic" }
exoclaw-channel-slack = { index = "clause-logic" }  # repeat per channel you use
```

Pip users: `pip install --extra-index-url https://clause-logic.github.io/registry/pypi/simple/ 'exoclaw-nanobot[slack]'`.
