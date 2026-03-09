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
