# exoclaw-channel-telegram

Telegram channel for [exoclaw](https://github.com/Clause-Logic/exoclaw) — long-poll inbound, MarkdownV2 outbound, streaming via message edits, optional inline keyboard buttons, video.

## Install

```bash
pip install exoclaw-channel-telegram
```

## Setup

1. Open Telegram, message `@BotFather`, send `/newbot`. Pick a name and username. Copy the token (`123456:ABC-DEF...`).
2. Get your numeric Telegram user ID — message `@userinfobot` and copy the ID it sends back. That's what goes in `allow_from`.

## Use

```python
import asyncio
from exoclaw_nanobot import create
from exoclaw_channel_telegram import TelegramChannel, TelegramConfig

async def main() -> None:
    tg = TelegramChannel(TelegramConfig(
        enabled=True,
        token="123456:ABC-DEF...",
        allow_from=["12345678"],   # your Telegram user ID(s)
    ))
    bot = await create(extra_channels=[tg])
    await bot.run()

asyncio.run(main())
```

## Config

| Field | Default | Description |
|---|---|---|
| `token` | — | Bot token from BotFather (required) |
| `allow_from` | `[]` | Telegram user IDs allowed to message the bot (empty = deny all) |
| `group_policy` | `"mention"` | `"mention"` (respond only when @mentioned in groups) or `"open"` |
| `proxy` | `None` | SOCKS5 proxy URL, e.g. `socks5://host:port` |
| `reply_to_message` | `False` | Quote the original message when replying |
| `react_emoji` | `"👀"` | Emoji reaction added when message received |
| `streaming` | `True` | Stream the agent's reply via message edits as it generates |
| `inline_keyboards` | `False` | Render `OutboundMessage.buttons` as Telegram inline keyboards. Requires the `callback_query` update; enable in BotFather if you've restricted updates. |

## Audit boundary

Vendored from HKUDS/nanobot via codemod. What's committed: upstream snapshot in `vendor/`, channel-specific patch in `patches/0001-test-skip-help-restart-command.patch` (skips an upstream test that asserts nanobot-specific command names exoclaw doesn't have), plus the small bootstrap files. The shipped `channel.py` and `tests/test_channel.py` are gitignored — generated at build time by [`exoclaw-channel-codemod`](../exoclaw-channel-codemod/), included in the wheel via the hatch hook. See [`exoclaw-nanobot-compat/README.md`](../exoclaw-nanobot-compat/README.md) for the full pattern.

## Maintenance

```bash
echo "<new-hkuds-sha>" > vendor/SHA
UPSTREAM=~/hkuds-nanobot bash ../exoclaw-channel-codemod/sync.sh telegram --apply
uv run pytest packages/exoclaw-channel-telegram/tests/
```
