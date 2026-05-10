# exoclaw-channel-discord

Discord channel for [exoclaw](https://github.com/Clause-Logic/exoclaw) — `discord.py`-based, threads, streaming via message edits, file attachments, reaction lifecycle (👀 → 🔧 → cleared).

## Install

```bash
pip install exoclaw-channel-discord
```

## Setup

1. <https://discord.com/developers/applications> → **New Application** → **Bot** → **Add Bot**. Copy the bot token.
2. **Bot** → **Privileged Gateway Intents** → enable **MESSAGE CONTENT INTENT**. (Optionally **SERVER MEMBERS INTENT** if you want server-member-based allowlists.)
3. **OAuth2 → URL Generator**: scope `bot`, permissions `Send Messages`, `Read Message History`, `Add Reactions`, `Attach Files`. Open the generated URL to invite the bot to your server.
4. In Discord client: enable **Developer Mode** (Settings → Advanced), right-click your avatar → **Copy User ID**. That's what goes in `allow_from`.

## Use

```python
import asyncio
from exoclaw_nanobot import create
from exoclaw_channel_discord import DiscordChannel, DiscordConfig

async def main() -> None:
    discord = DiscordChannel(DiscordConfig(
        enabled=True,
        token="YOUR_BOT_TOKEN",
        allow_from=["123456789012345678"],   # your Discord user ID(s)
    ))
    bot = await create(extra_channels=[discord])
    await bot.run()

asyncio.run(main())
```

## Config

| Field | Default | Description |
|---|---|---|
| `token` | — | Bot token from the Discord developer portal (required) |
| `allow_from` | `[]` | Discord user IDs allowed to message the bot (empty = deny all) |
| `allow_channels` | `[]` | Channel IDs the bot may respond in (empty = all) |
| `intents` | `37377` | discord.py intents bitmask. Default = guilds + messages + message content + DMs + reactions. |
| `group_policy` | `"mention"` | `"mention"` (respond only when @mentioned in guild channels) or `"open"` |
| `read_receipt_emoji` | `"👀"` | Reaction added on message receipt |
| `working_emoji` | `"🔧"` | Reaction added after `working_emoji_delay`s if reply hasn't started yet |
| `streaming` | `True` | Stream the agent's reply via message edits as it generates |
| `proxy` / `proxy_username` / `proxy_password` | `None` | Optional HTTP proxy for the Discord gateway connection |

## Audit boundary

Vendored from HKUDS/nanobot via codemod. What's committed: upstream snapshot in `vendor/`, optional patches in `patches/`, plus the small bootstrap files. The shipped `channel.py` and `tests/test_channel.py` are gitignored — generated at build time by [`exoclaw-channel-codemod`](../exoclaw-channel-codemod/), included in the wheel via the hatch hook. See [`exoclaw-nanobot-compat/README.md`](../exoclaw-nanobot-compat/README.md) for the full pattern.

## Maintenance

```bash
echo "<new-hkuds-sha>" > vendor/SHA
UPSTREAM=~/hkuds-nanobot bash ../exoclaw-channel-codemod/sync.sh discord --apply
uv run pytest packages/exoclaw-channel-discord/tests/
```
