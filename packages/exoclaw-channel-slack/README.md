# exoclaw-channel-slack

Slack channel for [exoclaw](https://github.com/Clause-Logic/exoclaw) — Socket Mode inbound, Block Kit + mrkdwn outbound, file uploads, threads, reactions.

## Install

```bash
pip install exoclaw-channel-slack
```

## Setup

You need a Slack app with both a **bot token** (`xoxb-…`) and an **app-level token** (`xapp-…`):

1. <https://api.slack.com/apps> → **Create New App** → from scratch
2. **Socket Mode** → enable. Generate an app-level token with `connections:write` scope. That's your `app_token`.
3. **OAuth & Permissions** → bot scopes:
   - `app_mentions:read`, `channels:history`, `groups:history`, `im:history`, `mpim:history`
   - `chat:write`, `files:write`, `reactions:write`, `users:read`
4. **Event Subscriptions** → enable. Subscribe to bot events: `message.channels`, `message.groups`, `message.im`, `message.mpim`, `app_mention`.
5. Install the app to your workspace. Copy the bot token (`xoxb-…`).

## Use

```python
import asyncio
from exoclaw_nanobot import create
from exoclaw_channel_slack import SlackChannel, SlackConfig

async def main() -> None:
    slack = SlackChannel(SlackConfig(
        enabled=True,
        bot_token="xoxb-...",
        app_token="xapp-...",
        allow_from=["U01ABCD1234"],   # Slack user IDs allowed to DM the bot
    ))
    bot = await create(extra_channels=[slack])
    await bot.run()

asyncio.run(main())
```

## Config

| Field | Default | Description |
|---|---|---|
| `bot_token` | — | `xoxb-…` bot token (required) |
| `app_token` | — | `xapp-…` app-level token for Socket Mode (required) |
| `allow_from` | `[]` | Slack user IDs allowed to DM the bot (empty = deny all DMs) |
| `group_policy` | `"mention"` | `"mention"` (respond only when @mentioned) or `"open"` (every message) |
| `group_allow_from` | `[]` | Channel IDs allowed for group messages (empty = all when `group_policy="mention"`) |
| `dm.policy` | `"open"` | `"open"` or `"allowlist"` (use `dm.allow_from`) |
| `reply_in_thread` | `True` | Start a thread on the original message instead of replying in-channel |
| `react_emoji` | `"eyes"` | Reaction added when message received |
| `done_emoji` | `"white_check_mark"` | Reaction added when reply sent |
| `include_thread_context` | `True` | Fetch up to `thread_context_limit` prior messages when replying in a thread |

Sessions are scoped per-thread when `reply_in_thread=True` — each `(channel, thread_ts)` is its own conversation.

## Audit boundary

Vendored from HKUDS/nanobot via codemod. What's committed: upstream snapshot in `vendor/`, optional channel-specific tweaks in `patches/`, plus `__init__.py` / `conftest.py` / `hatch_build.py` / `pyproject.toml`. The shipped `channel.py` and `tests/test_channel.py` are gitignored — generated at build time by [`exoclaw-channel-codemod`](../exoclaw-channel-codemod/), included in the wheel via the hatch hook. See [`exoclaw-nanobot-compat/README.md`](../exoclaw-nanobot-compat/README.md) for the full pattern.

## Maintenance

```bash
echo "<new-hkuds-sha>" > vendor/SHA
UPSTREAM=~/hkuds-nanobot bash ../exoclaw-channel-codemod/sync.sh slack --apply
uv run pytest packages/exoclaw-channel-slack/tests/
```
