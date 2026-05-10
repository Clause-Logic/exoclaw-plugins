# exoclaw-channel-whatsapp

WhatsApp channel for [exoclaw](https://github.com/Clause-Logic/exoclaw) — connects via the **Baileys** Node.js sidecar bridge that this package manages (copies into `EXOCLAW_DATA_DIR/whatsapp-auth/`, runs `npm start` on it, talks to it over a WebSocket).

> **Operational caveat**: this package will spawn `npm` and a Node child process. If you don't want a Node sidecar in your container, this isn't the channel for you.

## Install

```bash
pip install exoclaw-channel-whatsapp
```

You also need `node` + `npm` on `$PATH`. The bridge auto-installs its own dependencies on first run.

## Setup

1. First run will print a QR code in the terminal — scan it from WhatsApp on your phone (Settings → Linked Devices → Link a Device).
2. The bridge persists its session under `EXOCLAW_DATA_DIR/whatsapp-auth/`. Keep that directory across restarts to avoid re-pairing.
3. The bridge token (auto-generated on first run) is stored alongside the session for the WebSocket auth handshake.

## Use

```python
import asyncio
from exoclaw_nanobot import create
from exoclaw_channel_whatsapp import WhatsAppChannel, WhatsAppConfig

async def main() -> None:
    wa = WhatsAppChannel(WhatsAppConfig(
        enabled=True,
        allow_from=["1234567890"],     # phone numbers (without +) allowed to message
    ))
    bot = await create(extra_channels=[wa])
    await bot.run()

asyncio.run(main())
```

## Config

| Field | Default | Description |
|---|---|---|
| `bridge_url` | `"ws://localhost:3001"` | Where the bridge listens. Override if you run the bridge separately. |
| `bridge_token` | auto-generated | Auth token for the bridge WebSocket. Generated and persisted on first run. |
| `allow_from` | `[]` | Phone numbers (E.164 without `+`) allowed to message the bot |
| `group_policy` | `"open"` | `"open"` (respond to all group messages from allowed senders) or `"mention"` |

## Workspace footprint

This is the **most operationally heavy** of the channel packages:
- `EXOCLAW_DATA_DIR/whatsapp-auth/bridge-token` — auth token
- `EXOCLAW_DATA_DIR/whatsapp-auth/<session>` — Baileys session keys (lose them and you re-pair)
- `EXOCLAW_DATA_DIR/whatsapp-auth/<bridge-tree>` — copied Node bridge source
- A child `node` process running the bridge

If you're running in Docker: mount a volume at `EXOCLAW_DATA_DIR` (default `~/.exoclaw/data`) or you'll lose pairing on every restart, and ensure `node`/`npm` are in the image.

## Audit boundary

Vendored from HKUDS/nanobot via codemod. What's committed: upstream snapshot in `vendor/`, optional patches in `patches/`, plus the small bootstrap files. The shipped `channel.py` and `tests/test_channel.py` are gitignored — generated at build time by [`exoclaw-channel-codemod`](../exoclaw-channel-codemod/), included in the wheel via the hatch hook. See [`exoclaw-nanobot-compat/README.md`](../exoclaw-nanobot-compat/README.md) for the full pattern.

## Maintenance

```bash
echo "<new-hkuds-sha>" > vendor/SHA
UPSTREAM=~/hkuds-nanobot bash ../exoclaw-channel-codemod/sync.sh whatsapp --apply
uv run pytest packages/exoclaw-channel-whatsapp/tests/
```
