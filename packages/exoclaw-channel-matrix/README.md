# exoclaw-channel-matrix

Matrix channel for [exoclaw](https://github.com/Clause-Logic/exoclaw) — `matrix-nio` long-poll sync, end-to-end encryption, threads, media, mentions. Decentralized chat (matrix.org or any Matrix homeserver).

## Install

```bash
pip install exoclaw-channel-matrix
```

The `matrix-nio[e2e]` dependency requires `python-olm` for E2E encryption — needs `libolm-dev` (Linux) or `libolm` (macOS) installed system-wide. On Windows, E2E is disabled (the dep is gated).

## Setup

1. Get a Matrix account on any homeserver (e.g. `matrix.org`, or self-hosted Synapse/Dendrite).
2. The first run will log in with `password` and persist an `access_token` + `device_id` to disk under `EXOCLAW_DATA_DIR/matrix/<homeserver>_<user>.session.json`. Subsequent runs use the saved token. You can also pre-fill `access_token` + `device_id` to skip the password login entirely.
3. The bot's E2E encryption store lives in `EXOCLAW_DATA_DIR/matrix/matrix-store/` — back this up to preserve key history.

## Use

```python
import asyncio
from exoclaw_nanobot import create
from exoclaw_channel_matrix import MatrixChannel, MatrixConfig

async def main() -> None:
    matrix = MatrixChannel(MatrixConfig(
        enabled=True,
        homeserver="https://matrix.org",
        user_id="@yourbot:matrix.org",
        password="...",                   # used once; access_token persisted after
        allow_from=["@you:matrix.org"],   # users allowed to message the bot
    ))
    bot = await create(extra_channels=[matrix])
    await bot.run()

asyncio.run(main())
```

## Config

| Field | Default | Description |
|---|---|---|
| `homeserver` | `"https://matrix.org"` | Homeserver URL |
| `user_id` | — | Bot user ID, e.g. `@bot:matrix.org` (required) |
| `password` | — | Bot password (used for first login; persisted as `access_token` after) |
| `access_token` / `device_id` | — | Pre-filled credentials; skip the password login |
| `e2ee_enabled` | `True` | Participate in encrypted rooms (requires `matrix-nio[e2e]`) |
| `allow_from` | `[]` | Matrix user IDs allowed to DM the bot |
| `group_policy` | `"open"` | `"open"`, `"mention"`, or `"allowlist"` (use `group_allow_from`) |
| `allow_room_mentions` | `False` | Whether `@room` triggers a response |
| `max_media_bytes` | `20MB` | Cap on inbound media downloads |
| `streaming` | `False` | Stream the agent's reply via Matrix message edits (less reliable than per-message send) |

## Workspace footprint

Matrix is one of the channels with **load-bearing persistent state**:
- `EXOCLAW_DATA_DIR/matrix/matrix-store/` — E2E encryption keys (sled DB). Lose this and you can't decrypt past messages.
- `EXOCLAW_DATA_DIR/matrix/<host>_<user>.session.json` — access token + device ID

If you're running in Docker, mount a volume at `EXOCLAW_DATA_DIR` (default `~/.exoclaw/data`) or you'll lose decryption on every restart.

## Audit boundary

Vendored from HKUDS/nanobot via codemod. What's committed: upstream snapshot in `vendor/`, optional patches in `patches/`, plus the small bootstrap files. The shipped `channel.py` and `tests/test_channel.py` are gitignored — generated at build time by [`exoclaw-channel-codemod`](../exoclaw-channel-codemod/), included in the wheel via the hatch hook. See [`exoclaw-nanobot-compat/README.md`](../exoclaw-nanobot-compat/README.md) for the full pattern.

## Maintenance

```bash
echo "<new-hkuds-sha>" > vendor/SHA
UPSTREAM=~/hkuds-nanobot bash ../exoclaw-channel-codemod/sync.sh matrix --apply
uv run pytest packages/exoclaw-channel-matrix/tests/
```
