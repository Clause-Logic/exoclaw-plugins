# exoclaw-channel-email

Email channel for [exoclaw](https://github.com/Clause-Logic/exoclaw) — IMAP poll inbound, SMTP send outbound, attachment extraction, threaded replies via `In-Reply-To` headers. Pure-stdlib (`imaplib` + `smtplib`) — no extra runtime deps.

## Install

```bash
pip install exoclaw-channel-email
```

## Setup

For Gmail: enable 2FA, generate an [app password](https://myaccount.google.com/apppasswords), use that as `imap_password` and `smtp_password`. For other providers: use whatever credential pattern they expose for IMAP/SMTP.

## Use

```python
import asyncio
from exoclaw_nanobot import create
from exoclaw_channel_email import EmailChannel, EmailConfig

async def main() -> None:
    email = EmailChannel(EmailConfig(
        enabled=True,
        consent_granted=True,           # acknowledge bot will read INBOX

        imap_host="imap.gmail.com",
        imap_port=993,
        imap_username="bot@example.com",
        imap_password="...",            # app password for Gmail

        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_username="bot@example.com",
        smtp_password="...",
        from_address="bot@example.com",

        allow_from=["alice@example.com"],   # senders allowed to message the bot
    ))
    bot = await create(extra_channels=[email])
    await bot.run()

asyncio.run(main())
```

## Config

| Field | Default | Description |
|---|---|---|
| `consent_granted` | `False` | Must be `True` — acknowledges the bot will read your inbox. |
| `imap_host` / `imap_port` / `imap_username` / `imap_password` | — | IMAP credentials (required) |
| `imap_mailbox` | `"INBOX"` | Folder to poll |
| `imap_use_ssl` | `True` | Use IMAPS |
| `smtp_host` / `smtp_port` / `smtp_username` / `smtp_password` | — | SMTP credentials (required) |
| `smtp_use_tls` / `smtp_use_ssl` | `True` / `False` | STARTTLS or implicit TLS |
| `from_address` | — | `From:` header on outbound replies (required) |
| `allow_from` | `[]` | Email addresses allowed to message the bot (empty = deny all) |
| `auto_reply_enabled` | `True` | Send replies. `False` = read-only / archival mode. |
| `poll_interval_seconds` | `30` | IMAP poll cadence |
| `mark_seen` | `True` | Mark messages `\Seen` after processing |
| `max_body_chars` | `12000` | Truncate inbound bodies past this length |
| `subject_prefix` | `"Re: "` | Prepended to outbound reply subjects |

Sessions are scoped per-thread via `Message-ID`/`In-Reply-To` chain — replies in the same email thread share a conversation.

## Audit boundary

Vendored from HKUDS/nanobot via codemod. What's committed: upstream snapshot in `vendor/`, optional patches in `patches/`, plus the small bootstrap files. The shipped `channel.py` and `tests/test_channel.py` are gitignored — generated at build time by [`exoclaw-channel-codemod`](../exoclaw-channel-codemod/), included in the wheel via the hatch hook. See [`exoclaw-nanobot-compat/README.md`](../exoclaw-nanobot-compat/README.md) for the full pattern.

## Maintenance

```bash
echo "<new-hkuds-sha>" > vendor/SHA
UPSTREAM=~/hkuds-nanobot bash ../exoclaw-channel-codemod/sync.sh email --apply
uv run pytest packages/exoclaw-channel-email/tests/
```
