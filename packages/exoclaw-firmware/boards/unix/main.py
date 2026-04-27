"""Unix-port main entry point — runs after ``boot.py``.

Same chat loop the ESP32-S3 variant runs, but reads credentials
from environment variables instead of ``secrets.py``. Lets you
``OPENAI_API_KEY=… mise run sim`` from a host without copying
real keys into a board-specific config file.

Use case: develop the firmware logic without flashing a board.
The unix port supports everything the agent stack needs —
``asyncio.open_connection`` with TLS, file I/O, ``input()`` /
``print()`` for the serial channel — so the same ``run_serial_chat``
loop works against a terminal instead of USB-CDC.
"""

import asyncio
import os

from exoclaw._compat import Path
from exoclaw_firmware import run_serial_app

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not set — export it in your shell or "
        "mise.local.toml before running ``mise run sim``"
    )

workspace = Path(os.getenv("EXOCLAW_WORKSPACE") or ".sim-workspace")
base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
# Optional heartbeat tick to flush ``wake_mode="next-heartbeat"``
# cron jobs. Unset / 0 → no coalescing (every cron fire wakes
# immediately, the chatty default that keeps sim logs simple).
# Set ``EXOCLAW_HEARTBEAT_MS=300000`` (5 min) for a chip-style
# coalescing test.
_hb_env = os.getenv("EXOCLAW_HEARTBEAT_MS") or ""
heartbeat_interval_ms: int | None = int(_hb_env) if _hb_env.isdigit() and int(_hb_env) > 0 else None


async def _main() -> None:
    print("main: workspace={} model={}".format(workspace, model))
    if heartbeat_interval_ms:
        print("main: heartbeat every {}ms".format(heartbeat_interval_ms))
    print("main: ready — type a message and press enter (Ctrl-C to exit)")
    await run_serial_app(
        workspace=workspace,
        api_key=api_key,
        base_url=base_url,
        model=model,
        heartbeat_interval_ms=heartbeat_interval_ms,
    )


asyncio.run(_main())
