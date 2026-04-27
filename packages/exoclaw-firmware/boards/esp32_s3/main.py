"""ESP32-S3 main entry point — runs after ``boot.py``.

Drives an interactive chat over USB-CDC serial. Plug the board
into a host, open ``mpremote repl`` (or any serial terminal at
115200 baud), type messages, get responses. The simplest possible
channel — no chat-platform tokens, no webhook URLs.

Swap the call to ``run_serial_chat`` for ``run_demo`` if you want
the single-turn smoke test instead, or for your own channel loop
once you build one (Telegram long-poll, MQTT subscriber, etc.) on
top of ``build_agent``.
"""

import asyncio

from exoclaw._compat import Path
from exoclaw_firmware import run_serial_app

try:
    import secrets  # type: ignore[import-not-found]
except ImportError:
    raise RuntimeError(
        "secrets.py not found — copy secrets.py.example and fill in "
        "WIFI_SSID / WIFI_PASSWORD / OPENAI_API_KEY"
    )


async def _main() -> None:
    workspace = Path(getattr(secrets, "WORKSPACE", "/sd/exoclaw"))
    api_key = secrets.OPENAI_API_KEY
    base_url = getattr(secrets, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = getattr(secrets, "OPENAI_MODEL", "gpt-4o-mini")

    print("main: workspace={} model={}".format(workspace, model))
    print("main: ready — type a message and press enter (Ctrl-C to exit)")
    await run_serial_app(
        workspace=workspace,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


asyncio.run(_main())
