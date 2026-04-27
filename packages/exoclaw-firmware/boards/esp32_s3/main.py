"""ESP32-S3 main entry point — runs after ``boot.py``.

Loads the agent stack via ``exoclaw_firmware.app`` and drives one
demo turn against OpenAI. Replace the demo call with a real
channel loop (HTTP webhook, MQTT subscriber, polling queue) once
the smoke test passes.
"""

import asyncio

from exoclaw._compat import Path
from exoclaw_firmware import run_demo

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
    response = await run_demo(
        workspace=workspace,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    print("main: assistant response:")
    print(response or "(no content — model returned tool calls)")


asyncio.run(_main())
