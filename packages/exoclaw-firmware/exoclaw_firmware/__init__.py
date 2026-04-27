"""Deployable exoclaw image for MicroPython boards.

This package bundles everything you need to run an exoclaw agent
on a microcontroller: core ``exoclaw`` + the MP-compatible plugin
set (``exoclaw-conversation``, ``exoclaw-provider-openai``), plus
board-specific boot wrappers under ``boards/`` that handle WiFi /
SD-card / clock setup.

Public entry points:

- :func:`exoclaw_firmware.app.build_agent` — returns
  ``(provider, conversation)`` for callers wiring their own loop.
- :func:`exoclaw_firmware.app.run_demo` — single-turn smoke test
  against a hardcoded prompt. Use right after first flash to
  prove the stack works.
- :func:`exoclaw_firmware.app.run_serial_app` — full agent app
  with USB-CDC as the channel. Cron firings, heartbeat ticks, and
  the ``message`` tool all reach you over the same stdin/stdout
  the user types into. Drop tools and additional channels in via
  the function's kwargs.
- :class:`exoclaw_firmware.channel.SerialChannel` — the
  ``Channel`` implementation. First-party, baked-in: every chip
  has USB-CDC and needs *some* way for a human to talk to it.

Higher-bandwidth channels (HTTP webhook, MQTT subscriber,
WebSocket, BLE) are choose-your-own and live as separate plugins.
"""

from exoclaw_firmware.app import build_agent, run_demo, run_serial_app
from exoclaw_firmware.channel import SerialChannel

__all__ = ["SerialChannel", "build_agent", "run_demo", "run_serial_app"]
