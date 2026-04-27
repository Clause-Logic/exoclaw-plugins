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
- :func:`exoclaw_firmware.app.run_serial_chat` — interactive chat
  over USB-CDC serial. The minimum-viable channel: plug the board
  into a host, open ``mpremote repl`` (or any serial terminal),
  type messages. No network channel, no API tokens beyond OpenAI.

Higher-bandwidth channels (HTTP webhook, MQTT subscriber,
WebSocket, BLE) are choose-your-own on top of ``build_agent``.
"""

from exoclaw_firmware.app import build_agent, run_demo, run_serial_chat

__all__ = ["build_agent", "run_demo", "run_serial_chat"]
