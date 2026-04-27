"""Deployable exoclaw image for MicroPython boards.

This package bundles everything you need to run an exoclaw agent
on a microcontroller: core ``exoclaw`` + the MP-compatible plugin
set (``exoclaw-conversation``, ``exoclaw-provider-openai``), plus
board-specific boot wrappers under ``boards/`` that handle WiFi /
SD-card / clock setup.

Public entry point: :func:`exoclaw_firmware.app.run_demo` — builds
a minimal agent (OpenAI provider + file-backed conversation) and
runs a single turn against a hardcoded prompt. Used to verify the
whole stack works on hardware before wiring a real channel.

The agent loop / channel layer is intentionally not in this v0 —
choose-your-own-adventure: HTTP webhook, MQTT subscriber, polling
queue, or serial REPL. Each runs on top of the provider +
conversation pair this package builds.
"""

from exoclaw_firmware.app import build_agent, run_demo

__all__ = ["build_agent", "run_demo"]
