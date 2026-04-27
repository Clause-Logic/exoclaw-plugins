"""Smoke test for ``exoclaw_firmware.app``.

Hits the import graph and the ``build_agent`` factory under
CPython so we catch wiring mistakes (missing exports, mismatched
signatures, non-MP-friendly imports) without needing a chip.

Doesn't touch the network — ``run_demo`` is the integration path
and exercising it requires a real LLM endpoint."""

from __future__ import annotations

from exoclaw._compat import Path
from exoclaw_firmware import SerialChannel, build_agent, run_demo, run_serial_app


def test_public_surface_exported() -> None:
    """All entry points are importable from the package root."""
    assert callable(build_agent)
    assert callable(run_demo)
    assert callable(run_serial_app)
    assert callable(SerialChannel)


def test_serial_channel_satisfies_channel_protocol() -> None:
    """``SerialChannel`` implements ``Channel`` (``start``, ``stop``,
    ``send``, ``name`` attribute) so the channel manager can
    dispatch to it like any other channel."""
    ch = SerialChannel()
    assert ch.name == "serial"
    for method in ("start", "stop", "send"):
        assert callable(getattr(ch, method))


def test_build_agent_constructs_pair(tmp_path: Path) -> None:
    """Factory returns a (provider, conversation) pair the caller
    can drive a turn through. Doesn't actually call the LLM."""
    provider, conversation = build_agent(
        workspace=tmp_path,
        api_key="sk-test",
        model="gpt-4o-mini",
    )
    try:
        # Provider exposes the LLMProvider protocol surface used
        # by the agent loop.
        assert callable(getattr(provider, "chat"))
        assert provider.get_default_model() == "gpt-4o-mini"
        # Conversation's three core methods are present.
        for name in ("build_prompt", "record", "clear"):
            assert callable(getattr(conversation, name))
    finally:
        # provider owns its HTTPClient — close to silence the
        # unawaited-coroutine ResourceWarning.
        import asyncio

        asyncio.run(provider.close())
