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


def test_build_agent_extra_models_registers_second_deployment(tmp_path: Path) -> None:
    """When the caller passes ``extra_models``, the provider gets a
    second ``Deployment`` so a tool (e.g. ``WebSearchTool``) can
    call ``provider.chat(model=<extra-name>)`` without hitting the
    "no deployment for model" guard.

    The OpenRouter pattern: ``minimax/minimax-m2.7`` for chat,
    ``google/gemma-4-26b-a4b-it:online`` for search, both behind
    one OpenRouter key — the second deployment shares the chat
    deployment's ``base_url + api_key``."""
    chat_model = "minimax/minimax-m2.7"
    search_model = "google/gemma-4-26b-a4b-it:online"
    provider, _conversation = build_agent(
        workspace=tmp_path,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        model=chat_model,
        extra_models={search_model: "https://openrouter.ai/api/v1"},
    )
    try:
        # Both deployment keys should be registered on the
        # provider — accessing the private dict is fine in a
        # smoke test for wiring.
        deployments = provider._deployments
        assert chat_model in deployments
        assert search_model in deployments
        # Default still points at the chat model so normal turns
        # don't accidentally route through the search deployment.
        assert provider.get_default_model() == chat_model
    finally:
        import asyncio

        asyncio.run(provider.close())
