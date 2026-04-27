"""Builds the agent stack from config.

Smoke test: ``run_demo`` constructs a provider + conversation,
runs one turn against a hardcoded prompt, and returns the
assistant text. Proves the full path (WiFi → TLS → OpenAI → SSE
parse → tool dispatch) works end-to-end on hardware. Caller
prints the result. Real channel integration (HTTP webhook / MQTT
/ polling queue) lives on top of ``build_agent``.

Configuration comes from a ``secrets`` module the caller provides
(``import secrets`` on MP — typically ``boards/<board>/secrets.py``,
gitignored). Keeps API keys out of the source tree.
"""

from __future__ import annotations

from exoclaw._compat import Path, get_logger
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_provider_openai import Deployment, OpenAIStreamingProvider

logger = get_logger()


def build_agent(
    *,
    workspace: Path,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    request_timeout: float = 60.0,
) -> "tuple[OpenAIStreamingProvider, DefaultConversation]":
    """Build a (provider, conversation) pair ready to drive a turn.

    Args:
        workspace: Directory under which sessions / memory / skills
            are stored. On a chip with an SD card, mount it at
            ``/sd`` and pass ``Path("/sd/exoclaw")``.
        api_key: OpenAI-compatible API key. Loaded from the
            board-specific ``secrets.py`` at the call site.
        base_url: Override for OpenAI-compatible endpoints (Groq,
            OpenRouter, local llama.cpp server, etc.).
        model: Default model name. Must match a deployment key.
        request_timeout: Seconds to wait for a complete response
            before the fallback chain engages. The MP HTTP client
            reuses this for connect + read budgets.

    Returns:
        ``(provider, conversation)`` — call ``await
        conversation.build_prompt(...)`` then ``await
        provider.chat(messages=...)`` then ``await
        conversation.record(...)`` to drive a turn manually.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    deployments = {
        model: Deployment(base_url=base_url, api_key=api_key),
    }
    provider = OpenAIStreamingProvider(
        default_model=model,
        deployments=deployments,
        request_timeout=request_timeout,
    )
    conversation = DefaultConversation.create(
        workspace=workspace,
        provider=provider,
        model=model,
    )
    return provider, conversation


async def run_demo(
    *,
    workspace: Path,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    prompt: str = "Say hello in five words or less.",
    session_id: str = "firmware:demo",
) -> str | None:
    """Run a single turn end-to-end and return the assistant's text.

    Smoke test for "does the whole stack work on this board?" —
    stand-alone, no channel, no agent loop. Useful as the first
    thing ``main.py`` calls after WiFi comes up.

    Returns the assistant content (or ``None`` if the model
    returned only tool calls — unlikely for the demo prompt but
    handled).
    """
    provider, conversation = build_agent(
        workspace=workspace,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    try:
        messages = await conversation.build_prompt(
            session_id=session_id,
            message=prompt,
        )
        logger.info("firmware_demo_send", **{"prompt.chars": len(prompt)})
        response = await provider.chat(messages=messages, model=model)
        await conversation.record(
            session_id,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.content or ""},
            ],
        )
        return response.content
    finally:
        await provider.close()


async def run_serial_chat(
    *,
    workspace: Path,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    session_id: str = "firmware:serial",
    prompt_prefix: str = "you> ",
    reply_prefix: str = "bot> ",
) -> None:
    """Interactive chat loop over USB serial.

    The chip's USB-CDC port maps directly to ``input()`` / ``print()``
    in MicroPython, so plugging the board into a host and opening a
    serial terminal (``mpremote repl``, ``screen``, ``minicom``) is
    enough to talk to the agent. No network channel, no API
    plumbing — just stdin/stdout.

    Loop until the user sends EOF (Ctrl-D in most terminals) or
    interrupts (Ctrl-C). Each turn is persisted to the session
    JSONL so consolidation / memory still happen.
    """
    provider, conversation = build_agent(
        workspace=workspace,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    try:
        while True:
            try:
                line = input(prompt_prefix)
            except (EOFError, KeyboardInterrupt):
                # Clean exit on Ctrl-D / Ctrl-C — the host
                # terminal stays usable for ``mpremote repl``
                # afterwards.
                print()
                break
            line = line.strip()
            if not line:
                continue
            messages = await conversation.build_prompt(
                session_id=session_id,
                message=line,
            )
            response = await provider.chat(messages=messages, model=model)
            text = response.content or "(no content — model returned tool calls)"
            print(reply_prefix + text)
            await conversation.record(
                session_id,
                [
                    {"role": "user", "content": line},
                    {"role": "assistant", "content": response.content or ""},
                ],
            )
    finally:
        await provider.close()
