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


async def run_serial_app(
    *,
    workspace: Path,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    chat_id: str = "serial:default",
    prompt: str = "you> ",
    reply_prefix: str = "bot> ",
    tools: "list | None" = None,
    extra_channels: "list | None" = None,
    enable_cron: bool = True,
    heartbeat_interval_ms: int | None = None,
) -> None:
    """Run the full agent app with USB-CDC as a baseline channel.

    Builds the standard exoclaw stack (provider, conversation,
    bus, agent loop, channel manager) with :class:`SerialChannel`
    always wired in as a baseline. Cron firings, heartbeat ticks,
    and the ``message`` tool all flow through the same bus and reach
    you over USB-CDC.

    Loop runs until ``KeyboardInterrupt`` (Ctrl-C in the host
    terminal) or the chip resets. Each turn is persisted to the
    session JSONL so consolidation / memory still happen.

    ``tools`` bolts agent-callable tools (message, web search, …)
    onto the loop on top of the built-in cron tool. ``extra_channels``
    adds non-serial channels (Telegram long-poll, MQTT, etc.)
    alongside ``SerialChannel`` so a user can talk to the chip from
    the cloud OR from a USB cable using the same agent state.

    ``enable_cron`` (default ``True``) wires up the
    ``LocalCronBackend`` + ``CronTool`` so the agent can schedule
    its own jobs. Persisted to ``workspace/cron.json`` and survives
    reboots. Set to ``False`` for a chat-only chip.

    ``heartbeat_interval_ms`` (default ``None``) opts the cron
    service into periodic flushing of jobs scheduled with
    ``wake_mode="next-heartbeat"``. Only meaningful when
    ``enable_cron=True``. ``None`` disables coalescing — each cron
    fire wakes the agent immediately. A typical chip value is
    ``5 * 60 * 1000`` (5 minutes); a typical "quiet hours" value
    is ``60 * 60 * 1000``. Jobs scheduled with the default
    ``wake_mode="now"`` are unaffected.
    """
    from exoclaw.agent.tools.protocol import Tool
    from exoclaw.app import Exoclaw
    from exoclaw.bus.events import InboundMessage
    from exoclaw.bus.queue import MessageBus
    from exoclaw.channels.protocol import Channel

    from exoclaw_firmware.channel import SerialChannel

    provider, conversation = build_agent(
        workspace=workspace,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    serial = SerialChannel(chat_id=chat_id, prompt=prompt, reply_prefix=reply_prefix)
    # Avoid ``[serial, *extra]`` — MicroPython 1.27 doesn't support
    # PEP 448 list-unpacking inside list literals. Annotate the list
    # as ``list[Channel]`` so ty doesn't narrow to
    # ``list[SerialChannel]`` and reject the extend.
    channels: list[Channel] = [serial]
    if extra_channels:
        channels.extend(extra_channels)

    # Build the bus up-front so the cron backend can publish
    # inbound messages onto it when jobs fire. Exoclaw normally
    # builds its own bus; passing one in lets us pre-wire the
    # cron-fire callback.
    bus = MessageBus()

    # Optional cron — start the timer task before the agent loop
    # so jobs that fire during boot (e.g. ``at`` schedules in the
    # past after a reboot) reach the agent on the first cycle.
    cron_service = None
    all_tools: list[Tool] = list(tools or [])
    if enable_cron:
        from exoclaw_tools_cron.service import CronService, LocalCronBackend
        from exoclaw_tools_cron.tool import CronTool
        from exoclaw_tools_cron.types import CronJob

        async def _on_cron_job(job: CronJob) -> str | None:
            """Cron-fire callback — publish a synthetic inbound
            message so the agent processes the job's prompt
            exactly like a user-typed turn. The reply goes back
            over whichever channel the job targets (default
            serial) via the standard bus dispatch path."""
            target_channel = job.payload.channel or serial.name
            target_chat = job.payload.to or chat_id
            await bus.publish_inbound(
                InboundMessage(
                    channel=target_channel,
                    sender_id="cron",
                    chat_id=target_chat,
                    content=job.payload.message,
                )
            )
            return None

        cron_service = CronService(
            store_path=workspace / "cron.json",
            on_job=_on_cron_job,
            heartbeat_interval_ms=heartbeat_interval_ms,
        )
        cron_backend = LocalCronBackend(service=cron_service)
        # ``CronTool`` structurally satisfies the ``Tool`` Protocol
        # (``@runtime_checkable``) but ty can't prove the
        # subtype — same situation as ``DBOSExecutor`` /
        # ``Executor`` elsewhere in the workspace.
        all_tools.append(CronTool(backend=cron_backend))  # type: ignore[invalid-argument-type]
        await cron_service.start()

    app = Exoclaw(
        provider=provider,
        conversation=conversation,
        channels=channels,
        tools=all_tools,
        bus=bus,
        model=model,
    )
    try:
        await app.run()
    finally:
        if cron_service is not None:
            # ``CronService.stop`` is sync — cancels the timer task
            # in-place, no await needed.
            cron_service.stop()
        await provider.close()
