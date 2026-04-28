"""USB-CDC serial channel — first-party, baked into the firmware.

Implements the ``exoclaw.channels.protocol.Channel`` interface so
the firmware participates in the standard agent-loop / bus /
channel-manager pipeline. Reading lines from ``sys.stdin`` and
writing replies to ``sys.stdout`` becomes a real bus subscriber
rather than a synchronous ``input() → chat() → print()`` short-
circuit.

Why bake it into the firmware instead of shipping as a separate
plugin: every chip needs *some* way for a human to talk to it,
and USB-CDC is the one transport every MicroPython board ships
with by default. Cron jobs firing, heartbeat tasks completing,
and the ``message`` tool all need an outbound subscriber to
reach the user — without this channel the chip would be a
sealed box.

Other channels (Telegram long-poll, MQTT, WebSocket, BLE) live
as separate plugins because they're optional and platform-
specific. Serial is the floor.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from exoclaw._compat import get_logger
from exoclaw.bus.events import InboundMessage, OutboundMessage
from exoclaw.utils import create_isolated_task

if TYPE_CHECKING:
    from exoclaw.bus.protocol import Bus

logger = get_logger()


class SerialChannel:
    """USB-CDC channel implementing the exoclaw ``Channel`` protocol.

    The reader task polls ``sys.stdin`` for available data via
    ``select.poll`` (the cross-runtime API — CPython has both
    ``select`` and ``poll``, MicroPython only ships ``poll``).
    Works on the unix port, ESP32-S3 USB-CDC, and any MP target
    with stdin file-descriptor support. Each line becomes an
    ``InboundMessage``; ``send()`` writes outbound messages to
    ``sys.stdout``.

    No prompt-toolkit, no rich, no termios — just stdin/stdout
    so the chip's ~64-256 KiB heap doesn't blow up on imports.
    """

    name = "serial"

    def __init__(
        self,
        *,
        chat_id: str = "serial:default",
        prompt: str = "you> ",
        reply_prefix: str = "bot> ",
        poll_interval: float = 0.05,
        line_interceptor: "Any | None" = None,
    ) -> None:
        self._chat_id = chat_id
        self._prompt = prompt
        self._reply_prefix = reply_prefix
        self._poll_interval = poll_interval
        # Optional async ``Callable[[str], str | None]`` that gets
        # each line BEFORE it's published as an inbound message.
        # If it returns a string, that string is used instead of
        # the typed line — useful for transforming control tokens
        # (``/talk`` → live mic transcription on the unix sim).
        # If it returns ``None`` the original line passes through
        # unchanged. Lets the SerialChannel stay generic while
        # voice / vision / etc. plug in via composition.
        self._line_interceptor = line_interceptor
        self._running = False
        self._reader_task: asyncio.Task[None] | None = None

    async def start(self, bus: "Bus") -> None:
        """Spin up the stdin-reader task. The channel manager owns
        outbound dispatch (it calls ``send`` per-message), so we
        only need to drive the inbound side here."""
        self._running = True
        self._reader_task = create_isolated_task(self._read_loop(bus))
        # Print the initial prompt so the user knows the chip is
        # listening. Subsequent prompts print after each reply.
        sys.stdout.write(self._prompt)
        try:
            sys.stdout.flush()
        except (AttributeError, OSError):
            # Some MP boards' stdout doesn't have flush() — line
            # buffering on USB-CDC handles the writeback anyway.
            pass

    async def stop(self) -> None:
        """Signal the reader task to exit."""
        self._running = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

    async def send(self, msg: OutboundMessage) -> None:
        """Write an outbound message to stdout. Called by the
        channel manager when the agent loop produces a reply.

        Filters ``metadata['_progress']`` messages — those are
        in-flight tool-call status updates the agent emits during
        a turn. Without the filter the console gets spammy and
        the prompt repeats during streaming. Same convention
        ``exoclaw-channel-cli`` and ``exoclaw-channel-pipe`` use.
        """
        if msg.metadata and msg.metadata.get("_progress"):
            return
        content = msg.content or "(no content)"
        sys.stdout.write(self._reply_prefix + content + "\n")
        sys.stdout.write(self._prompt)
        try:
            sys.stdout.flush()
        except (AttributeError, OSError):
            pass

    async def _read_loop(self, bus: "Bus") -> None:
        """Poll stdin for available data; publish each line as an
        ``InboundMessage``. Non-blocking via ``select.poll`` so the
        agent loop's other tasks (cron firing, heartbeat ticks)
        keep running while we wait for input.

        ``select.poll`` is the cross-runtime API — CPython has both
        ``select`` and ``poll``; MicroPython only ships ``poll``.
        """
        import select

        try:
            poller = select.poll()
            poller.register(sys.stdin, select.POLLIN)
        except (OSError, ValueError, AttributeError):
            # Flip ``_running`` so the channel manager / app sees an
            # accurate "not listening" state rather than a phantom
            # active channel that never publishes.
            self._running = False
            logger.warning("serial_stdin_unselectable")
            return

        while self._running:
            events = poller.poll(0)
            if not events:
                await asyncio.sleep(self._poll_interval)
                continue
            line = sys.stdin.readline()
            if not line:
                # EOF — typically Ctrl-D on a terminal. End the
                # channel; the rest of the system keeps running
                # (e.g. cron jobs continue to fire even with no
                # interactive console).
                logger.info("serial_eof")
                self._running = False
                return
            content = line.rstrip("\r\n")
            if not content:
                # Blank line — re-prompt without bothering the agent.
                sys.stdout.write(self._prompt)
                try:
                    sys.stdout.flush()
                except (AttributeError, OSError):
                    pass
                continue
            if self._line_interceptor is not None:
                try:
                    replacement = await self._line_interceptor(content)
                except Exception as e:  # noqa: BLE001 — interceptor errors shouldn't kill the channel
                    logger.warning(
                        "serial_interceptor_failed",
                        **{"error": str(e)},
                    )
                    replacement = None
                if replacement is not None:
                    content = replacement
                if not content:
                    sys.stdout.write(self._prompt)
                    try:
                        sys.stdout.flush()
                    except (AttributeError, OSError):
                        pass
                    continue
            await bus.publish_inbound(
                InboundMessage(
                    channel=self.name,
                    sender_id="user",
                    chat_id=self._chat_id,
                    content=content,
                )
            )
