"""Non-interactive pipe channel — reads lines from stdin, writes to stdout.

Works in scripts, CI, pipes, and non-TTY environments. No terminal required.

Usage:
    echo "what is 2+2?" | python app.py
    cat prompts.txt | python app.py
    python app.py <<< "summarize this repo"
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from loguru import logger

from exoclaw.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from exoclaw.bus.protocol import Bus


class PipeChannel:
    """
    Non-interactive channel that reads from stdin line-by-line.

    Each line becomes an inbound message. Responses are written to stdout.
    EOF on stdin triggers shutdown.

    Implements the exoclaw Channel protocol.
    """

    name = "pipe"

    def __init__(self, chat_id: str = "pipe") -> None:
        self._chat_id = chat_id
        self._running = False
        self._bus: Bus | None = None

    async def start(self, bus: Bus) -> None:
        """Read stdin lines and publish as inbound messages."""
        self._bus = bus
        self._running = True

        turn_done = asyncio.Event()
        turn_done.set()
        turn_response: list[str] = []

        async def _consume_outbound() -> None:
            while self._running:
                try:
                    msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                    if msg.metadata and msg.metadata.get("_progress"):
                        # Print progress to stderr so stdout stays clean
                        print(f"  > {msg.content}", file=sys.stderr)
                    elif not turn_done.is_set():
                        if msg.content:
                            turn_response.append(msg.content)
                        turn_done.set()
                    elif msg.content:
                        # Async message (subagent result, etc.)
                        print(msg.content, flush=True)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

        outbound_task = asyncio.create_task(_consume_outbound())

        try:
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            loop = asyncio.get_event_loop()
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)

            while self._running:
                try:
                    line = await reader.readline()
                    if not line:
                        # EOF
                        break
                    text = line.decode().rstrip("\n\r")
                    if not text:
                        continue

                    logger.debug("pipe input: {}", text)

                    turn_done.clear()
                    turn_response.clear()

                    await bus.publish_inbound(InboundMessage(
                        channel=self.name,
                        sender_id="user",
                        chat_id=self._chat_id,
                        content=text,
                    ))

                    await turn_done.wait()

                    if turn_response:
                        print(turn_response[0], flush=True)

                except asyncio.CancelledError:
                    break
        finally:
            self._running = False
            outbound_task.cancel()
            await asyncio.gather(outbound_task, return_exceptions=True)

    async def stop(self) -> None:
        """Signal shutdown."""
        self._running = False
        logger.info("PipeChannel stopping")

    async def send(self, msg: OutboundMessage) -> None:
        """Write an outbound message to stdout."""
        if msg.content:
            print(msg.content, flush=True)
