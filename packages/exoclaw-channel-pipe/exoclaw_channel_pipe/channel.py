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

    Each line becomes an inbound message. Responses arrive via send()
    (called by nanobot's outbound dispatcher). EOF triggers shutdown.

    Implements the exoclaw Channel protocol.
    """

    name = "pipe"

    def __init__(self, chat_id: str | None = None) -> None:
        import uuid

        self._chat_id = chat_id or f"pipe-{uuid.uuid4().hex[:8]}"
        self._running = False
        self._turn_done = asyncio.Event()
        self._turn_done.set()

    async def start(self, bus: Bus) -> None:
        """Read stdin lines and publish as inbound messages."""
        self._running = True

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

                    self._turn_done.clear()

                    await bus.publish_inbound(InboundMessage(
                        channel=self.name,
                        sender_id="user",
                        chat_id=self._chat_id,
                        content=text,
                    ))

                    # Wait for response via send()
                    await self._turn_done.wait()

                except asyncio.CancelledError:
                    break
        finally:
            self._running = False

    async def stop(self) -> None:
        """Signal shutdown."""
        self._running = False
        self._turn_done.set()  # Unblock if waiting
        logger.info("PipeChannel stopping")

    async def send(self, msg: OutboundMessage) -> None:
        """Called by nanobot's outbound dispatcher with agent responses."""
        if msg.metadata and msg.metadata.get("_progress"):
            print(f"  > {msg.content}", file=sys.stderr, flush=True)
            return
        if msg.content:
            print(msg.content, flush=True)
        if not self._turn_done.is_set():
            self._turn_done.set()
