"""Channel protocol — the only thing ChannelManager depends on."""

from typing import Protocol, runtime_checkable

from nanobot.bus.events import OutboundMessage


@runtime_checkable
class Channel(Protocol):
    """
    Protocol for chat channel implementations.

    Any object with these three methods and a name attribute satisfies
    this protocol — no inheritance from BaseChannel required.
    """

    name: str

    async def start(self) -> None:
        """Connect to the platform and begin receiving messages."""
        ...

    async def stop(self) -> None:
        """Disconnect and release resources."""
        ...

    async def send(self, msg: OutboundMessage) -> None:
        """Deliver an outbound message to the platform."""
        ...
