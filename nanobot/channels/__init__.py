"""Chat channels — protocol, manager, and explicit loader."""

from nanobot.channels.discovery import load_channels
from nanobot.channels.manager import ChannelManager
from nanobot.channels.protocol import Channel

__all__ = ["Channel", "ChannelManager", "load_channels"]
