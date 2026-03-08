"""Chat channels — protocol, manager, and explicit loader."""

from nanobot.channels.base import BaseChannel
from nanobot.channels.discovery import load_channels
from nanobot.channels.manager import ChannelManager
from nanobot.channels.protocol import Channel

__all__ = ["BaseChannel", "Channel", "ChannelManager", "load_channels"]
