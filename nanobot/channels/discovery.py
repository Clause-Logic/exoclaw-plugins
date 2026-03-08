"""
Channel loader — imports factory callables listed explicitly in config.

Config shape:

    channels:
      load:
        - "nanobot_telegram:create_channel"
        - "nanobot_discord:create_channel"
      config:
        telegram:
          token: "..."
          allowFrom: ["*"]
        discord:
          token: "..."
          allowFrom: ["*"]

Each factory must satisfy:

    def create_channel(config: dict, bus: MessageBus) -> Channel | None:
        ...

Return None to signal the channel is disabled or not configured.
"""

from __future__ import annotations

import importlib
from typing import Any

from loguru import logger

from nanobot.bus.protocol import Bus
from nanobot.channels.protocol import Channel


def load_channels(channels_config: Any, bus: Bus) -> list[Channel]:
    """
    Import and call each factory listed in channels_config.load.

    Args:
        channels_config: The ChannelsConfig object from root config.
        bus: The message bus passed to each factory.

    Returns:
        List of Channel instances returned by the factories.
    """
    channels: list[Channel] = []
    config_blobs: dict = channels_config.config or {}

    for factory_path in channels_config.load:
        if ":" not in factory_path:
            logger.error(
                "Invalid channel factory path '{}' — expected 'module:callable'",
                factory_path,
            )
            continue

        module_path, factory_name = factory_path.rsplit(":", 1)

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.error("Cannot import channel module '{}': {}", module_path, e)
            continue

        factory = getattr(module, factory_name, None)
        if factory is None:
            logger.error(
                "Module '{}' has no attribute '{}'", module_path, factory_name
            )
            continue

        try:
            ch = factory(config_blobs, bus)
        except Exception as e:
            logger.error("Channel factory '{}' raised: {}", factory_path, e)
            continue

        if ch is not None:
            channels.append(ch)
            logger.info("Loaded channel '{}' via '{}'", ch.name, factory_path)

    return channels
