"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ChannelsConfig(Base):
    """Configuration for chat channels."""

    # Explicit list of channel factory import paths to load, e.g.:
    #   ["nanobot_telegram:create_channel", "nanobot_discord:create_channel"]
    # Each factory receives (config: dict, bus: MessageBus) -> Channel | None.
    load: list[str] = Field(default_factory=list)
    # Opaque per-channel config blobs keyed by channel name.
    # Each channel package is responsible for parsing its own section.
    config: dict[str, Any] = Field(default_factory=dict)


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    memory_window: int = 100
    reasoning_effort: str | None = None  # low / medium / high — enables LLM thinking mode


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
