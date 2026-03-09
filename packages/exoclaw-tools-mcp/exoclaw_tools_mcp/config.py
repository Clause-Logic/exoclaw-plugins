"""MCP server configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    # Transport type: "stdio", "sse", or "streamableHttp".
    # If None, inferred from command/url.
    type: str | None = None

    # stdio transport
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None

    # HTTP transports (sse, streamableHttp)
    url: str | None = None
    headers: dict[str, str] | None = None

    # Tool call timeout in seconds
    tool_timeout: int = 30
