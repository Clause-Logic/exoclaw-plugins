# exoclaw-tools-mcp

MCP client for exoclaw — connects to MCP servers over stdio, SSE, or streamable HTTP and registers their tools as native exoclaw tools.

## Install

```
pip install exoclaw-tools-mcp
```

## Usage

```python
from contextlib import AsyncExitStack
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_tools_mcp.tool import connect_mcp_servers
from exoclaw_tools_mcp.config import MCPServerConfig

stack = AsyncExitStack()
registry = ToolRegistry()

await connect_mcp_servers(
    mcp_servers={
        "filesystem": MCPServerConfig(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
        "my-api":     MCPServerConfig(url="https://example.com/mcp"),
    },
    registry=registry,
    stack=stack,
)
# registry now contains all tools from both servers as mcp_<server>_<tool> entries
```

Transport type is inferred automatically: `command` → stdio; URLs ending in `/sse` → SSE; other URLs → streamable HTTP. Custom headers and per-tool timeouts are supported via `MCPServerConfig`.
