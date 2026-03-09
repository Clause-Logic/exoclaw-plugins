"""Tests for exoclaw-tools-mcp package."""

from __future__ import annotations

import asyncio
from typing import Any
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_tools_mcp.config import MCPServerConfig
from exoclaw_tools_mcp.tool import MCPToolWrapper, connect_mcp_servers


# ---------------------------------------------------------------------------
# MCPServerConfig
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_defaults(self) -> None:
        cfg = MCPServerConfig()
        assert cfg.type is None
        assert cfg.command is None
        assert cfg.args == []
        assert cfg.env is None
        assert cfg.url is None
        assert cfg.headers is None
        assert cfg.tool_timeout == 30

    def test_stdio_config(self) -> None:
        cfg = MCPServerConfig(command="npx", args=["-y", "server"], tool_timeout=60)
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "server"]
        assert cfg.tool_timeout == 60

    def test_sse_config(self) -> None:
        cfg = MCPServerConfig(url="http://localhost:8080/sse", headers={"Authorization": "Bearer tok"})
        assert cfg.url == "http://localhost:8080/sse"
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_streamable_http_config(self) -> None:
        cfg = MCPServerConfig(url="http://localhost:8080/mcp", type="streamableHttp")
        assert cfg.type == "streamableHttp"


# ---------------------------------------------------------------------------
# MCPToolWrapper
# ---------------------------------------------------------------------------


def _make_tool_def(name: str = "search", description: str = "Search", schema: dict[str, Any] | None = None) -> MagicMock:
    td = MagicMock()
    td.name = name
    td.description = description
    td.inputSchema = schema or {"type": "object", "properties": {"query": {"type": "string"}}}
    return td


def _make_session() -> MagicMock:
    session = MagicMock()
    session.call_tool = AsyncMock()
    return session


class TestMCPToolWrapperProperties:
    def test_name_prefixed(self) -> None:
        wrapper = MCPToolWrapper(_make_session(), "myserver", _make_tool_def("search"))
        assert wrapper.name == "mcp_myserver_search"

    def test_description(self) -> None:
        wrapper = MCPToolWrapper(_make_session(), "s", _make_tool_def(description="Do a search"))
        assert wrapper.description == "Do a search"

    def test_description_falls_back_to_name(self) -> None:
        td = _make_tool_def(name="mytool", description="")
        td.description = None
        wrapper = MCPToolWrapper(_make_session(), "s", td)
        assert wrapper.description == "mytool"

    def test_parameters(self) -> None:
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        wrapper = MCPToolWrapper(_make_session(), "s", _make_tool_def(schema=schema))
        assert wrapper.parameters == schema

    def test_parameters_fallback_when_none(self) -> None:
        td = _make_tool_def()
        td.inputSchema = None
        wrapper = MCPToolWrapper(_make_session(), "s", td)
        assert wrapper.parameters == {"type": "object", "properties": {}}

    def test_stores_original_name(self) -> None:
        wrapper = MCPToolWrapper(_make_session(), "srv", _make_tool_def("do_thing"))
        assert wrapper._original_name == "do_thing"


class TestMCPToolWrapperExecute:
    async def test_returns_text_content(self) -> None:
        session = _make_session()
        text_block = MagicMock()
        text_block.text = "result text"
        result_obj = MagicMock()
        result_obj.content = [text_block]
        session.call_tool.return_value = result_obj

        wrapper = MCPToolWrapper(session, "s", _make_tool_def())

        with patch("exoclaw_tools_mcp.tool.asyncio.wait_for", new=AsyncMock(return_value=result_obj)):
            with patch("exoclaw_tools_mcp.tool.types") as mock_types:
                mock_types.TextContent = type(text_block)
                result = await wrapper.execute(query="hello")

        assert result == "result text"

    async def test_non_text_content_stringified(self) -> None:
        session = _make_session()

        class ImageBlock:
            def __str__(self) -> str:
                return "image_data"

        block = ImageBlock()
        result_obj = MagicMock()
        result_obj.content = [block]
        session.call_tool.return_value = result_obj

        wrapper = MCPToolWrapper(session, "s", _make_tool_def())

        with patch("exoclaw_tools_mcp.tool.asyncio.wait_for", new=AsyncMock(return_value=result_obj)):
            with patch("exoclaw_tools_mcp.tool.types") as mock_types:
                mock_types.TextContent = type(None)  # block won't match
                result = await wrapper.execute()

        assert "image_data" in result

    async def test_empty_content_returns_no_output(self) -> None:
        session = _make_session()
        result_obj = MagicMock()
        result_obj.content = []
        session.call_tool.return_value = result_obj

        wrapper = MCPToolWrapper(session, "s", _make_tool_def())

        with patch("exoclaw_tools_mcp.tool.asyncio.wait_for", new=AsyncMock(return_value=result_obj)):
            with patch("exoclaw_tools_mcp.tool.types"):
                result = await wrapper.execute()

        assert result == "(no output)"

    async def test_timeout_returns_message(self) -> None:
        session = _make_session()
        wrapper = MCPToolWrapper(session, "s", _make_tool_def(), tool_timeout=5)

        with patch("exoclaw_tools_mcp.tool.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with patch("exoclaw_tools_mcp.tool.types"):
                result = await wrapper.execute()

        assert "timed out" in result
        assert "5s" in result

    async def test_multiple_text_blocks_joined(self) -> None:
        session = _make_session()

        class FakeText:
            def __init__(self, t: str) -> None:
                self.text = t

        block1 = FakeText("line one")
        block2 = FakeText("line two")
        result_obj = MagicMock()
        result_obj.content = [block1, block2]

        wrapper = MCPToolWrapper(session, "s", _make_tool_def())

        with patch("exoclaw_tools_mcp.tool.asyncio.wait_for", new=AsyncMock(return_value=result_obj)):
            with patch("exoclaw_tools_mcp.tool.types") as mock_types:
                mock_types.TextContent = FakeText
                result = await wrapper.execute()

        assert result == "line one\nline two"


# ---------------------------------------------------------------------------
# connect_mcp_servers
# ---------------------------------------------------------------------------


def _make_mcp_modules(tools: list[MagicMock] | None = None) -> tuple[MagicMock, MagicMock]:
    """Return (session_mock, tools_result_mock)."""
    tool_defs = tools or [_make_tool_def("tool1")]
    tools_result = MagicMock()
    tools_result.tools = tool_defs

    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=tools_result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session, tools_result


class TestConnectMCPServersSkip:
    async def test_no_command_or_url_skipped(self) -> None:
        registry = ToolRegistry()
        cfg = MCPServerConfig()  # no command, no url

        async with AsyncExitStack() as stack:
            await connect_mcp_servers({"empty": cfg}, registry, stack)

        assert len(registry.get_definitions()) == 0

    async def test_unknown_transport_skipped(self) -> None:
        registry = ToolRegistry()
        cfg = MCPServerConfig(type="ftp", url="ftp://example.com")

        async with AsyncExitStack() as stack:
            await connect_mcp_servers({"bad": cfg}, registry, stack)

        assert len(registry.get_definitions()) == 0


class TestConnectMCPServersStdio:
    async def test_stdio_registers_tools(self) -> None:
        registry = ToolRegistry()
        cfg = MCPServerConfig(command="npx", args=["-y", "server"])

        session, _ = _make_mcp_modules()

        with patch("exoclaw_tools_mcp.tool.stdio_client") as mock_transport, \
             patch("exoclaw_tools_mcp.tool.ClientSession") as MockSession, \
             patch("exoclaw_tools_mcp.tool.StdioServerParameters"):

            mock_rw = (MagicMock(), MagicMock())
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_rw)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_transport.return_value = mock_ctx

            MockSession.return_value = session

            async with AsyncExitStack() as stack:
                await connect_mcp_servers({"myserver": cfg}, registry, stack)

        assert len(registry.get_definitions()) == 1
        assert registry.get_definitions()[0]["function"]["name"] == "mcp_myserver_tool1"

    async def test_stdio_inferred_from_command(self) -> None:
        registry = ToolRegistry()
        cfg = MCPServerConfig(command="python", args=["server.py"])
        # type is None — should be inferred as stdio

        session, _ = _make_mcp_modules()

        with patch("exoclaw_tools_mcp.tool.stdio_client") as mock_transport, \
             patch("exoclaw_tools_mcp.tool.ClientSession") as MockSession, \
             patch("exoclaw_tools_mcp.tool.StdioServerParameters"):

            mock_rw = (MagicMock(), MagicMock())
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_rw)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_transport.return_value = mock_ctx
            MockSession.return_value = session

            async with AsyncExitStack() as stack:
                await connect_mcp_servers({"srv": cfg}, registry, stack)

        assert len(registry.get_definitions()) == 1


class TestConnectMCPServersSSE:
    async def test_sse_inferred_from_url(self) -> None:
        registry = ToolRegistry()
        cfg = MCPServerConfig(url="http://localhost:8080/sse")

        session, _ = _make_mcp_modules()

        with patch("exoclaw_tools_mcp.tool.sse_client") as mock_transport, \
             patch("exoclaw_tools_mcp.tool.ClientSession") as MockSession:

            mock_rw = (MagicMock(), MagicMock())
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_rw)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_transport.return_value = mock_ctx
            MockSession.return_value = session

            async with AsyncExitStack() as stack:
                await connect_mcp_servers({"sse_srv": cfg}, registry, stack)

        assert len(registry.get_definitions()) == 1


class TestConnectMCPServersStreamableHttp:
    async def test_streamable_http_from_url(self) -> None:
        registry = ToolRegistry()
        cfg = MCPServerConfig(url="http://localhost:8080/mcp")

        session, _ = _make_mcp_modules()

        with patch("exoclaw_tools_mcp.tool.streamable_http_client") as mock_transport, \
             patch("exoclaw_tools_mcp.tool.ClientSession") as MockSession, \
             patch("exoclaw_tools_mcp.tool.httpx.AsyncClient") as MockHttpx:

            mock_rw = (MagicMock(), MagicMock(), MagicMock())
            mock_http_ctx = MagicMock()
            mock_http_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_http_ctx.__aexit__ = AsyncMock(return_value=False)
            MockHttpx.return_value = mock_http_ctx

            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_rw)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_transport.return_value = mock_ctx

            MockSession.return_value = session

            async with AsyncExitStack() as stack:
                await connect_mcp_servers({"http_srv": cfg}, registry, stack)

        assert len(registry.get_definitions()) == 1

    async def test_explicit_streamable_http_type(self) -> None:
        registry = ToolRegistry()
        cfg = MCPServerConfig(url="http://localhost/mcp", type="streamableHttp")

        session, _ = _make_mcp_modules()

        with patch("exoclaw_tools_mcp.tool.streamable_http_client") as mock_transport, \
             patch("exoclaw_tools_mcp.tool.ClientSession") as MockSession, \
             patch("exoclaw_tools_mcp.tool.httpx.AsyncClient") as MockHttpx:

            mock_rw = (MagicMock(), MagicMock(), MagicMock())
            mock_http_ctx = MagicMock()
            mock_http_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_http_ctx.__aexit__ = AsyncMock(return_value=False)
            MockHttpx.return_value = mock_http_ctx

            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_rw)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_transport.return_value = mock_ctx

            MockSession.return_value = session

            async with AsyncExitStack() as stack:
                await connect_mcp_servers({"http2": cfg}, registry, stack)

        assert len(registry.get_definitions()) == 1


class TestConnectMCPServersErrorHandling:
    async def test_connection_failure_logged_continues(self) -> None:
        registry = ToolRegistry()
        good_cfg = MCPServerConfig(command="npx")
        bad_cfg = MCPServerConfig(command="bad")

        session, _ = _make_mcp_modules()
        call_count = 0

        def transport_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("connection refused")
            mock_rw = (MagicMock(), MagicMock())
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_rw)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        with patch("exoclaw_tools_mcp.tool.stdio_client", side_effect=transport_factory), \
             patch("exoclaw_tools_mcp.tool.ClientSession", return_value=session), \
             patch("exoclaw_tools_mcp.tool.StdioServerParameters"):

            async with AsyncExitStack() as stack:
                await connect_mcp_servers(
                    {"bad": bad_cfg, "good": good_cfg}, registry, stack
                )

        # good server still registered
        assert len(registry.get_definitions()) == 1

    async def test_multiple_servers_multiple_tools(self) -> None:
        registry = ToolRegistry()
        cfg1 = MCPServerConfig(command="srv1")
        cfg2 = MCPServerConfig(command="srv2")

        td1 = _make_tool_def("alpha")
        td2a = _make_tool_def("beta")
        td2b = _make_tool_def("gamma")

        session1, _ = _make_mcp_modules([td1])
        session2, _ = _make_mcp_modules([td2a, td2b])
        sessions = iter([session1, session2])

        def transport_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        def session_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
            return next(sessions)

        with patch("exoclaw_tools_mcp.tool.stdio_client", side_effect=transport_factory), \
             patch("exoclaw_tools_mcp.tool.ClientSession", side_effect=session_factory), \
             patch("exoclaw_tools_mcp.tool.StdioServerParameters"):

            async with AsyncExitStack() as stack:
                await connect_mcp_servers({"s1": cfg1, "s2": cfg2}, registry, stack)

        assert len(registry.get_definitions()) == 3
