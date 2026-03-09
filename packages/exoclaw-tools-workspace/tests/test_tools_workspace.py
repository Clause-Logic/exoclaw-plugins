"""Tests for exoclaw-tools-workspace package."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw_tools_workspace.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    _resolve_path,
)
from exoclaw_tools_workspace.shell import ExecTool


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_absolute_path(self, tmp_path: Path) -> None:
        result = _resolve_path(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_relative_with_workspace(self, tmp_path: Path) -> None:
        result = _resolve_path("subdir/file.txt", workspace=tmp_path)
        assert result == (tmp_path / "subdir" / "file.txt").resolve()

    def test_allowed_dir_enforced(self, tmp_path: Path) -> None:
        other = tmp_path / "other"
        with pytest.raises(PermissionError):
            _resolve_path("/etc/passwd", allowed_dir=other)

    def test_allowed_dir_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        result = _resolve_path(str(f), allowed_dir=tmp_path)
        assert result == f.resolve()


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

class TestReadFileTool:
    async def test_read_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(str(f))
        assert result == "hello world"

    async def test_file_not_found(self, tmp_path: Path) -> None:
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(str(tmp_path / "missing.txt"))
        assert result.startswith("Error")

    async def test_not_a_file(self, tmp_path: Path) -> None:
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(str(tmp_path))
        assert result.startswith("Error")

    async def test_truncates_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("x" * (ReadFileTool._MAX_CHARS + 10))
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(str(f))
        assert "truncated" in result

    async def test_permission_error(self, tmp_path: Path) -> None:
        tool = ReadFileTool(workspace=tmp_path, allowed_dir=tmp_path / "sub")
        f = tmp_path / "test.txt"
        f.write_text("secret")
        result = await tool.execute(str(f))
        assert result.startswith("Error")

    async def test_relative_path_resolved(self, tmp_path: Path) -> None:
        f = tmp_path / "rel.txt"
        f.write_text("content")
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute("rel.txt")
        assert result == "content"

    def test_name_description_parameters(self) -> None:
        tool = ReadFileTool()
        assert tool.name == "read_file"
        assert "path" in tool.parameters["properties"]


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------

class TestWriteFileTool:
    async def test_write_file(self, tmp_path: Path) -> None:
        tool = WriteFileTool(workspace=tmp_path)
        result = await tool.execute("out.txt", "hello")
        assert "Successfully" in result
        assert (tmp_path / "out.txt").read_text() == "hello"

    async def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        tool = WriteFileTool(workspace=tmp_path)
        result = await tool.execute("a/b/c.txt", "nested")
        assert "Successfully" in result
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "nested"

    async def test_permission_error(self, tmp_path: Path) -> None:
        tool = WriteFileTool(workspace=tmp_path, allowed_dir=tmp_path / "sub")
        result = await tool.execute(str(tmp_path / "out.txt"), "x")
        assert result.startswith("Error")

    def test_name(self) -> None:
        assert WriteFileTool().name == "write_file"


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------

class TestEditFileTool:
    async def test_edit_file(self, tmp_path: Path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(str(f), "world", "earth")
        assert "Successfully" in result
        assert f.read_text() == "hello earth"

    async def test_file_not_found(self, tmp_path: Path) -> None:
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(str(tmp_path / "missing.txt"), "a", "b")
        assert result.startswith("Error")

    async def test_old_text_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(str(f), "xyz", "abc")
        assert result.startswith("Error")

    async def test_ambiguous_match(self, tmp_path: Path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("foo foo foo")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(str(f), "foo", "bar")
        assert "Warning" in result or "times" in result

    async def test_not_found_with_similar_text(self, tmp_path: Path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("hello world\ngoodbye world")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(str(f), "hello wrold", "hello earth")
        assert "Error" in result

    async def test_permission_error(self, tmp_path: Path) -> None:
        tool = EditFileTool(workspace=tmp_path, allowed_dir=tmp_path / "sub")
        result = await tool.execute(str(tmp_path / "f.txt"), "a", "b")
        assert result.startswith("Error")

    def test_name(self) -> None:
        assert EditFileTool().name == "edit_file"


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------

class TestListDirTool:
    async def test_list_dir(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(str(tmp_path))
        assert "file.txt" in result
        assert "subdir" in result

    async def test_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(str(empty))
        assert "empty" in result

    async def test_not_a_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x")
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(str(f))
        assert result.startswith("Error")

    async def test_dir_not_found(self, tmp_path: Path) -> None:
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(str(tmp_path / "missing"))
        assert result.startswith("Error")

    async def test_permission_error(self, tmp_path: Path) -> None:
        tool = ListDirTool(workspace=tmp_path, allowed_dir=tmp_path / "sub")
        result = await tool.execute(str(tmp_path))
        assert result.startswith("Error")

    def test_name(self) -> None:
        assert ListDirTool().name == "list_dir"


# ---------------------------------------------------------------------------
# ExecTool
# ---------------------------------------------------------------------------

class TestExecTool:
    async def test_basic_command(self) -> None:
        tool = ExecTool(timeout=5)
        result = await tool.execute("echo hello")
        assert "hello" in result

    async def test_stderr_included(self) -> None:
        tool = ExecTool(timeout=5)
        result = await tool.execute("echo err >&2")
        assert "err" in result

    async def test_nonzero_exit(self) -> None:
        tool = ExecTool(timeout=5)
        result = await tool.execute("exit 1")
        assert "Exit code" in result

    async def test_timeout(self) -> None:
        tool = ExecTool(timeout=1)
        result = await tool.execute("sleep 10")
        assert "timed out" in result

    async def test_denied_rm_rf(self) -> None:
        tool = ExecTool()
        result = await tool.execute("rm -rf /tmp/something")
        assert result.startswith("Error")
        assert "blocked" in result

    async def test_denied_fork_bomb(self) -> None:
        tool = ExecTool()
        result = await tool.execute(":(){ :|:& };:")
        assert result.startswith("Error")

    async def test_allow_patterns(self) -> None:
        tool = ExecTool(allow_patterns=[r"echo"])
        result = await tool.execute("echo hi")
        assert "hi" in result

    async def test_blocked_by_allowlist(self) -> None:
        tool = ExecTool(allow_patterns=[r"echo"])
        result = await tool.execute("ls /tmp")
        assert result.startswith("Error")

    async def test_working_dir_override(self, tmp_path: Path) -> None:
        tool = ExecTool(timeout=5)
        result = await tool.execute("pwd", working_dir=str(tmp_path))
        assert str(tmp_path) in result

    async def test_path_append(self) -> None:
        tool = ExecTool(timeout=5, path_append="/usr/local/bin")
        result = await tool.execute("echo ok")
        assert "ok" in result

    async def test_output_truncated(self) -> None:
        tool = ExecTool(timeout=5)
        result = await tool.execute("python3 -c \"print('x' * 20000)\"")
        assert "truncated" in result or len(result) <= 10100

    async def test_restrict_to_workspace_traversal(self, tmp_path: Path) -> None:
        tool = ExecTool(restrict_to_workspace=True, working_dir=str(tmp_path))
        result = await tool.execute("cat ../../../etc/passwd")
        assert result.startswith("Error")

    def test_name_description(self) -> None:
        tool = ExecTool()
        assert tool.name == "exec"
        assert "command" in tool.parameters["properties"]

    def test_extract_absolute_paths(self) -> None:
        paths = ExecTool._extract_absolute_paths("cat /etc/passwd | grep root")
        assert "/etc/passwd" in paths


# ---------------------------------------------------------------------------
# WebSearchTool / WebFetchTool
# ---------------------------------------------------------------------------

class TestWebSearchTool:
    async def test_no_api_key(self) -> None:
        from exoclaw_tools_workspace.web import WebSearchTool
        tool = WebSearchTool()
        result = await tool.execute("test query")
        assert "Error" in result or "BRAVE_API_KEY" in result

    async def test_search_via_model(self) -> None:
        from exoclaw_tools_workspace.web import WebSearchTool
        provider = MagicMock()
        resp = MagicMock()
        resp.content = "result1\nresult2"
        provider.chat = AsyncMock(return_value=resp)
        tool = WebSearchTool(provider=provider, search_model="gpt-4o")
        result = await tool.execute("python asyncio")
        assert "result" in result

    async def test_search_via_model_error(self) -> None:
        from exoclaw_tools_workspace.web import WebSearchTool
        provider = MagicMock()
        provider.chat = AsyncMock(side_effect=Exception("boom"))
        tool = WebSearchTool(provider=provider, search_model="gpt-4o")
        result = await tool.execute("query")
        assert "Error" in result

    async def test_brave_search_success(self) -> None:
        from exoclaw_tools_workspace.web import WebSearchTool
        import httpx
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"web": {"results": [
            {"title": "Test", "url": "https://example.com", "description": "desc"}
        ]}}
        mock_resp.raise_for_status = MagicMock()

        tool = WebSearchTool(api_key="test-key")
        with patch("exoclaw_tools_workspace.web.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)
            result = await tool.execute("test")
        assert "Test" in result

    async def test_brave_no_results(self) -> None:
        from exoclaw_tools_workspace.web import WebSearchTool
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"web": {"results": []}}
        mock_resp.raise_for_status = MagicMock()

        tool = WebSearchTool(api_key="test-key")
        with patch("exoclaw_tools_workspace.web.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)
            result = await tool.execute("test")
        assert "No results" in result

    def test_name_description(self) -> None:
        from exoclaw_tools_workspace.web import WebSearchTool
        t = WebSearchTool()
        assert t.name == "web_search"

    def test_api_key_from_env(self, monkeypatch: Any) -> None:
        from exoclaw_tools_workspace.web import WebSearchTool
        monkeypatch.setenv("BRAVE_API_KEY", "env-key")
        t = WebSearchTool()
        assert t.api_key == "env-key"


class TestWebFetchTool:
    async def test_invalid_url_scheme(self) -> None:
        from exoclaw_tools_workspace.web import WebFetchTool
        tool = WebFetchTool()
        result = await tool.execute("ftp://example.com")
        data = json.loads(result)
        assert "error" in data

    async def test_invalid_url_no_domain(self) -> None:
        from exoclaw_tools_workspace.web import WebFetchTool
        tool = WebFetchTool()
        result = await tool.execute("http://")
        data = json.loads(result)
        assert "error" in data

    async def test_fetch_html(self) -> None:
        from exoclaw_tools_workspace.web import WebFetchTool
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        mock_resp.url = "https://example.com"
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        tool = WebFetchTool()
        with patch("exoclaw_tools_workspace.web.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)
            result = await tool.execute("https://example.com")
        data = json.loads(result)
        assert data["status"] == 200

    async def test_fetch_json(self) -> None:
        from exoclaw_tools_workspace.web import WebFetchTool
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"key": "value"}
        mock_resp.text = '{"key": "value"}'
        mock_resp.url = "https://api.example.com/data"
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        tool = WebFetchTool()
        with patch("exoclaw_tools_workspace.web.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)
            result = await tool.execute("https://api.example.com/data")
        data = json.loads(result)
        assert data["extractor"] == "json"

    async def test_fetch_error(self) -> None:
        from exoclaw_tools_workspace.web import WebFetchTool
        import httpx
        tool = WebFetchTool()
        with patch("exoclaw_tools_workspace.web.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(side_effect=Exception("connection error"))
            result = await tool.execute("https://example.com")
        data = json.loads(result)
        assert "error" in data

    async def test_truncates_long_content(self) -> None:
        from exoclaw_tools_workspace.web import WebFetchTool
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = "x" * 100000
        mock_resp.url = "https://example.com"
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        tool = WebFetchTool(max_chars=100)
        with patch("exoclaw_tools_workspace.web.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)
            result = await tool.execute("https://example.com")
        data = json.loads(result)
        assert data["truncated"] is True

    def test_name(self) -> None:
        from exoclaw_tools_workspace.web import WebFetchTool
        assert WebFetchTool().name == "web_fetch"


# ---------------------------------------------------------------------------
# Additional coverage: filesystem edge cases, shell workspace restriction
# ---------------------------------------------------------------------------

class TestResolvepathPermissionError:
    def test_permission_error_from_resolve(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError):
            _resolve_path("/etc/shadow", allowed_dir=tmp_path)


class TestEditFileToolExtra:
    async def test_not_found_with_similar_match(self, tmp_path: Path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("hello world\ngoodbye world\nhello again")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(str(f), "hello wrold", "hello earth")
        assert "Error" in result
        assert "similar" in result.lower() or "%" in result

    async def test_not_found_no_similar(self, tmp_path: Path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(str(f), "xyzxyzxyz completely different text", "replacement")
        assert "Error" in result
        assert "No similar" in result

    async def test_general_exception(self, tmp_path: Path) -> None:
        tool = EditFileTool(workspace=tmp_path)
        with patch("pathlib.Path.read_text", side_effect=OSError("disk error")):
            result = await tool.execute(str(tmp_path / "f.txt"), "a", "b")
        assert "Error" in result


class TestExecToolExtra:
    async def test_restrict_to_workspace_absolute_outside(self, tmp_path: Path) -> None:
        tool = ExecTool(restrict_to_workspace=True, working_dir=str(tmp_path))
        result = await tool.execute(f"cat /etc/passwd")
        assert result.startswith("Error")
        assert "outside" in result

    async def test_restrict_to_workspace_absolute_inside(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.txt"
        f.write_text("ok")
        tool = ExecTool(restrict_to_workspace=True, working_dir=str(tmp_path), timeout=5)
        result = await tool.execute(f"cat {f}")
        assert "ok" in result

    async def test_command_exception(self) -> None:
        tool = ExecTool(timeout=5)
        with patch("asyncio.create_subprocess_shell", side_effect=OSError("no shell")):
            result = await tool.execute("echo hi")
        assert "Error" in result


class TestWebSearchToolExtra:
    async def test_no_web_key_in_response(self) -> None:
        from exoclaw_tools_workspace.web import WebSearchTool
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        tool = WebSearchTool(api_key="test-key")
        with patch("exoclaw_tools_workspace.web.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)
            result = await tool.execute("test")
        assert "No results" in result


class TestWebFetchToolExtra:
    async def test_fetch_raw_text(self) -> None:
        from exoclaw_tools_workspace.web import WebFetchTool
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = "raw content here"
        mock_resp.url = "https://example.com/file.txt"
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        tool = WebFetchTool()
        with patch("exoclaw_tools_workspace.web.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)
            result = await tool.execute("https://example.com/file.txt")
        data = json.loads(result)
        assert data["status"] == 200


# ---------------------------------------------------------------------------
# Tool property coverage and exception paths
# ---------------------------------------------------------------------------

class TestToolProperties:
    def test_read_file_description_and_params(self) -> None:
        t = ReadFileTool()
        assert "file" in t.description.lower()
        assert "path" in t.parameters["properties"]

    def test_write_file_description_and_params(self) -> None:
        t = WriteFileTool()
        assert "write" in t.description.lower()
        assert "content" in t.parameters["properties"]

    def test_edit_file_description_and_params(self) -> None:
        t = EditFileTool()
        assert "edit" in t.description.lower()
        assert "old_text" in t.parameters["properties"]

    def test_list_dir_description_and_params(self) -> None:
        t = ListDirTool()
        assert "list" in t.description.lower()
        assert "path" in t.parameters["properties"]

    def test_exec_tool_description_and_params(self) -> None:
        t = ExecTool()
        assert "command" in t.description.lower()
        assert "command" in t.parameters["properties"]


class TestReadFileToolExtra:
    async def test_file_too_large(self, tmp_path: Path) -> None:
        f = tmp_path / "huge.bin"
        # Write more than MAX_CHARS * 4 bytes
        f.write_bytes(b"x" * (ReadFileTool._MAX_CHARS * 4 + 1))
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(str(f))
        assert "too large" in result

    async def test_general_exception(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("content")
        tool = ReadFileTool(workspace=tmp_path)
        with patch.object(Path, "read_text", side_effect=OSError("disk error")):
            result = await tool.execute(str(f))
        assert "Error" in result


class TestWriteFileToolExtra:
    async def test_general_exception(self, tmp_path: Path) -> None:
        tool = WriteFileTool(workspace=tmp_path)
        with patch.object(Path, "write_text", side_effect=OSError("disk error")):
            result = await tool.execute(str(tmp_path / "out.txt"), "content")
        assert "Error" in result


class TestEditFileToolNotFound:
    async def test_not_found_message_similar_multiline(self, tmp_path: Path) -> None:
        """_not_found_message: best_ratio > 0.5 branch (multi-line old_text with 2/3 matching lines)"""
        f = tmp_path / "edit.txt"
        f.write_text("line1\nline2\nline3")
        tool = EditFileTool(workspace=tmp_path)
        # 2 of 3 lines match exactly — ratio = 0.67 > 0.5
        result = await tool.execute(str(f), "line1\nline2\nchanged_line3", "replacement")
        assert "Error" in result
        assert "%" in result  # shows similarity percentage

    async def test_general_exception(self, tmp_path: Path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        tool = EditFileTool(workspace=tmp_path)
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = await tool.execute(str(f), "hello world", "replacement")
        assert "Error" in result


class TestListDirToolExtra:
    async def test_general_exception(self, tmp_path: Path) -> None:
        tool = ListDirTool(workspace=tmp_path)
        with patch.object(Path, "iterdir", side_effect=OSError("permission denied")):
            result = await tool.execute(str(tmp_path))
        assert "Error" in result
