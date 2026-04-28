"""File system tools: read, write, edit, list — cross-runtime.

Same 4-tool surface (read_file, write_file, edit_file, list_dir) on
CPython and MicroPython. Chip-friendly compromises:

- ``_compat.Path`` instead of ``pathlib.Path`` (the MP shim provides
  ``read_text`` / ``write_text`` / ``exists`` / ``iterdir`` / ``mkdir``
  / ``relative_to`` — same surface as CPython for our tool needs).
- Sandbox via workspace-prefix + ``..`` segment reject + symlink-
  resolved ``relative_to`` check. ``Path.resolve()`` is called on
  CPython for symlink-escape protection (a symlink inside the
  workspace pointing to ``/etc/passwd`` resolves before the sandbox
  check fires). On MP the ``Path`` shim's ``resolve()`` is a no-op
  (chip filesystems don't have meaningful symlinks), so the call
  is cross-runtime safe.
- Smaller char caps on MP (32 KB read, 4× early-exit bound); CPython
  keeps 128 KB / 4× as before. Chip RAM doesn't grant the big budget.
- ``edit_file``'s "old_text not found" error uses a simple substring
  proximity hint instead of ``difflib.unified_diff``. ``difflib`` isn't
  in chip MP's frozen module set.
- ``open(..., encoding="utf-8")`` is CPython-only — MP's ``open()``
  doesn't accept the kwarg. Gated on ``IS_MICROPYTHON``.
- File size is read via ``os.stat`` directly (both runtimes have it
  on a string path) instead of ``Path.stat()`` (the MP shim doesn't
  implement ``stat`` on the ``Path`` object).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from exoclaw._compat import IS_MICROPYTHON, Path
from exoclaw.agent.tools.protocol import ToolBase

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _file_size(path: "str | Path") -> int:
    """Return file size in bytes, or ``-1`` if ``stat`` fails. Works
    on both runtimes — the MP ``Path`` shim doesn't implement
    ``Path.stat()`` but ``os.stat(str_path)`` does work on MP."""
    try:
        return os.stat(str(path))[6]
    except OSError:
        return -1


def _resolve_path(
    path: str, workspace: "Path | None" = None, allowed_dir: "Path | None" = None
) -> Path:
    """Resolve a user-supplied path against ``workspace`` (relative
    paths) and reject anything outside ``allowed_dir`` (defaults to
    ``workspace`` if unset).

    Two-layer sandbox:

    1. ``..`` segment reject — covers the obvious traversal attempt
       like ``../etc/passwd``. Done before any joining because
       ``Path("a/../b")`` is treated as ``b`` after resolve on
       CPython, which would silently bypass a naive ``relative_to``
       check on the un-resolved path.
    2. ``relative_to`` against the sandbox after a runtime-correct
       resolve. On CPython we call ``Path.resolve()`` so a symlink
       inside the workspace pointing to ``/etc/passwd`` is resolved
       to its target before the sandbox check fires (without this,
       the link itself sits inside the workspace and ``relative_to``
       passes — symlink-escape). On MicroPython the ``Path`` shim's
       ``resolve`` is a no-op (chip filesystems don't have meaningful
       symlinks, and the resolve helper isn't worth a syscall) so
       the segment-reject + ``relative_to`` does the work.
    """
    sandbox = allowed_dir or workspace
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        raise OSError("path may not contain '..' segments: {!r}".format(path))

    # Strip a leading workspace-name prefix the model sometimes
    # includes (e.g. ``.sim-workspace/screen.md`` when the
    # workspace IS ``.sim-workspace``). Without this, the join
    # double-nests the path. Common with models that see the
    # workspace name in the system prompt and prepend it.
    if workspace is not None:
        ws_str = str(workspace).rstrip("/")
        if path.startswith(ws_str + "/"):
            path = path[len(ws_str) + 1:]
        elif path == ws_str:
            path = "."

    p = Path(path)
    is_absolute = path.startswith("/")
    if not is_absolute and workspace is not None:
        p = workspace / path
    # Resolve symlinks on CPython BEFORE the sandbox check so a
    # symlink inside ``workspace`` pointing outside is rejected.
    # The MP shim's ``resolve`` is a no-op (returns self), so the
    # call is cross-runtime safe and the chip gets the cheap version.
    p = p.resolve()
    sandbox_resolved = sandbox.resolve() if sandbox is not None else None
    if sandbox_resolved is not None:
        try:
            p.relative_to(sandbox_resolved)
        except ValueError:
            raise OSError("path {!r} is outside sandbox {!r}".format(str(p), str(sandbox_resolved)))
    return p


# ── Per-runtime caps ────────────────────────────────────────────────
# Chip MP gets a smaller budget than CPython. The 4× early-exit
# bound is the "file too large, use offset/limit" threshold — scaled
# from the inline ``_MAX_CHARS`` cap by the same factor on both.
_MAX_CHARS_CPYTHON = 128_000
_MAX_CHARS_MP = 32_000
_MAX_CHARS = _MAX_CHARS_MP if IS_MICROPYTHON else _MAX_CHARS_CPYTHON


def _open_text(path_str: str, mode: str = "r") -> Any:
    """Open a text file with UTF-8. CPython needs ``encoding="utf-8"``
    explicitly; MP's ``open()`` doesn't accept the kwarg but defaults
    to bytes-as-utf-8 already on the unix port."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        return open(path_str, mode)
    return open(path_str, mode, encoding="utf-8")  # pragma: no cover (micropython)


class ReadFileTool(ToolBase):
    """Read file contents. Supports ranged reads (offset/limit) for
    large files.

    The class-level ``_MAX_CHARS`` is set per-runtime: 128 KB on
    CPython, 32 KB on MicroPython. Files larger than 4× this cap
    require offset/limit — the agent gets a hint message, not a
    silent truncation, so the model can pick a reasonable chunk size."""

    _MAX_CHARS = _MAX_CHARS

    def __init__(self, workspace: "Path | None" = None, allowed_dir: "Path | None" = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Use offset and limit to read a specific "
            "line range instead of the entire file (recommended for large files)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read"},
                "offset": {
                    "type": "integer",
                    "description": "Line number to start from (0-based, default 0)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self, path: str, offset: int = 0, limit: "int | None" = None, **kwargs: Any
    ) -> str:
        try:
            if offset < 0:
                return "Error: offset must be >= 0"
            if limit is not None and limit < 1:
                return "Error: limit must be >= 1"

            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return "Error: File not found: {}".format(path)
            if not file_path.is_file():
                return "Error: Not a file: {}".format(path)

            size = _file_size(file_path)

            # Early exit before allocation: if the caller didn't
            # narrow with offset/limit and the file is way over budget,
            # bounce them with a hint instead of buffering it.
            if offset == 0 and limit is None and size > self._MAX_CHARS * 4:
                return (
                    "Error: File too large ({} bytes). "
                    "Use offset and limit to read a portion, e.g. "
                    "read_file(path, offset=0, limit=50).".format(size)
                )

            # Ranged read — stream lines so the heap holds only the
            # selected window, not the whole file.
            if offset > 0 or limit is not None:
                selected: list[str] = []
                total_lines = 0
                end = offset + limit if limit is not None else None
                with _open_text(str(file_path)) as fh:
                    for i, line in enumerate(fh):
                        total_lines = i + 1
                        if i < offset:
                            continue
                        if end is not None and i >= end:
                            for _ in fh:
                                total_lines += 1
                            break
                        selected.append(line)

                actual_end = offset + len(selected)
                header = "[lines {}-{} of {}]\n".format(offset + 1, actual_end, total_lines)
                text = "".join(selected)
                if len(text) > self._MAX_CHARS:
                    text = text[: self._MAX_CHARS] + "\n... (truncated)"
                return header + text

            # Full file — within the inline cap.
            content = file_path.read_text(encoding="utf-8")
            if len(content) > self._MAX_CHARS:
                return (
                    content[: self._MAX_CHARS]
                    + "\n\n... (truncated — file is {} chars, showing first {}. "
                    "Use offset/limit to read more.)".format(len(content), self._MAX_CHARS)
                )
            return content
        except OSError as e:
            return "Error: {}".format(e)
        except Exception as e:
            return "Error reading file: {}".format(e)

    async def execute_streaming(
        self, path: str, offset: int = 0, limit: "int | None" = None, **kwargs: Any
    ) -> "AsyncIterator[str]":
        """Step D opt-in: stream the file's content from disk in
        fixed-character chunks, lifting the inline ``_MAX_CHARS`` cap.
        Only the full-file path streams; ranged reads fall back to
        the inline path because line-counting still needs full reads.
        """
        if offset < 0:
            yield "Error: offset must be >= 0"
            return
        if limit is not None and limit < 1:
            yield "Error: limit must be >= 1"
            return

        if offset > 0 or limit is not None:
            yield await self.execute(path, offset=offset, limit=limit, **kwargs)
            return

        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
        except OSError as e:
            yield "Error: {}".format(e)
            return
        if not file_path.exists():
            yield "Error: File not found: {}".format(path)
            return
        if not file_path.is_file():
            yield "Error: Not a file: {}".format(path)
            return

        try:
            with _open_text(str(file_path)) as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        return
                    yield chunk
        except OSError as e:
            yield "Error: {}".format(e)
        except Exception as e:
            yield "Error reading file: {}".format(e)


class WriteFileTool(ToolBase):
    """Write content to a file. Creates parent directories as needed."""

    def __init__(self, workspace: "Path | None" = None, allowed_dir: "Path | None" = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return "Successfully wrote {} bytes to {}".format(len(content), file_path)
        except OSError as e:
            return "Error: {}".format(e)
        except Exception as e:
            return "Error writing file: {}".format(e)


class EditFileTool(ToolBase):
    """Edit a file by replacing exact ``old_text`` with ``new_text``.
    The match must be unique — if ``old_text`` appears multiple times,
    the call is rejected so the agent can re-fetch with more context."""

    def __init__(self, workspace: "Path | None" = None, allowed_dir: "Path | None" = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_text with new_text. The old_text must exist "
            "exactly in the file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_text": {"type": "string", "description": "The text to replace with"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, path: str, old_text: str, new_text: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return "Error: File not found: {}".format(path)

            content = file_path.read_text(encoding="utf-8")

            if old_text not in content:
                return self._not_found_message(old_text, content, path)

            count = content.count(old_text)
            if count > 1:
                return (
                    "Warning: old_text appears {} times. Please provide more context "
                    "to make it unique.".format(count)
                )

            new_content = content.replace(old_text, new_text, 1)
            file_path.write_text(new_content, encoding="utf-8")
            return "Successfully edited {}".format(file_path)
        except OSError as e:
            return "Error: {}".format(e)
        except Exception as e:
            return "Error editing file: {}".format(e)

    @staticmethod
    def _not_found_message(old_text: str, content: str, path: str) -> str:
        """Build a helpful error when ``old_text`` isn't in ``content``.

        Uses simple character-prefix matching to find the longest
        prefix of ``old_text`` that DOES appear in ``content`` and
        echoes the first 200 chars of context around the partial
        match. ``difflib`` isn't in chip MP's frozen module set, so
        this is a runtime-portable replacement for the
        ``unified_diff`` view the CPython tool used to render. Same
        signal — "your old_text doesn't quite match, here's where I
        think it might've been intended" — at a fraction of the
        runtime + memory cost.
        """
        best_prefix_len = 0
        best_pos = -1
        # Walk down from the full ``old_text`` to its first character,
        # looking for the longest prefix that occurs in ``content``.
        # Cap the inner search at 64 chars — anything more granular is
        # unhelpful noise, and on a chip we don't want to scan a big
        # file 200 times.
        for length in range(min(len(old_text), 64), 0, -1):
            prefix = old_text[:length]
            pos = content.find(prefix)
            if pos != -1:
                best_prefix_len = length
                best_pos = pos
                break
        if best_pos == -1 or best_prefix_len < 3:
            return (
                "Error: old_text not found in {}. No similar text found. "
                "Verify the file content.".format(path)
            )
        # 100-char window before and after the partial match.
        ctx_start = max(0, best_pos - 100)
        ctx_end = min(len(content), best_pos + best_prefix_len + 100)
        line_no = content[:best_pos].count("\n") + 1
        snippet = content[ctx_start:ctx_end]
        return (
            "Error: old_text not found in {}. Closest match (first {} chars of "
            "old_text) at line {}:\n{}".format(path, best_prefix_len, line_no, snippet)
        )


class ListDirTool(ToolBase):
    """List directory contents."""

    def __init__(self, workspace: "Path | None" = None, allowed_dir: "Path | None" = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The directory path to list"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            dir_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not dir_path.exists():
                return "Error: Directory not found: {}".format(path)
            if not dir_path.is_dir():
                return "Error: Not a directory: {}".format(path)

            items = []
            # ``Path.iterdir`` is the cross-runtime listing API; the
            # MP shim wraps ``os.listdir``. Sort by name for stable
            # output across runs (CPython's ``iterdir`` order is
            # filesystem-defined; MP just returns ``listdir``'s list).
            for item in sorted(dir_path.iterdir(), key=lambda p: p.name):
                prefix = "[d] " if item.is_dir() else "[f] "
                items.append(prefix + item.name)

            if not items:
                return "Directory {} is empty".format(path)

            return "\n".join(items)
        except OSError as e:
            return "Error: {}".format(e)
        except Exception as e:
            return "Error listing directory: {}".format(e)
