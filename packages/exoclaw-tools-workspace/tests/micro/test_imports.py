"""MicroPython import + round-trip smoke test for ``exoclaw-tools-workspace``.

Pure-Python — no pytest. Driven by the workspace's
``mise run test-micro`` task on a coverage-variant MicroPython
binary. Exercises the runtime-divergent paths in the chip-relevant
``filesystem`` module: ``_compat.Path`` integration, ``os.stat``-based
size lookup, ``open()`` without ``encoding=`` kwarg on MP, and the
``difflib``-free edit-not-found error path.

The full agent-loop integration test (``read_file`` → ``edit_file`` →
verify) lives in the CPython suite; this gate is just "does it
import cleanly + can each tool exec without crashing on MP."
"""

import asyncio
import os


def test_top_level_imports():
    from exoclaw_tools_workspace import (
        EditFileTool,
        ListDirTool,
        ReadFileTool,
        WriteFileTool,
    )

    assert callable(ReadFileTool)
    assert callable(WriteFileTool)
    assert callable(EditFileTool)
    assert callable(ListDirTool)


def test_skill_entry_point_returns_dict():
    """``exoclaw_tools_workspace.skills.workspace`` is the entry
    point that the firmware stage task consumes. Reads ``SKILL.md``
    adjacent to the package, which on MP must use the ``Path`` shim.

    Deliberately does NOT include ``path`` in the payload — the
    bundler does ``shutil.copytree`` of the whole package when
    ``path`` is set, which here would pull ``shell.py`` /
    ``web.py`` (CPython-only deps) onto the chip's flash."""
    from exoclaw_tools_workspace.skills import workspace

    skill = workspace()
    assert isinstance(skill, dict)
    assert skill["name"] == "workspace"
    assert "content" in skill
    assert skill["content"]
    # ``path`` MUST be absent so the bundler writes only ``SKILL.md``.
    assert "path" not in skill


def _scratch_dir():
    """Pick a writable scratch root for both unix-port and chip-MP.
    The unix-port respects ``$TMPDIR``; chip ports usually only have
    ``/`` and ``/lib``. ``mise run sim`` runs from ``.stage`` so use
    that as the parent."""
    return os.getenv("TMPDIR") or "."


def _rm_tree(path):
    """Recursively delete ``path``. ``shutil.rmtree`` isn't on chip
    MP, so we walk via ``os.listdir`` + ``os.stat``. Best-effort —
    swallow ``OSError`` so a stray symlink or permission glitch
    doesn't crash the test suite cleanup."""
    try:
        path_str = str(path)
        try:
            entries = os.listdir(path_str)
        except OSError:
            return
        for name in entries:
            child = path_str + "/" + name
            try:
                mode = os.stat(child)[0]
            except OSError:
                continue
            if mode & 0o040000:  # directory
                _rm_tree(child)
            else:
                try:
                    os.remove(child)
                except OSError:
                    pass
        try:
            os.rmdir(path_str)
        except OSError:
            pass
    except OSError:
        pass


def test_read_write_edit_round_trip():
    """End-to-end: write a file, read it back, edit it, verify the
    change. Exercises ``_resolve_path`` workspace-relative resolution,
    ``Path.write_text`` / ``read_text`` on the MP shim, and
    ``EditFileTool``'s exact-match find/replace.

    Cleans up the per-run scratch directory in ``finally`` so on
    chip the SD card doesn't accumulate ``ws-test-*`` directories
    across reboots.
    """
    from exoclaw._compat import Path
    from exoclaw_tools_workspace import EditFileTool, ReadFileTool, WriteFileTool

    workspace = Path(_scratch_dir()) / "ws-test-{}".format(os.urandom(2).hex())
    workspace.mkdir(parents=True, exist_ok=True)

    write = WriteFileTool(workspace=workspace)
    read = ReadFileTool(workspace=workspace)
    edit = EditFileTool(workspace=workspace)

    async def _run():
        # Write — relative path joined with workspace.
        wrote = await write.execute("notes.md", "hello world")
        assert "Successfully wrote" in wrote, wrote

        # Read back full file.
        got = await read.execute("notes.md")
        assert got == "hello world", repr(got)

        # Edit — exact-match find/replace.
        edited = await edit.execute("notes.md", "world", "chip")
        assert "Successfully edited" in edited, edited

        # Read back to confirm.
        got2 = await read.execute("notes.md")
        assert got2 == "hello chip", repr(got2)

    try:
        asyncio.run(_run())
    finally:
        _rm_tree(workspace)


def test_resolve_path_rejects_traversal():
    """``..`` segments are rejected before joining. The MP shim's
    ``Path.resolve`` is a no-op so without the explicit segment
    check, ``workspace / "../etc/passwd"`` would land outside the
    sandbox unnoticed."""
    from exoclaw._compat import Path
    from exoclaw_tools_workspace.filesystem import _resolve_path

    workspace = Path("/tmp/ws-traversal-test")
    try:
        _resolve_path("../etc/passwd", workspace=workspace)
    except OSError:
        return
    raise AssertionError("expected OSError for '..' segment")


def test_edit_file_not_found_returns_proximity_hint():
    """When ``old_text`` doesn't match exactly, the error message
    contains a longest-prefix proximity hint (chip-friendly
    replacement for the ``difflib.unified_diff`` view used on
    CPython before this rewrite). The ``difflib`` module isn't in
    chip MP's frozen module set, so this is the hot-path behaviour
    on chip.

    Cleans up the per-run scratch directory in ``finally`` so on
    chip the SD card doesn't accumulate ``ws-noterr-*`` directories
    across reboots.
    """
    from exoclaw._compat import Path
    from exoclaw_tools_workspace import EditFileTool, WriteFileTool

    workspace = Path(_scratch_dir()) / "ws-noterr-{}".format(os.urandom(2).hex())
    workspace.mkdir(parents=True, exist_ok=True)

    write = WriteFileTool(workspace=workspace)
    edit = EditFileTool(workspace=workspace)

    async def _run():
        await write.execute("doc.md", "the quick brown fox jumps over the lazy dog")
        # ``old_text`` close-but-no-cigar — first 8 chars match,
        # rest doesn't. The error should pick up the prefix and
        # point at line 1.
        out = await edit.execute("doc.md", "the quick GREEN fox", "the slow")
        assert "old_text not found" in out, out
        # Some hint about the closest match should land in the message.
        assert "Closest match" in out, out

    try:
        asyncio.run(_run())
    finally:
        _rm_tree(workspace)
