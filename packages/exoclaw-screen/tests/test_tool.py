"""``RepaintScreenTool`` integration test — drives a fake Display
and asserts the tool reads the file + calls ``show_markdown``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from exoclaw_screen.protocol import (
    COLOR_MONO,
    REFRESH_SLOW,
    DisplayCapabilities,
)
from exoclaw_screen.tool import RepaintScreenTool


class _FakeDisplay:
    """Minimal Display impl that records the markdown it received."""

    def __init__(self, screen_path: str) -> None:
        self.capabilities = DisplayCapabilities(
            width=800,
            height=480,
            color_mode=COLOR_MONO,
            refresh_class=REFRESH_SLOW,
            char_cols=80,
            char_rows=24,
            supports_partial=True,
            screen_path=screen_path,
        )
        self.last_markdown: str | None = None
        self.cleared = False
        self.fail_with: Exception | None = None

    async def show_markdown(self, markdown: str) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.last_markdown = markdown

    async def clear(self) -> None:
        self.cleared = True


@pytest.mark.asyncio
async def test_repaint_reads_file_and_pushes_to_display(tmp_path: Path) -> None:
    screen = tmp_path / "screen.md"
    screen.write_text("# Hello\n", encoding="utf-8")

    display = _FakeDisplay(str(screen))
    tool = RepaintScreenTool(display=display)

    result = await tool.execute()

    assert "Repainted" in result
    assert display.last_markdown == "# Hello\n"


@pytest.mark.asyncio
async def test_repaint_missing_file_returns_error(tmp_path: Path) -> None:
    screen = tmp_path / "screen.md"
    # Don't write — file doesn't exist.

    display = _FakeDisplay(str(screen))
    tool = RepaintScreenTool(display=display)

    result = await tool.execute()

    assert "Error" in result
    assert "screen file not found" in result
    assert display.last_markdown is None


@pytest.mark.asyncio
async def test_repaint_surfaces_display_errors(tmp_path: Path) -> None:
    screen = tmp_path / "screen.md"
    screen.write_text("# Hi\n", encoding="utf-8")

    display = _FakeDisplay(str(screen))
    display.fail_with = RuntimeError("driver locked up")
    tool = RepaintScreenTool(display=display)

    result = await tool.execute()
    assert "Error" in result
    assert "driver locked up" in result


def test_tool_metadata_shape() -> None:
    """The tool's ``name`` / ``description`` / ``parameters`` shape
    is what the LLM-side tool registration will see; lock the names
    so a rename is intentional."""
    display = _FakeDisplay("/tmp/screen.md")
    tool = RepaintScreenTool(display=display)
    assert tool.name == "repaint_screen"
    assert "screen.md" in tool.description.lower() or "screen" in tool.description.lower()
    params: dict[str, Any] = tool.parameters
    assert params["type"] == "object"
    assert params["required"] == []


def test_skill_entry_point_returns_dict() -> None:
    """``exoclaw_screen.skills.screen`` is the entry point that the
    firmware stage task consumes — same shape as the cron / subagent
    / workspace skill modules. ``path`` is deliberately absent so
    the bundler writes only ``SKILL.md``."""
    from exoclaw_screen.skills import screen

    skill = screen()
    assert isinstance(skill, dict)
    assert skill["name"] == "screen"
    assert "content" in skill
    assert skill["content"]
    # ``path`` MUST be absent — same lesson as the workspace package
    # (the bundler does ``shutil.copytree`` of ``path`` if set, which
    # would pull renderer/pillow.py + Pillow-dependent code onto the
    # chip).
    assert "path" not in skill
