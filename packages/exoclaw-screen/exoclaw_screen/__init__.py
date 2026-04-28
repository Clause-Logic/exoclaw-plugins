"""File-backed display surface for exoclaw.

The agent edits ``screen.md`` with the standard file tools; the
firmware reads the file via ``RepaintScreenTool``, parses
markdown + IAL + Pandoc fenced divs, lays out boxes for the
panel's resolution, and pushes to whichever backend the board
implements behind the ``Display`` Protocol.

See ``SKILL.md`` for the agent-facing grammar reference."""

from exoclaw_screen.protocol import (
    COLOR_GRAY2,
    COLOR_GRAY4,
    COLOR_MONO,
    COLOR_RGB565,
    COLOR_RGB888,
    REFRESH_FAST,
    REFRESH_MEDIUM,
    REFRESH_SLOW,
    REFRESH_VERY_SLOW,
    Display,
    DisplayCapabilities,
)
from exoclaw_screen.tool import RepaintScreenTool

__all__ = [
    "COLOR_GRAY2",
    "COLOR_GRAY4",
    "COLOR_MONO",
    "COLOR_RGB565",
    "COLOR_RGB888",
    "Display",
    "DisplayCapabilities",
    "REFRESH_FAST",
    "REFRESH_MEDIUM",
    "REFRESH_SLOW",
    "REFRESH_VERY_SLOW",
    "RepaintScreenTool",
]
