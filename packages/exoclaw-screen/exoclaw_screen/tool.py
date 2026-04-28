"""``RepaintScreenTool`` — argless tool the agent calls after
editing the screen-state file."""

from __future__ import annotations

from typing import Any

from exoclaw._compat import Path, get_logger
from exoclaw.agent.tools.protocol import ToolBase

from exoclaw_screen.protocol import Display

logger = get_logger()


class RepaintScreenTool(ToolBase):
    """Tool the agent calls to push the current ``screen.md`` to
    the panel.

    Usage pattern: agent edits ``screen.md`` with the standard
    file tools (``read_file`` / ``edit_file`` / ``write_file``)
    and then calls ``repaint_screen()``. No parameters — the file
    path is fixed at construction time from the ``Display``
    capabilities so the agent can't accidentally repaint a
    different file.
    """

    def __init__(self, display: Display) -> None:
        self._display = display
        # Resolve the screen-state file path once. Boards configure
        # this through ``DisplayCapabilities.screen_path`` — chip
        # boards typically pass an absolute SD-card path
        # (``/sd/exoclaw/screen.md``); host sims pass a workspace-
        # relative path.
        self._screen_path = Path(display.capabilities.screen_path)

    @property
    def name(self) -> str:
        return "repaint_screen"

    @property
    def description(self) -> str:
        return (
            "Re-render the current screen.md onto the device's display. "
            "Call this after editing screen.md to apply the change."
        )

    @property
    def parameters(self) -> "dict[str, Any]":
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        if not self._screen_path.exists():
            return "Error: screen file not found at {}. Write it first with write_file.".format(
                self._screen_path
            )
        try:
            md = self._screen_path.read_text(encoding="utf-8")
        except OSError as e:
            return "Error: failed to read {}: {}".format(self._screen_path, e)
        try:
            await self._display.show_markdown(md)
        except Exception as e:  # noqa: BLE001 — surface backend errors verbatim
            logger.error(
                "repaint_screen_failed",
                **{"screen.path": str(self._screen_path), "error": str(e)},
            )
            return "Error: display.show_markdown raised: {}".format(e)
        return "Repainted {} ({} chars)".format(self._screen_path, len(md))
