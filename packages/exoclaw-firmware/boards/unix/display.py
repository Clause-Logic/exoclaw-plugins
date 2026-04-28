"""Unix-port host-preview ``Display`` implementation.

The unix board runs under MicroPython's unix-port. MP can't load
Pillow, so this Display delegates rasterisation to a CPython
subprocess (``render.py``).

HUD overlay (status pip + captions bar) is part of the Display
Protocol. ``set_status`` / ``set_caption`` store state and
trigger a re-render so the HUD updates immediately — same
semantics as a partial-refresh on e-ink. The renderer composites
them on top of the main canvas.
"""

from __future__ import annotations

import os

from exoclaw_screen.protocol import DisplayCapabilities


class HostPreviewDisplay:
    """Pillow-backed ``Display`` for the unix-port sim."""

    def __init__(
        self,
        capabilities: DisplayCapabilities,
        out_path: str,
        base_path: "str | None" = None,
    ) -> None:
        self.capabilities = capabilities
        self._out_path = out_path
        if base_path is None:
            sp = capabilities.screen_path
            if "/" in sp:
                base_path = sp.rsplit("/", 1)[0] or "."
            else:
                base_path = "."
        self._base_path = base_path
        self._status = ""
        self._caption = ""

    def set_status(self, status: str) -> None:
        self._status = status
        self._render()

    def set_caption(self, text: str) -> None:
        self._caption = text
        self._render()

    async def show_markdown(self, markdown: str) -> None:
        md_path = self.capabilities.screen_path
        with open(md_path, "w") as f:
            f.write(markdown)
        self._render()

    async def clear(self) -> None:
        await self.show_markdown("")

    def _render(self) -> None:
        """Shell out to render.py to produce the PNG."""
        md_path = self.capabilities.screen_path
        # Ensure the md file exists even if show_markdown hasn't
        # been called yet (HUD-only updates on a fresh boot).
        try:
            os.stat(md_path)
        except OSError:
            with open(md_path, "w") as f:
                f.write("")

        host_python = os.getenv("EXOCLAW_HOST_PYTHON") or "python3"
        argv = [
            host_python,
            "-P",
            "render.py",
            md_path,
            self._out_path,
            str(self.capabilities.width),
            str(self.capabilities.height),
            self.capabilities.color_mode,
            self.capabilities.refresh_class,
            str(self.capabilities.char_cols),
            str(self.capabilities.char_rows),
            "1" if self.capabilities.supports_partial else "0",
        ]
        if self._base_path:
            argv.append(self._base_path)
        # HUD args — passed after base_path. render.py reads
        # them positionally if present.
        argv.append(self._status or "")
        argv.append(self._caption or "")

        cmd = " ".join('"' + a + '"' for a in argv)
        rc = os.system(cmd)
        if rc != 0:
            raise RuntimeError(
                "render.py exited rc={} (cmd: {})".format(rc, cmd)
            )
