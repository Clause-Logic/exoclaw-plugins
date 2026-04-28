"""Unix-port host-preview ``Display`` implementation.

The unix board runs under MicroPython's unix-port (so the chip-
relevant agent + provider + tool code paths get exercised
under MP semantics). MP unix-port can't load Pillow — Pillow is
a CPython C extension — so this Display delegates the actual
rasterisation to a CPython subprocess via ``os.system``.

Each ``show_markdown`` call:

1. Writes the agent's markdown to disk (already at
   ``capabilities.screen_path`` since the agent edited
   ``screen.md`` via the file tools).
2. Shells out to the CPython renderer CLI:
   ``$EXOCLAW_HOST_PYTHON -m exoclaw_screen.renderer.host_render``
3. The CLI imports Pillow, parses + lays out + renders, exits.

Result: ``repaint_screen`` produces a real PNG on the host while
the agent + tool dispatch all stay on MP unix-port — runtime
parity with the chip (no CPython-only code paths get exercised
that the chip wouldn't see).

``$EXOCLAW_HOST_PYTHON`` defaults to plain ``python3`` (must be
on PATH and have Pillow installed). The ``mise run sim`` task
sets it explicitly to the workspace's venv python so the user
doesn't have to install Pillow globally.
"""

from __future__ import annotations

import os

from exoclaw_screen.protocol import DisplayCapabilities


class HostPreviewDisplay:
    """Pillow-backed ``Display`` for the unix-port sim. Routes
    rasterisation through a CPython subprocess."""

    def __init__(
        self,
        capabilities: DisplayCapabilities,
        out_path: str,
        base_path: "str | None" = None,
    ) -> None:
        self.capabilities = capabilities
        self._out_path = out_path
        self._base_path = base_path

    async def show_markdown(self, markdown: str) -> None:
        # Write the markdown to disk first. ``capabilities.screen_path``
        # is the canonical agent-facing path (the agent edited it
        # via ``write_file``); we re-write it here so ``show_markdown``
        # works even if the caller passes ad-hoc markdown that
        # didn't come from the file. Idempotent — same bytes on the
        # round-trip case.
        md_path = self.capabilities.screen_path
        with open(md_path, "w") as f:
            f.write(markdown)

        host_python = os.getenv("EXOCLAW_HOST_PYTHON") or "python3"
        # All args are primitive scalars — quote each with double
        # quotes; no embedded ``"`` in any of them by construction
        # (paths come from the workspace, mode strings are fixed
        # constants from ``protocol``).
        argv = [
            host_python,
            # ``-P``: don't prepend the script's directory (or cwd
            # for ``-m``) to ``sys.path``. Critical here because the
            # MP unix-port sim's cwd is ``.stage/`` which contains
            # the MP-stub ``typing.py`` — without ``-P`` CPython
            # would pick the stub over its real stdlib ``typing``
            # and ImportError on ``from typing import final``.
            "-P",
            "-m",
            "exoclaw_screen.renderer.host_render",
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
        cmd = " ".join('"' + a + '"' for a in argv)
        rc = os.system(cmd)
        if rc != 0:
            # ``os.system`` returns the wait-status; non-zero means
            # the renderer failed. Surface it so ``RepaintScreenTool``
            # reports the error to the agent rather than silently
            # producing a stale PNG.
            raise RuntimeError("host_render exited rc={} (cmd: {})".format(rc, cmd))

    async def clear(self) -> None:
        # "Show empty markdown" — same path, just an empty doc.
        await self.show_markdown("")
