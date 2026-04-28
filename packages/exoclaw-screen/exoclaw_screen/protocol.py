"""``Display`` Protocol seam — the boundary between
hardware-specific rendering and the cross-runtime markdown layout
engine.

Same shape as the other Protocols in the codebase (``Bus``,
``Channel``, ``Tool``, ``Provider``): runtime structurally checked,
boards implement it however their hardware demands. The agent
never imports a concrete display class; it only ever sees the
Protocol via dependency injection.

The capability descriptor (``DisplayCapabilities``) is part of the
Protocol because the SKILL.md template substitutes its values at
runtime — telling the agent its character budget, refresh class,
and whether colour is meaningful for this device. Same skill
template, different per-board values.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# ── Color modes recognised by the layout engine + renderers ──────
# String constants instead of an Enum because chip MicroPython's
# ``enum`` module is missing some surface; plain strings work
# everywhere with no shim.
COLOR_MONO = "mono"
"""1-bit black/white. Most e-ink panels."""
COLOR_GRAY2 = "gray2"
"""2-bit greyscale (4 levels)."""
COLOR_GRAY4 = "gray4"
"""4-bit greyscale (16 levels)."""
COLOR_RGB565 = "rgb565"
"""16-bit RGB. Most chip-driven LCDs."""
COLOR_RGB888 = "rgb888"
"""24-bit RGB. Host preview / browser targets."""


# ── Refresh classes (seconds, declarative bucket) ────────────────
# Layout engine uses this to decide whether partial-refresh tracking
# is worth doing — under ``REFRESH_FAST`` it isn't (just always full
# refresh), above ``REFRESH_SLOW`` it always is (skip if nothing
# changed, ghost-aware redraw cadence).
REFRESH_FAST = "fast"
"""LCDs, OLEDs — sub-second refresh, no ghost concern."""
REFRESH_MEDIUM = "medium"
"""Fast e-ink (partial update modes) — ~0.3s."""
REFRESH_SLOW = "slow"
"""Full e-ink refresh — 1–3s. UC8179 etc."""
REFRESH_VERY_SLOW = "very_slow"
"""Color e-ink (Spectra 6) — 10–20s. Avoid avoidable redraws."""


class DisplayCapabilities:
    """Per-board hardware facts that the layout engine and skill
    template consult.

    Fields are deliberately concrete (not flags / bitfields) so
    SKILL.md substitution is straightforward. Boards construct one
    of these alongside their ``Display`` impl and pass it through.
    """

    def __init__(
        self,
        width: int,
        height: int,
        color_mode: str,
        refresh_class: str,
        char_cols: int,
        char_rows: int,
        supports_partial: bool,
        screen_path: str = "screen.md",
    ) -> None:
        # Pixel resolution of the panel (or panel region exoclaw owns).
        self.width = width
        self.height = height
        # One of the ``COLOR_*`` constants above. Layout engine
        # quantises text colour against this — e.g. on ``"mono"`` a
        # ``{color=red}`` IAL just means "bold/inverse" and the
        # renderer ignores the actual hue.
        self.color_mode = color_mode
        # One of the ``REFRESH_*`` constants. Determines whether
        # partial-refresh tracking is engaged.
        self.refresh_class = refresh_class
        # Layout-engine character budget at the default body font.
        # SKILL.md exposes this so the agent self-truncates instead
        # of letting the engine truncate silently.
        self.char_cols = char_cols
        self.char_rows = char_rows
        # Hardware capability — boards that can't do partial refresh
        # set this to ``False`` and the layout engine never tries.
        self.supports_partial = supports_partial
        # Path to the screen-state file the agent edits. Defaults
        # to ``"screen.md"`` (relative to workspace); chip boards
        # typically pass an absolute SD-card path
        # (``/sd/exoclaw/screen.md``).
        self.screen_path = screen_path


@runtime_checkable
class Display(Protocol):
    """Cross-runtime display surface. Boards implement this against
    their actual driver; the agent only sees this Protocol.

    ``capabilities`` is a Protocol-level attribute (not a method)
    because the layout engine and SKILL.md substitution consult it
    eagerly at construction time. Boards must populate it before the
    Display is wired into ``RepaintScreenTool``.
    """

    capabilities: DisplayCapabilities

    async def show_markdown(self, markdown: str) -> None:
        """Replace the screen's content with the rendered markdown.

        Parser + layout engine run on the input string, the renderer
        emits framebuffer bytes, the driver pushes them. ``show``
        does not return until the refresh has been queued — actual
        e-ink panel update may happen asynchronously below the seam,
        but the caller can assume the next ``show_markdown`` call
        will produce a coherent layout (no half-rendered frames
        racing).
        """
        ...

    async def clear(self) -> None:
        """Blank the screen.

        Use sparingly on e-ink — most boards prefer "show empty
        markdown" over a full clear, since clearing tends to flash
        the panel through full-refresh waveforms unnecessarily.
        """
        ...
