"""Render backends for ``LayoutBlock`` lists.

Per-platform renderers consume the same flat block list shape
produced by ``layout.lay_out()``. Concrete backends:

- ``pillow.PillowRenderer`` — CPython host preview, writes a PNG
  next to ``screen.md`` so devs can see what the panel would
  show. Gated on Pillow being installed (``pip install
  exoclaw-screen[host-preview]``).

Boards ship their own chip-side renderers under
``exoclaw-firmware/boards/<board>/display.py`` — they implement
the ``Display`` Protocol from ``exoclaw_screen.protocol`` and
typically translate the layout-block list into LVGL widgets +
SPI commands.

This package's renderer namespace is deliberately small — the
agent and tool surface don't depend on it; it's a host-developer
convenience for prototyping screen layouts before chip hardware
is in hand."""
