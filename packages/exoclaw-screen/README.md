# exoclaw-screen

File-backed display surface for [exoclaw](https://github.com/Clause-Logic/exoclaw).
The agent edits `screen.md` with the file tools it already has
(`exoclaw-tools-workspace`), then calls `repaint_screen()`. The
firmware reads the file, parses markdown + IAL + Pandoc fenced
divs, lays out boxes against the panel's resolution, and pushes
to whichever display backend the board has wired in.

The package itself is pure-Python and runs on both CPython and
MicroPython. Per-board renderers live in firmware `boards/`
directories and implement the `exoclaw_screen.Display` Protocol.

See `SKILL.md` for the agent-facing documentation.
