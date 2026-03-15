"""Entry point for exoclaw skill discovery."""

from pathlib import Path


def spawn() -> dict[str, str]:
    """Return the spawn/subagent skill for agent context."""
    return {
        "name": "spawn",
        "content": (Path(__file__).parent / "SKILL.md").read_text(),
    }
