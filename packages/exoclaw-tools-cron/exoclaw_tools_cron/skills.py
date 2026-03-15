"""Entry point for exoclaw skill discovery."""

from pathlib import Path


def cron() -> dict[str, str]:
    """Return the cron scheduling skill for agent context."""
    return {
        "name": "cron",
        "content": (Path(__file__).parent / "SKILL.md").read_text(),
    }
