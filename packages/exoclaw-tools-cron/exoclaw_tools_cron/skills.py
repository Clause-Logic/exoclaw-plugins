"""Entry point for exoclaw skill discovery."""

from pathlib import Path


def cron() -> dict[str, str]:
    """Return the cron scheduling skill for agent context."""
    skill_dir = Path(__file__).parent
    return {
        "name": "cron",
        "content": (skill_dir / "SKILL.md").read_text(),
        "path": str(skill_dir),
    }
