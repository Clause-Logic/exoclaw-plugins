"""Entry point for exoclaw skill discovery."""

from pathlib import Path


def spawn() -> dict[str, str]:
    """Return the spawn/subagent skill for agent context."""
    skill_dir = Path(__file__).parent
    return {
        "name": "spawn",
        "content": (skill_dir / "SKILL.md").read_text(),
        "path": str(skill_dir),
    }
