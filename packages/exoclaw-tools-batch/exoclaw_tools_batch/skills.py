"""Entry point for exoclaw skill discovery."""

from pathlib import Path


def etl() -> dict[str, str]:
    """Return the ETL skill for agent context."""
    skill_dir = Path(__file__).parent
    return {
        "name": "etl",
        "content": (skill_dir / "SKILL.md").read_text(),
        "path": str(skill_dir),
    }
