"""Entry point for exoclaw skill discovery."""

from pathlib import Path


def etl() -> dict[str, str]:
    """Return the ETL skill for agent context."""
    return {
        "name": "etl",
        "content": (Path(__file__).parent / "SKILL.md").read_text(),
    }
