"""Entry point for exoclaw skill discovery.

Returns the screen skill payload — just the SKILL.md content,
no ``path`` (we want the bundler to write only ``SKILL.md``,
not ``shutil.copytree`` the whole package onto the chip — same
lesson as the workspace package).
"""

from exoclaw._compat import Path


def screen() -> dict[str, str]:
    """Return the screen-control skill for agent context."""
    skill_dir = Path(__file__).parent
    return {
        "name": "screen",
        "content": (skill_dir / "SKILL.md").read_text(),
    }
