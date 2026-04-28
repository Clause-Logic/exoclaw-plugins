"""Entry point for exoclaw skill discovery."""

from exoclaw._compat import Path


def web() -> dict[str, str]:
    """Return the web-tools skill payload. ``content``-only — no
    ``path`` (the bundler does ``shutil.copytree`` on ``path`` if
    set, which would copy ``html_to_markdown.py``'s 1700+ lines
    onto the chip's flash twice; same lesson as the workspace
    and screen packages)."""
    skill_dir = Path(__file__).parent
    return {
        "name": "web",
        "content": (skill_dir / "SKILL.md").read_text(),
    }
