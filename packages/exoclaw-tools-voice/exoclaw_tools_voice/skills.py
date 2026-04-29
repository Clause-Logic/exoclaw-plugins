"""Entry point for exoclaw skill discovery."""

from exoclaw._compat import Path


def voice() -> dict[str, str]:
    """Return the voice-tools skill payload. Content-only — no
    ``path`` (the bundler does ``shutil.copytree`` on ``path``
    if set, which would copy the whole package onto chip flash
    twice; same lesson as workspace/screen/web)."""
    skill_dir = Path(__file__).parent
    return {
        "name": "voice",
        "content": (skill_dir / "SKILL.md").read_text(),
    }
