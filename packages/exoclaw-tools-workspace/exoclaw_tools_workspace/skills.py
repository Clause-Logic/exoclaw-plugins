"""Entry point for exoclaw skill discovery."""

from exoclaw._compat import Path


def workspace() -> dict[str, str]:
    """Return the workspace file-tools skill for agent context.

    Read by the firmware-stage host bundler via the
    ``[project.entry-points."exoclaw.skills"]`` line in
    ``pyproject.toml``. The bundler copies ``SKILL.md`` plus this
    package's directory into ``.stage/exoclaw_firmware/skills/``
    so the chip's ``SkillsLoader`` finds the skill file at runtime.
    """
    skill_dir = Path(__file__).parent
    return {
        "name": "workspace",
        "content": (skill_dir / "SKILL.md").read_text(),
        "path": str(skill_dir),
    }
