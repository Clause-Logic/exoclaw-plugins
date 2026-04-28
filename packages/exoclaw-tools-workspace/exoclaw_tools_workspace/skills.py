"""Entry point for exoclaw skill discovery."""

from exoclaw._compat import Path


def workspace() -> dict[str, str]:
    """Return the workspace file-tools skill for agent context.

    Read by the firmware-stage host bundler via the
    ``[project.entry-points."exoclaw.skills"]`` line in
    ``pyproject.toml``. We deliberately don't return ``path`` —
    when the bundler sees a ``path`` it does ``shutil.copytree`` of
    the whole package, which here would include ``shell.py`` and
    ``web.py`` (CPython-only, deps not on chip). The
    ``content``-only payload makes the bundler write just
    ``SKILL.md`` into ``.stage/exoclaw_firmware/skills/workspace/``
    — exactly what the chip needs.
    """
    skill_dir = Path(__file__).parent
    return {
        "name": "workspace",
        "content": (skill_dir / "SKILL.md").read_text(),
    }
