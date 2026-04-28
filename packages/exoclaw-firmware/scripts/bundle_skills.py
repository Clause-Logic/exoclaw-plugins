"""Bundle skills from selected pip packages into the staging tree.

Reads the firmware ``pyproject.toml`` for the explicit allow-list:

    [tool.exoclaw.firmware]
    bundle_skills_from = ["exoclaw-tools-cron"]

For each named package, walks the ``exoclaw.skills`` entry points
published by that package via ``importlib.metadata``, calls each
loader, and writes the resulting skill into ``<dest>/<skill-name>/``.

Each entry-point loader is expected to return a dict shaped like::

    {"name": "cron", "content": "<SKILL.md text>", "path": "/abs/path"}

``path`` is optional: if present, the entire skill directory is
copied (preserves ``hooks/``, supporting files, etc.); if absent,
only ``SKILL.md`` is written.

This script runs on the dev host (CPython, ``importlib.metadata``
fully supported), not on the chip — MP has no entry-point machinery.
The chip just sees the resulting flat directory tree at runtime."""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover (3.11+ project)
    import tomli as tomllib


def _read_bundle_list(pyproject: Path) -> list[str]:
    """Return ``[tool.exoclaw.firmware] bundle_skills_from``, or []."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    firmware = data.get("tool", {}).get("exoclaw", {}).get("firmware", {})
    raw = firmware.get("bundle_skills_from", [])
    if not isinstance(raw, list):
        raise ValueError(
            f"[tool.exoclaw.firmware] bundle_skills_from must be a list, got {type(raw).__name__}"
        )
    return [str(p) for p in raw]


def _entry_points_by_package(group: str) -> dict[str, list[importlib.metadata.EntryPoint]]:
    """Return ``{normalized_dist_name: [entry_points...]}`` for ``group``."""
    out: dict[str, list[importlib.metadata.EntryPoint]] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name is None:
            continue
        normalized = name.replace("-", "_").lower()
        for ep in dist.entry_points:
            if ep.group == group:
                out.setdefault(normalized, []).append(ep)
    return out


_SKILL_IGNORE = shutil.ignore_patterns(
    # CPython artefacts that shouldn't ship to the chip — they'd
    # both bloat the image and confuse MP, which has its own
    # ``.mpy`` compile cache. The stage task strips ``__pycache__``
    # from other staged trees too; we filter here at copy time so
    # the cleanup pass earlier in the stage script can't miss us.
    "__pycache__",
    "*.pyc",
    "*.pyo",
)


def _write_skill(dest_root: Path, payload: dict[str, Any]) -> Path:
    """Write one skill dict into ``dest_root/<name>/``.

    If the payload includes ``path``, the directory tree at that path
    is copied verbatim (so a package skill with ``hooks/`` /
    supporting files preserves its layout). Otherwise we just write
    the markdown content as ``SKILL.md``.

    ``__pycache__`` / ``.pyc`` are filtered on the directory copy
    path; for non-skill code that lives alongside the SKILL.md (like
    ``exoclaw-tools-cron``'s ``skills.py`` entry-point loader), the
    bytecode would just be dead weight on the chip."""
    name = payload["name"]
    content = payload["content"]
    target = dest_root / name
    src_path = payload.get("path")
    if src_path is not None and Path(src_path).is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src_path, target, ignore=_SKILL_IGNORE)
    else:
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(content, encoding="utf-8")
    return target


def bundle(pyproject: Path, dest_root: Path) -> list[str]:
    """Bundle skills from packages listed in ``pyproject`` into
    ``dest_root``. Returns the list of bundled skill names."""
    packages = _read_bundle_list(pyproject)
    if not packages:
        return []

    eps_by_pkg = _entry_points_by_package("exoclaw.skills")
    bundled: list[str] = []
    missing: list[str] = []

    for pkg in packages:
        normalized = pkg.replace("-", "_").lower()
        eps = eps_by_pkg.get(normalized, [])
        if not eps:
            missing.append(pkg)
            continue
        for ep in eps:
            loader = ep.load()
            payload = loader()
            if not isinstance(payload, dict) or "name" not in payload or "content" not in payload:
                raise ValueError(
                    f"entry point '{ep.name}' from '{pkg}' returned malformed payload "
                    "(expected dict with 'name' and 'content')"
                )
            _write_skill(dest_root, payload)
            bundled.append(payload["name"])

    if missing:
        # Fail loud — if a deployment claims to bundle skills from a
        # package that isn't installed, the resulting firmware would
        # silently lose those skills. Better to break the build.
        raise SystemExit(
            "bundle_skills: requested packages not installed (skills not discoverable): "
            + ", ".join(missing)
        )

    return bundled


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: bundle_skills.py <pyproject.toml> <dest-dir>", file=sys.stderr)
        return 2
    pyproject = Path(argv[1])
    dest_root = Path(argv[2])
    dest_root.mkdir(parents=True, exist_ok=True)
    bundled = bundle(pyproject, dest_root)
    if bundled:
        print(f"bundle_skills: wrote {len(bundled)} skill(s) to {dest_root}: {', '.join(bundled)}")
    else:
        print("bundle_skills: nothing to bundle (no [tool.exoclaw.firmware] bundle_skills_from)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
