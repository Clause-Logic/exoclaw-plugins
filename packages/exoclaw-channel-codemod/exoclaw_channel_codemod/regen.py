"""Regenerate a channel package's `channel.py` + `tests/test_channel.py`
from `vendor/upstream*.py` + `patches/*.patch`.

Pure Python so it can be invoked from:
  - conftest.py (so `pytest packages/exoclaw-channel-X/tests/` works from source)
  - hatch build hook (so the wheel ships the materialized output)
  - sync.sh (the maintainer-side upstream-sync workflow)

Idempotent — only writes when content changes, so editor file-watchers and
``ruff format --check`` don't churn.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

from . import codemod as _codemod


def _read_sha(pkg_dir: Path) -> str:
    sha_file = pkg_dir / "vendor" / "SHA"
    return sha_file.read_text().strip() if sha_file.exists() else "unknown"


def _channel_name(pkg_dir: Path) -> str:
    """Derive the channel name from the package's pyproject.toml.

    Reads `[project].name` (always `exoclaw-channel-<name>`) instead of the
    directory name — directory naming differs between source layout
    (`exoclaw-channel-slack`) and sdist build temp dirs
    (`exoclaw_channel_slack-0.1.0`).
    """
    pyproject = pkg_dir / "pyproject.toml"
    if pyproject.is_file():
        meta = tomllib.loads(pyproject.read_text())
        name = meta.get("project", {}).get("name", "")
        if name.startswith("exoclaw-channel-"):
            return name.removeprefix("exoclaw-channel-")
    # Fallback for ad-hoc invocations
    return pkg_dir.name.removeprefix("exoclaw-channel-").removeprefix("exoclaw_channel_")


def _write_if_changed(path: Path, content: str) -> bool:
    """Write only if content differs. Returns True if written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == content:
        return False
    path.write_text(content)
    return True


def _apply_patches(target: Path, patch_glob: str, pkg_dir: Path) -> int:
    """Apply matching patches in lexical order via `patch -p1`. Returns count."""
    patches_dir = pkg_dir / "patches"
    if not patches_dir.is_dir():
        return 0
    count = 0
    for patch in sorted(patches_dir.glob(patch_glob)):
        result = subprocess.run(
            ["patch", "--silent", "--no-backup-if-mismatch", "-p1", "-i", str(patch)],
            cwd=target.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"patch {patch.name} failed to apply against {target.name}.\n"
                f"Upstream code likely moved; regenerate this patch against the "
                f"new codemod output.\n"
                f"stderr:\n{result.stderr}"
            )
        count += 1
    return count


def regenerate(pkg_dir: Path) -> dict[str, int]:
    """Regenerate channel.py (+ test_channel.py if upstream test exists).

    Returns ``{"source_patches": N, "test_patches": M}``.
    """
    pkg_dir = pkg_dir.resolve()
    name = _channel_name(pkg_dir)
    sha = _read_sha(pkg_dir)
    vendor = pkg_dir / "vendor"
    upstream_src = vendor / "upstream.py"
    upstream_test = vendor / "upstream_test.py"
    chan_out = pkg_dir / f"exoclaw_channel_{name}" / "channel.py"
    test_out = pkg_dir / "tests" / "test_channel.py"

    if not upstream_src.exists():
        raise FileNotFoundError(f"missing vendor/upstream.py in {pkg_dir}")

    src, src_warnings = _codemod.transform_source(upstream_src.read_text(), sha, name)
    _write_if_changed(chan_out, src)
    src_patches = _apply_patches(chan_out, "*-source-*.patch", pkg_dir)

    test_patches = 0
    if upstream_test.exists():
        pkg = f"exoclaw_channel_{name}"
        tsrc, test_warnings = _codemod.transform_test(upstream_test.read_text(), sha, name, pkg)
        _write_if_changed(test_out, tsrc)
        test_patches = _apply_patches(test_out, "*-test-*.patch", pkg_dir)

    return {"source_patches": src_patches, "test_patches": test_patches}


def regenerate_all_channels(repo_root: Path | None = None) -> None:
    """Regenerate every ``packages/exoclaw-channel-*/`` package whose
    ``vendor/upstream.py`` exists. No-op for packages without vendor (the
    hand-written cli/heartbeat/pipe channels)."""
    if repo_root is None:
        # Walk up from this file: exoclaw_channel_codemod/regen.py →
        # packages/exoclaw-channel-codemod/exoclaw_channel_codemod/ →
        # packages/exoclaw-channel-codemod/ → packages/ → repo root.
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
    for pkg in sorted((repo_root / "packages").glob("exoclaw-channel-*")):
        if not (pkg / "vendor" / "upstream.py").exists():
            continue
        regenerate(pkg)
