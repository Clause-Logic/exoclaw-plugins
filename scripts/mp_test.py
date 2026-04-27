#!/usr/bin/env python3
"""MicroPython test driver for the plugins workspace.

Walks ``packages/*/pyproject.toml`` for ``[tool.exoclaw] mp_compat
= true`` markers, stages those packages alongside core ``exoclaw``
into ``.mp-stage/``, and runs core's MP test runner against each
package's ``tests/micro/`` directory.

Subcommands:

- ``list``         — print the list of MP-compat package names.
- ``stage``        — assemble ``.mp-stage/`` (core + all MP-compat
                     packages + MP stubs from core).
- ``test <pkg>``   — stage + run MP tests for one package.
- ``test-all``     — stage + run MP tests for every MP-compat package.

Locates the core ``exoclaw`` source via ``EXOCLAW_CORE_DIR`` env
var (defaults to ``../exoclaw`` from the workspace root). The MP
binary path comes from ``EXOCLAW_MICROPYTHON_BIN``; CI builds it
from source (see core's ``pr.yml``), local dev sets it in
``mise.local.toml``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
PACKAGES_DIR = WORKSPACE / "packages"
STAGE_DIR = WORKSPACE / ".mp-stage"


def _core_dir() -> Path:
    """Locate core exoclaw — env override or sibling ``../exoclaw``."""
    override = os.environ.get("EXOCLAW_CORE_DIR")
    if override:
        return Path(override).resolve()
    return (WORKSPACE / ".." / "exoclaw").resolve()


def _mp_binary() -> str:
    """Path to a coverage-variant MicroPython binary. Required for
    settrace-based coverage; the brew bottle doesn't support it.
    See core's ``pr.yml`` for the build recipe."""
    binary = os.environ.get("EXOCLAW_MICROPYTHON_BIN")
    if not binary:
        sys.exit(
            "EXOCLAW_MICROPYTHON_BIN not set — point it at a coverage-variant\n"
            "MicroPython binary (see core's tests/test_micropython_runner.py)."
        )
    return binary


def _mp_compat_packages() -> list[tuple[str, dict]]:
    """Return ``[(package_name, exoclaw_meta), ...]`` for every
    package whose pyproject opts into MP CI."""
    out: list[tuple[str, dict]] = []
    for pkg_dir in sorted(PACKAGES_DIR.iterdir()):
        cfg = pkg_dir / "pyproject.toml"
        if not cfg.is_file():
            continue
        meta = tomllib.loads(cfg.read_text())
        exo = meta.get("tool", {}).get("exoclaw", {})
        if exo.get("mp_compat"):
            out.append((pkg_dir.name, exo))
    return out


def _stage(packages: list[str]) -> Path:
    """Build ``.mp-stage/`` containing core + the named packages +
    MP stubs. Caller passes the list returned by
    ``_mp_compat_packages()`` filtered to whatever they want
    tested in this run."""
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir()

    core = _core_dir()

    def _ignore_cpython_only(_dir: str, names: list[str]) -> list[str]:
        # Strip CPython-only impls (``_cpython.py`` siblings) and
        # ``__pycache__`` directories at copy time. ``_mp_lib/``
        # holds runtime fillers that get flattened to the stage
        # root below — exclude the in-package copy so plain
        # ``import typing`` resolves there.
        return [
            n
            for n in names
            if n in ("__pycache__", "_cpython.py", "_mp_lib")
        ]

    # Core source → ``.mp-stage/exoclaw``.
    shutil.copytree(
        core / "exoclaw",
        STAGE_DIR / "exoclaw",
        ignore=_ignore_cpython_only,
    )

    # MP-runtime fillers at stage root — flat so plain ``import
    # typing`` / ``import dataclasses`` / etc. resolve. Two sources:
    # ``exoclaw/_mp_lib/`` (production fillers core owns — typing,
    # dataclasses) and ``tests/_micropython_stubs/`` (test-only —
    # datetime, __future__).
    for fill in (core / "exoclaw" / "_mp_lib").glob("*.py"):
        shutil.copy(fill, STAGE_DIR / fill.name)
    stubs = core / "tests" / "_micropython_stubs"
    for stub in stubs.glob("*.py"):
        shutil.copy(stub, STAGE_DIR / stub.name)

    # Each plugin's source → ``.mp-stage/<plugin_module>``.
    for pkg_name in packages:
        pkg_dir = PACKAGES_DIR / pkg_name
        # Plugin module name = package name with - swapped for _.
        # e.g. ``exoclaw-conversation`` → ``exoclaw_conversation``.
        module_name = pkg_name.replace("-", "_")
        src = pkg_dir / module_name
        if not src.is_dir():
            sys.exit(
                f"package {pkg_name} marked mp_compat but has no "
                f"{module_name}/ source directory"
            )
        shutil.copytree(
            src,
            STAGE_DIR / module_name,
            ignore=_ignore_cpython_only,
        )

    return STAGE_DIR


def _run_micro(pkg_name: str, exo_meta: dict) -> int:
    """Run MP tests for one package. Returns the runner's
    JSON-parsed result printed to stdout, exits non-zero on
    failure."""
    pkg_dir = PACKAGES_DIR / pkg_name
    tests_subdir = exo_meta.get("mp_tests_dir", "tests/micro")
    tests_dir = pkg_dir / tests_subdir
    if not tests_dir.is_dir():
        print(
            f"[skip] {pkg_name}: no {tests_subdir}/ directory; "
            f"add MP tests there to enable the gate."
        )
        return 0

    core = _core_dir()
    runner = core / "tests" / "_micropython_runner" / "run.py"
    binary = _mp_binary()
    module_name = pkg_name.replace("-", "_")

    env = os.environ.copy()
    # ``.frozen`` first so MP's frozen ``asyncio`` resolves before
    # anything in the stage; stage second so vendored stubs override
    # any missing stdlib (``typing``, ``dataclasses``, ``datetime``).
    env["MICROPYPATH"] = ".frozen:" + str(STAGE_DIR)

    cmd = [
        binary,
        # 8 MiB heap matches the ESP32-S3 target and gives the
        # multi-test runs headroom.
        "-X",
        "heapsize=8M",
        str(runner),
        "--tests-dir",
        str(tests_dir),
        # Trace coverage on this plugin's source only — the global
        # coverage threshold is per-file and we don't want core
        # files (already gated by core's CI) to dominate.
        "--cov-dir",
        str(STAGE_DIR / module_name),
    ]
    print(f"[run] {pkg_name}: {tests_dir.relative_to(WORKSPACE)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Runner emits one JSON line on stdout. Print preceding stderr
    # so import-time failures surface.
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not stdout_lines:
        print(f"[fail] {pkg_name}: runner produced no output", file=sys.stderr)
        return 1
    try:
        report = json.loads(stdout_lines[-1])
    except json.JSONDecodeError:
        print(
            f"[fail] {pkg_name}: last line not JSON: {stdout_lines[-1]!r}",
            file=sys.stderr,
        )
        return 1

    failed = report.get("failed", [])
    if failed:
        print(f"[fail] {pkg_name}: {len(failed)} test(s) failed", file=sys.stderr)
        for f in failed:
            print(f"  {f.get('test')} — {f.get('error')}", file=sys.stderr)
        return 1
    passed = len(report.get("passed", []))
    print(f"[ok]   {pkg_name}: {passed} passed")
    return 0


def _cmd_list() -> int:
    for name, _ in _mp_compat_packages():
        print(name)
    return 0


def _cmd_stage(pkg_name: str | None = None) -> int:
    pkgs = _mp_compat_packages()
    names = [n for n, _ in pkgs]
    if pkg_name is not None:
        if pkg_name not in names:
            sys.exit(
                f"{pkg_name} is not marked mp_compat in pyproject.toml; "
                f"available: {names}"
            )
        names = [pkg_name]
    _stage(names)
    print(f"[stage] {STAGE_DIR.relative_to(WORKSPACE)}: core + {names}")
    return 0


def _cmd_test(pkg_name: str) -> int:
    pkgs = _mp_compat_packages()
    by_name = {n: meta for n, meta in pkgs}
    if pkg_name not in by_name:
        sys.exit(
            f"{pkg_name} is not marked mp_compat in pyproject.toml; "
            f"available: {sorted(by_name)}"
        )
    # Stage every MP-compat package, not just the one under test —
    # plugins import each other (firmware → conversation +
    # provider-openai) and a single-package stage breaks those
    # imports under MP.
    _stage([n for n, _ in pkgs])
    return _run_micro(pkg_name, by_name[pkg_name])


def _cmd_test_all() -> int:
    pkgs = _mp_compat_packages()
    if not pkgs:
        print("no MP-compat packages found")
        return 0
    _stage([n for n, _ in pkgs])
    rc = 0
    for name, meta in pkgs:
        rc |= _run_micro(name, meta)
    return rc


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: mp_test.py {list|stage|test <pkg>|test-all}")
    cmd = sys.argv[1]
    if cmd == "list":
        sys.exit(_cmd_list())
    if cmd == "stage":
        target = sys.argv[2] if len(sys.argv) > 2 else None
        sys.exit(_cmd_stage(target))
    if cmd == "test":
        if len(sys.argv) < 3:
            sys.exit("usage: mp_test.py test <pkg>")
        sys.exit(_cmd_test(sys.argv[2]))
    if cmd == "test-all":
        sys.exit(_cmd_test_all())
    sys.exit(f"unknown subcommand: {cmd}")


if __name__ == "__main__":
    main()
