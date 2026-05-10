"""Codemod that vendors HKUDS/nanobot channels into exoclaw-channel-* packages.

Used by:
  - per-channel `hatch_build.py` (build-time materialization for wheel/sdist)
  - per-channel `conftest.py` (test-time materialization)
  - `sync.sh` (maintainer-side upstream-sync workflow)

Public API:
  regenerate(pkg_dir)             — regen one package
  regenerate_all_channels(repo)   — regen every package with vendor/upstream.py
  transform_source / transform_test — pure transforms exposed for tests
"""

from .codemod import transform_source, transform_test
from .regen import regenerate, regenerate_all_channels

__all__ = [
    "regenerate",
    "regenerate_all_channels",
    "transform_source",
    "transform_test",
]
