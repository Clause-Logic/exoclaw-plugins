"""Backwards-compatibility tests for the deprecated re-export shim.

After the 0.9.0 merge into ``exoclaw-subagent``, this package only
exists to keep existing imports working for one release cycle. These
tests pin the import paths host code is likely using so the shim
can't silently rot.
"""

from __future__ import annotations


def test_top_level_reexport() -> None:
    """``from exoclaw_tools_spawn import …`` keeps working."""
    from exoclaw_subagent import SpawnManager as CanonicalManager
    from exoclaw_subagent import SpawnTool as CanonicalTool
    from exoclaw_tools_spawn import SpawnManager, SpawnTool

    assert SpawnTool is CanonicalTool
    assert SpawnManager is CanonicalManager


def test_tool_submodule_reexport() -> None:
    """``from exoclaw_tools_spawn.tool import …`` keeps working.

    This is the import path nanobot and host apps were using before the
    merge — the most important compatibility surface.
    """
    from exoclaw_subagent import SpawnManager as CanonicalManager
    from exoclaw_subagent import SpawnTool as CanonicalTool
    from exoclaw_tools_spawn.tool import SpawnManager, SpawnTool

    assert SpawnTool is CanonicalTool
    assert SpawnManager is CanonicalManager


def test_skills_entry_point_shim() -> None:
    """``exoclaw_tools_spawn.skills:spawn`` returns the same dict shape
    as the canonical loader, with content from ``exoclaw_subagent``'s
    SKILL.md.
    """
    from exoclaw_subagent.skills import spawn as canonical_spawn
    from exoclaw_tools_spawn.skills import spawn

    shim_result = spawn()
    canonical_result = canonical_spawn()

    assert shim_result == canonical_result
    assert shim_result["name"] == "spawn"
    assert "Background Subagents" in shim_result["content"]
