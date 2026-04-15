"""Deprecated re-export shim for ``exoclaw-subagent``.

The ``SpawnTool`` and ``SpawnManager`` protocol moved to
``exoclaw-subagent`` in 0.9.0. This package now exists only to keep
existing imports working for one release cycle:

    # Old (still works)
    from exoclaw_tools_spawn.tool import SpawnTool, SpawnManager

    # New
    from exoclaw_subagent import SpawnTool, SpawnManager

Update your imports and depend on ``exoclaw-subagent`` directly. This
package will be removed in a future release.
"""

from exoclaw_subagent import SpawnManager, SpawnTool

__all__ = ["SpawnManager", "SpawnTool"]
