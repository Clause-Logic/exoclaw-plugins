"""Deprecated submodule shim — re-exports from ``exoclaw_subagent``.

``SpawnTool`` and ``SpawnManager`` moved to ``exoclaw_subagent`` in
0.9.0. This module is kept so that ``from exoclaw_tools_spawn.tool
import SpawnTool, SpawnManager`` continues to work for one release
cycle. New code should ``from exoclaw_subagent import …`` directly.
"""

from exoclaw_subagent.spawn_tool import SpawnManager, SpawnTool

__all__ = ["SpawnManager", "SpawnTool"]
