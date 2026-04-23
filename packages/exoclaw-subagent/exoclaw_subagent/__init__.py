"""Subagent manager and LLM-facing spawn tool for exoclaw.

Until 0.9.0 the LLM-facing ``SpawnTool`` and the ``SpawnManager``
protocol lived in the separate ``exoclaw-tools-spawn`` package. They
were always paired with ``SubagentManager`` in practice, and the split
forced cross-package coordination on every change to the spawn surface.
Both now live here. ``exoclaw-tools-spawn`` is kept as a deprecated
re-export shim for one release cycle so existing imports keep working.
"""

from exoclaw_subagent.batch_store import (
    AnnounceCallback,
    BatchSnapshot,
    BatchStore,
    InMemoryBatchStore,
)
from exoclaw_subagent.manager import SubagentManager
from exoclaw_subagent.spawn_tool import SpawnManager, SpawnTool
from exoclaw_subagent.spawner import (
    AsyncioSpawner,
    Runner,
    SpawnerFactory,
    SubagentHandle,
    SubagentSpawner,
)

__all__ = [
    "AnnounceCallback",
    "AsyncioSpawner",
    "BatchSnapshot",
    "BatchStore",
    "InMemoryBatchStore",
    "Runner",
    "SpawnerFactory",
    "SpawnManager",
    "SpawnTool",
    "SubagentHandle",
    "SubagentManager",
    "SubagentSpawner",
]
