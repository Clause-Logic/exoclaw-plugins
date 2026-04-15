# exoclaw-tools-spawn (deprecated)

> **Deprecated.** ``SpawnTool`` and the ``SpawnManager`` protocol moved
> into [`exoclaw-subagent`](../exoclaw-subagent) in **0.9.0**. This
> package now exists only as a re-export shim so existing imports keep
> working for one release cycle. New code should depend on
> ``exoclaw-subagent`` directly. This package will be removed in a
> future release.

## Migration

```python
# Old (still works via the shim)
from exoclaw_tools_spawn.tool import SpawnTool, SpawnManager

# New
from exoclaw_subagent import SpawnTool, SpawnManager
```

The two were always paired in practice — every consumer that used
``SpawnTool`` also pulled in ``exoclaw-subagent`` for ``SubagentManager``
— and the split was forcing cross-package version bumps on every
change to the spawn surface. Merging them removes that coordination
tax. The shim package keeps the entry point ``exoclaw_tools_spawn.skills:spawn``
functional so host configs that list ``exoclaw-tools-spawn`` as a
skills package keep getting the spawn skill until they migrate.
