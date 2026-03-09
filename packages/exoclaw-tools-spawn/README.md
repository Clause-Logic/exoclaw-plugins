# exoclaw-tools-spawn

Subagent spawn tool implementing the exoclaw `ToolBase` protocol — lets the agent delegate tasks to background subagents and receive results asynchronously.

## Install

```
pip install exoclaw-tools-spawn
```

## Usage

```python
from exoclaw_tools_spawn.tool import SpawnTool, SpawnManager

# SpawnManager is a Protocol — implement it or use exoclaw-subagent's SubagentManager
spawn_tool = SpawnTool(manager=subagent_manager)

# Update context per turn
spawn_tool.set_context(channel="cli", chat_id="direct")
```

`SpawnTool` exposes a `spawn` action to the LLM. The `SpawnManager` protocol requires a single `spawn(task, label, origin_channel, origin_chat_id, session_key, search)` coroutine. The concrete implementation is provided by `exoclaw-subagent`.
