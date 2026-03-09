# exoclaw-subagent

Concrete `SpawnManager` implementation for exoclaw — spawns background subagents by nesting a fresh `AgentLoop` and announces results back to the originating session via the bus.

## Install

```
pip install exoclaw-subagent
```

## Usage

```python
from exoclaw_subagent.manager import SubagentManager

subagent_manager = SubagentManager(
    provider=provider,
    bus=bus,
    conversation_factory=lambda: DefaultConversation.create(
        workspace=workspace,
        provider=provider,
        model=model,
    ),
    tools=tools,
    model=model,
    max_iterations=15,
)

# Pass to SpawnTool
from exoclaw_tools_spawn.tool import SpawnTool
spawn_tool = SpawnTool(manager=subagent_manager)
```

`SubagentManager.spawn()` returns immediately; the task runs in a background `asyncio` task. On completion, the result is injected back into the originating session as a system `InboundMessage` on the bus.
