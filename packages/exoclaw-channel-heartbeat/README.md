# exoclaw-channel-heartbeat

Periodic heartbeat service that wakes the agent on a schedule to check for active tasks in `HEARTBEAT.md`.

## Install

```
pip install exoclaw-channel-heartbeat
```

## Usage

```python
from pathlib import Path
from exoclaw_channel_heartbeat.service import HeartbeatService

heartbeat = HeartbeatService(
    workspace=Path("~/.nanobot/workspace").expanduser(),
    provider=provider,       # any exoclaw LLMProvider
    model="anthropic/claude-opus-4-5",
    on_execute=agent_loop.process_direct,   # called when tasks are found
    on_notify=send_to_user,                 # optional: deliver the result
    interval_s=30 * 60,
)

await heartbeat.start()
```

Each tick reads `HEARTBEAT.md`, asks the LLM via a structured tool call whether there are active tasks (`skip` / `run`), and only invokes `on_execute` when the decision is `run`.
