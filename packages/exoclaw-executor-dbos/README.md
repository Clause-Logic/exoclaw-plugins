# exoclaw-executor-dbos

DBOS-backed durable executor for exoclaw. Makes agent turns survive process restarts.

Every LLM call and tool execution is a DBOS step, checkpointed to SQLite. If the process restarts mid-turn, DBOS replays completed steps and continues from the next one.

## Session steering

Construct the executor with a persistent workspace to enable durable
safe-boundary steering:

```python
executor = DBOSExecutor(steering_workspace=workspace)
```

An inbound follow-up for a currently active session is recorded under the
channel event's DBOS workflow ID, then `AgentLoop(on_steer=executor.drain_steering)`
injects it after the current model or tool boundary. Channel retries use the
same workflow ID as normal inbound processing, so they do not generate a
duplicate response.
