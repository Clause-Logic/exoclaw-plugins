# exoclaw-executor-dbos

DBOS-backed durable executor for exoclaw. Makes agent turns survive process restarts.

Every LLM call and tool execution is a DBOS step, checkpointed to SQLite. If the process restarts mid-turn, DBOS replays completed steps and continues from the next one.
