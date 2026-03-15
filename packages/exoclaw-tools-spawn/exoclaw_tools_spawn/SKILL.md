---
name: spawn
description: Delegate tasks to background subagents that run independently and report back
---

# Background Subagents

Spawn independent agents that run in the background. You get an immediate response and the result is delivered later.

## When to use spawn

- The task needs LLM reasoning, tool access, and may take multiple steps
- Each subagent gets a full agent loop with its own conversation
- Don't spawn subagents for simple tasks you can do directly
- If the `batch` tool is available, prefer it for deterministic fan-out (fetch URLs, read files) — spawn is for tasks that need independent reasoning

## Usage

```json
{"task": "Review PR #42 and post a summary comment", "label": "PR review"}
```

- **task** — full description of what the subagent should do
- **label** — short display name (optional, shown in notifications)

## How it works

1. Subagent starts immediately in the background
2. You get back: `"Subagent [PR review] started (id: abc123)"`
3. The subagent runs with its own conversation and tools
4. When done, the result appears as a system message in your session
5. Summarize the result naturally for the user

## Tips

- Subagents have access to the same tools as you
- Results are delivered to the session that spawned them
- Multiple subagents can run concurrently
- Failed subagents report errors back — you don't need to poll
