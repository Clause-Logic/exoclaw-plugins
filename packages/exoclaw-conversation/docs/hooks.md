# Hooks

Reference for the lifecycle hooks available in the exoclaw conversation system. This documents the hook points from openclaw's plugin architecture and tracks which are implemented in `exoclaw-conversation`.

## OpenClaw Hook Catalog

OpenClaw (the TypeScript reference implementation) defines 30 hook points across two systems. This catalog serves as the design reference for porting hooks to `exoclaw-conversation`.

### Execution Modes

- **Parallel (void):** All handlers run concurrently. No return value — observe only.
- **Sequential (modifying):** Handlers run in priority order. Return values merge across handlers (first defined override wins).
- **Synchronous sequential:** Like modifying, but handlers must be synchronous.

### Agent Hooks

| Hook | When | Mode | Inputs | Output |
|---|---|---|---|---|
| `before_model_resolve` | Before model/provider selection | Modifying | `{ prompt }` | `{ modelOverride?, providerOverride? }` |
| `before_prompt_build` | Before prompt is assembled for the LLM | Modifying | `{ prompt, messages }` | `{ systemPrompt?, prependContext? }` |
| `before_agent_start` | Legacy — combines model resolve + prompt build | Modifying | `{ prompt, messages? }` | Combined output of both above |
| `llm_input` | Immediately before payload sent to LLM | Void | `{ runId, sessionId, provider, model, systemPrompt?, prompt, historyMessages, imagesCount }` | Observe only |
| `llm_output` | Immediately after LLM response received | Void | `{ runId, sessionId, provider, model, assistantTexts, lastAssistant?, usage? }` | Observe only |
| `agent_end` | After agent run completes (success or failure) | Void | `{ messages, success, error?, durationMs? }` | Observe only |
| `before_compaction` | Before session transcript compaction | Void | `{ messageCount, compactingCount?, tokenCount?, messages?, sessionFile? }` | Observe only |
| `after_compaction` | After session transcript compaction | Void | `{ messageCount, tokenCount?, compactedCount, sessionFile? }` | Observe only |
| `before_reset` | Before session is cleared | Void | `{ sessionFile?, messages?, reason? }` | Observe only |

### Tool Hooks

| Hook | When | Mode | Inputs | Output |
|---|---|---|---|---|
| `before_tool_call` | Before a tool is invoked | Modifying | `{ toolName, params }` | `{ params?, block?, blockReason? }` |
| `after_tool_call` | After a tool call completes | Void | `{ toolName, params, result?, error?, durationMs? }` | Observe only |
| `tool_result_persist` | Before tool result written to session | Sync sequential | `{ toolName?, toolCallId?, message, isSynthetic? }` | `{ message? }` to replace |
| `before_message_write` | Before any message written to session | Sync sequential | `{ message }` | `{ block?, message? }` |

### Message Hooks

| Hook | When | Mode | Inputs | Output |
|---|---|---|---|---|
| `message_received` | Inbound message from any channel | Void | `{ from, content, timestamp?, metadata? }` | Observe only |
| `message_sending` | Before outbound message sent | Modifying | `{ to, content, metadata? }` | `{ content?, cancel? }` |
| `message_sent` | After outbound message sent | Void | `{ to, content, success, error? }` | Observe only |

### Session Hooks

| Hook | When | Mode | Inputs | Output |
|---|---|---|---|---|
| `session_start` | New session begins | Void | `{ sessionId, resumedFrom? }` | Observe only |
| `session_end` | Session ends | Void | `{ sessionId, messageCount, durationMs? }` | Observe only |

### Subagent Hooks

| Hook | When | Mode | Inputs | Output |
|---|---|---|---|---|
| `subagent_spawning` | Before subagent is spawned | Modifying | `{ childSessionKey, agentId, label?, mode, requester?, threadRequested }` | `{ status, threadBindingReady? }` or `{ status: "error", error }` |
| `subagent_delivery_target` | Resolving where to deliver subagent output | Modifying | `{ childSessionKey, requesterSessionKey, requesterOrigin?, ... }` | `{ origin? }` to override routing |
| `subagent_spawned` | After subagent spawned | Void | `{ runId, childSessionKey, agentId, label?, mode, requester?, ... }` | Observe only |
| `subagent_ended` | After subagent run completes | Void | `{ targetSessionKey, targetKind, reason, outcome?, error? }` | Observe only |

### Gateway / Internal Hooks

| Hook | When | Mode | Inputs | Output |
|---|---|---|---|---|
| `gateway_start` | HTTP server starts | Void | `{ port }` | Observe only |
| `gateway_stop` | Shutting down | Void | `{ reason? }` | Observe only |
| `command` | Any slash command | Internal | `{ sessionEntry, sessionId, commandSource, senderId, workspaceDir }` | Push to `event.messages[]` |
| `command:new` | `/new` command | Internal | Same as `command` | Push to `event.messages[]` |
| `command:reset` | `/reset` command | Internal | Same as `command` | Push to `event.messages[]` |
| `command:stop` | `/stop` command | Internal | Same as `command` | Push to `event.messages[]` |
| `agent:bootstrap` | Before bootstrap files injected | Internal | `{ workspaceDir, bootstrapFiles }` | Mutate `bootstrapFiles` to add/remove |
| `message:received` | Inbound message (internal) | Internal | `{ from, content, channelId, metadata? }` | Push to `event.messages[]` |
| `message:sent` | Outbound message (internal) | Internal | `{ to, content, success, error?, channelId }` | Push to `event.messages[]` |

---

## exoclaw-conversation Implementation Status

The Python `exoclaw-conversation` library currently supports:

### Implemented

- **`bootstrap.md` injection** — Skills can provide `hooks/exoclaw/bootstrap.md` files that are injected into the system prompt. Loaded by `SkillsLoader.get_bootstrap_injections()`.
- **Skill hook scripts** — `SkillsLoader.get_skill_hook_scripts(hook_name)` discovers executable scripts in `skills/{name}/hooks/exoclaw/{hook_name}`. Callers are responsible for execution.
- **Agent hooks** — `.md` files that spawn fire-and-forget agent turns in response to lifecycle events. See below.

### Agent Hooks

Agent hooks are markdown files in a skill's `hooks/exoclaw/` directory named after the lifecycle event (e.g. `agent_end.md`). When the event fires, each hook spawns an out-of-band agent turn using the markdown content as the prompt.

#### File format

```markdown
---
tools: set_chat_name
skills: chat
---
If this is the first turn and the chat has not been named yet,
generate a short descriptive name and call set_chat_name.
```

- **`tools:`** — comma-separated tool names the hook turn has access to. Empty = inherit from parent.
- **`skills:`** — comma-separated skill names to activate. Empty = inherit from parent.
- **Markdown body** — the prompt for the agent turn.

#### Discovery

`SkillsLoader.get_agent_hooks(hook_name)` scans all installed skills for `hooks/exoclaw/{hook_name}.md` files and returns a list of `AgentHook` dataclasses with the parsed prompt, tools, and skills.

#### Execution rules

- Hook turns are **fire-and-forget** — they do not block the parent turn.
- Hook turns receive the session path and event context so they can read the conversation.
- **No recursion** — turns spawned by hooks must not fire agent hooks themselves.

#### Supported hook points

| Hook | When | Context available |
|---|---|---|
| `agent_end` | After an agent turn completes (success or failure) | session path, chat_id, turn_count, success/error |

### Not Yet Implemented

The following hooks from openclaw have no Python equivalent yet:

- `before_prompt_build` — dynamic context injection beyond bootstrap (modifying/sequential)
- `before_tool_call` / `after_tool_call` — tool-level middleware and observation
- `llm_input` / `llm_output` — observability
- Session lifecycle hooks — cleanup and state management
