# Skills

Reference for the skill system in `exoclaw-conversation`. Skills are markdown files that teach the agent how to use specific tools or perform tasks. They are the primary mechanism for extending agent capabilities without changing code.

## Concepts

A **skill** is a directory containing a `SKILL.md` file with optional frontmatter and a markdown body. The frontmatter declares metadata (name, description, tools, requirements). The body is injected into the system prompt when the skill is active.

Skills can also provide **hooks** (lifecycle scripts and agent hook prompts) and declare **tool dependencies** so the agent only sees tools relevant to its current capabilities.

### Active vs Installed

Not all installed skills are active at any given time. The distinction matters because:

- Only **active** skills have their content injected into the system prompt.
- Only **active** skills contribute tools to the agent's tool set.
- Only **active** skills have their agent hooks fired on lifecycle events.

A skill becomes active through one of three paths:

1. **Always-on** — The skill's frontmatter includes `always: true`. These activate on every turn automatically.
2. **Channel/config** — The caller passes skill names via the `skills` parameter to `build_prompt()`. This is how the host application (e.g. standd_agent) enables skills per channel or deployment config.
3. **Dynamic load** — The agent calls the `load_skill` tool at runtime, which invokes `SkillsLoader.activate_skill(name)`. This lets the agent pull in capabilities on demand based on the user's request.

Active skills for a turn are computed as `always_skills + explicitly_requested_skills`.

## Skill Directory Layout

```
skills/
  my-skill/
    SKILL.md                          # Required — skill content + frontmatter
    hooks/
      exoclaw/
        bootstrap.md                  # Injected into system prompt for all sessions
        agent_end.md                  # Agent hook — fires after turn completion
```

Skills are discovered from three sources in priority order (highest first):

1. **Workspace** — `{workspace}/skills/{name}/SKILL.md`
2. **Builtin** — From a configured builtin skills directory
3. **Package** — Via Python entry points in the `exoclaw.skills` group

When the same skill name exists in multiple sources, the highest-priority source wins.

## SKILL.md Format

```markdown
---
name: my-skill
description: Short description shown in the skills summary
tools: tool_a, tool_b
always: true
metadata: {"exoclaw": {"requires": {"bins": ["git"], "env": ["API_KEY"]}}}
---

# My Skill

Instructions for the agent go here. This content is injected into the
system prompt when the skill is active.
```

### Frontmatter Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Skill name (must match directory name) |
| `description` | string | One-line summary shown in the `<skills>` listing |
| `tools` | comma-separated | Optional tool names this skill requires. When active, these tools are added to the agent's available set via `get_active_optional_tools()` |
| `always` | boolean | If `true`, this skill is active on every turn |
| `metadata` | JSON string | Nested config under `exoclaw` (or `nanobot`/`openclaw`). Supports `requires.bins` and `requires.env` for availability checks |

### Requirements

Skills can declare dependencies that must be present for the skill to be available:

- **`requires.bins`** — CLI binaries that must exist on `$PATH` (checked via `shutil.which`)
- **`requires.env`** — Environment variables that must be set

Unavailable skills appear in the `<skills>` summary with `available="false"` and a description of what's missing. The agent can see them but cannot activate them.

## Activation Flow

### At Prompt Build Time

When `DefaultConversation.build_prompt()` is called:

1. `ContextBuilder.build_system_prompt()` computes active skills:
   - `always_skills` — from `SkillsLoader.get_always_skills()`
   - `extra_skills` — from the `skills` kwarg (channel/config driven)
   - `active_skills = always_skills + extra_skills`
2. Active skills' `SKILL.md` content is loaded and injected under `# Active Skills`
3. Active skills' tool declarations are collected into `_active_optional_tools`
4. Non-active but available skills appear in a `<skills>` XML block so the agent can discover and load them via `load_skill`

### Dynamic Activation via `load_skill`

The `load_skill` tool lets the agent activate skills on demand:

1. Agent sees available skills listed in `<skills>` in the system prompt
2. Agent calls `load_skill(name="my-skill")`
3. `SkillsLoader.activate_skill(name)` returns the skill content + declared tool names
4. The consumer merges the returned tool names into the active set
5. Subsequent LLM calls include the new tools

The standard tool definition is exported as `LOAD_SKILL_TOOL_DEF` for consumers to include in their tool list.

## Hooks

Skills can provide hooks that run at lifecycle points. See [hooks.md](hooks.md) for the full hook catalog.

### Bootstrap Hooks

`hooks/exoclaw/bootstrap.md` — Content injected into the system prompt for every session, regardless of whether the skill is active. Used for global context that should always be present. Discovered by `SkillsLoader.get_bootstrap_injections()`.

### Agent Hooks

`hooks/exoclaw/{hook_name}.md` — Markdown files that spawn fire-and-forget agent turns when lifecycle events fire. Only hooks from **active skills** are executed (see [Active vs Installed](#active-vs-installed)).

Agent hooks support frontmatter to control the hook turn's capabilities:

```markdown
---
tools: set_chat_name
skills: chat
---
If this chat has not been named yet, generate a short descriptive
name and call set_chat_name.
```

See [hooks.md](hooks.md) for supported hook points and execution rules.

### Script Hooks

`hooks/exoclaw/{hook_name}` (no extension, executable) — Shell scripts run by the caller at lifecycle points. Discovered by `SkillsLoader.get_skill_hook_scripts(hook_name)`.

## API Reference

### SkillsLoader

The core class for skill discovery, loading, and metadata.

| Method | Description |
|---|---|
| `list_skills(filter_unavailable=True)` | List all skills. Returns `[{name, path, source}]` |
| `load_skill(name)` | Load a skill's SKILL.md content. Returns `str \| None` |
| `activate_skill(name)` | Load content + resolve tool names. Returns `LoadSkillResult` |
| `load_skills_for_context(skill_names)` | Load multiple skills formatted for system prompt injection |
| `build_skills_summary(only=None)` | Build XML `<skills>` listing. `only` filters to a set of names |
| `get_always_skills()` | List skill names with `always: true` |
| `get_tools_for_skills(skill_names)` | Union of tool names declared by the given skills |
| `get_agent_hooks(hook_name)` | Discover agent hooks across installed skills |
| `get_bootstrap_injections()` | Collect bootstrap.md content from all skills |
| `get_skill_hook_scripts(hook_name)` | Find executable hook scripts across all skills |
| `get_skill_metadata(name)` | Parse frontmatter for a skill |

### Key Data Classes

**`LoadSkillResult`** — Returned by `activate_skill()`:
- `content: str` — The skill's SKILL.md content (or error message)
- `tool_names: list[str]` — Tool names declared by the skill

**`AgentHook`** — Returned by `get_agent_hooks()`:
- `skill_name: str` — The skill that owns this hook
- `hook_name: str` — The lifecycle event name
- `prompt: str` — Markdown body (frontmatter stripped)
- `tools: list[str]` — Tool names from frontmatter
- `skills: list[str]` — Skill names from frontmatter
