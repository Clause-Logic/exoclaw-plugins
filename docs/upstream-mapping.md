# Upstream Mapping: nanobot → exoclaw

Tracks where each nanobot source file lives in the exoclaw ecosystem.
Upstream source: `~/dev/nanobot` (`main` branch).

---

## exoclaw core (no-batteries branch of nanobot)

These files were refactored into the `exoclaw` package itself — protocol-only,
no batteries included.

| nanobot | exoclaw |
|---|---|
| `nanobot/agent/loop.py` | `exoclaw/agent/loop.py` |
| `nanobot/agent/tools/base.py` | `exoclaw/agent/tools/protocol.py` |
| `nanobot/agent/tools/registry.py` | `exoclaw/agent/tools/registry.py` |
| `nanobot/bus/events.py` | `exoclaw/bus/events.py` |
| `nanobot/bus/queue.py` | `exoclaw/bus/queue.py` |
| `nanobot/channels/manager.py` | `exoclaw/channels/manager.py` |
| `nanobot/providers/base.py` | `exoclaw/providers/protocol.py` |

---

## exoclaw-plugins packages (this repo)

### exoclaw-conversation

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/agent/context.py` | `exoclaw_conversation/context.py` |
| `nanobot/agent/memory.py` | `exoclaw_conversation/memory.py` |
| `nanobot/agent/skills.py` | `exoclaw_conversation/skills.py` |
| `nanobot/session/manager.py` | `exoclaw_conversation/session/manager.py` |

### exoclaw-provider-litellm

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/providers/litellm_provider.py` | `exoclaw_provider_litellm/provider.py` |

### exoclaw-tools-workspace

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/agent/tools/filesystem.py` | `exoclaw_tools_workspace/filesystem.py` |
| `nanobot/agent/tools/shell.py` | `exoclaw_tools_workspace/shell.py` |
| `nanobot/agent/tools/web.py` | `exoclaw_tools_workspace/web.py` |

### exoclaw-tools-cron

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/cron/types.py` | `exoclaw_tools_cron/types.py` |
| `nanobot/cron/service.py` | `exoclaw_tools_cron/service.py` |
| `nanobot/agent/tools/cron.py` | `exoclaw_tools_cron/tool.py` |

### exoclaw-tools-message

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/agent/tools/message.py` | `exoclaw_tools_message/tool.py` |

### exoclaw-nanobot

Meta-bundle that pulls in all exoclaw-plugins and provides one-line wiring.
Config and provider registry live here (no nanobot equivalent — nanobot baked
these into the main package).

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/config/schema.py` | `exoclaw_nanobot/config/schema.py` |
| `nanobot/config/loader.py` | `exoclaw_nanobot/config/loader.py` |
| `nanobot/providers/registry.py` | `exoclaw_nanobot/providers.py` |

---

### exoclaw-tools-spawn

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/agent/tools/spawn.py` | `exoclaw_tools_spawn/tool.py` |

### exoclaw-channel-cli

New in exoclaw — no direct nanobot equivalent (nanobot's CLI was baked into
`nanobot/cli/commands.py` and not a standalone channel).

### exoclaw-channel-heartbeat

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/heartbeat/service.py` | `exoclaw_channel_heartbeat/service.py` |

---

## No home yet

Files with no exoclaw equivalent as of the last update.

### exoclaw-subagent

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/agent/subagent.py` | `exoclaw_subagent/manager.py` |

Note: nanobot's `SubagentManager` was ~200 lines with a bespoke agent loop.
`exoclaw-subagent` is ~50 lines — it delegates to `AgentLoop.process_direct`.

### exoclaw-tools-mcp

| nanobot | exoclaw-plugins |
|---|---|
| `nanobot/agent/tools/mcp.py` | `exoclaw_tools_mcp/tool.py` |

Note: `MCPServerConfig` dataclass is new — replaces nanobot's dependency on `config/schema.py`.

### Providers

| nanobot | candidate package |
|---|---|
| `nanobot/providers/azure_openai_provider.py` | `exoclaw-provider-azure` |
| `nanobot/providers/custom_provider.py` | `exoclaw-provider-openai` |
| `nanobot/providers/openai_codex_provider.py` | `exoclaw-provider-openai` |
| `nanobot/providers/registry.py` | `exoclaw-nanobot/exoclaw_nanobot/providers.py` |
| `nanobot/providers/transcription.py` | `exoclaw-transcription` |

### Config

| nanobot | candidate package |
|---|---|
| `nanobot/config/schema.py` | `exoclaw-nanobot/exoclaw_nanobot/config/schema.py` |
| `nanobot/config/loader.py` | `exoclaw-nanobot/exoclaw_nanobot/config/loader.py` |

### Channels (tier 3)

| nanobot | candidate package |
|---|---|
| `nanobot/channels/telegram.py` | `exoclaw-channel-telegram` |
| `nanobot/channels/discord.py` | `exoclaw-channel-discord` |
| `nanobot/channels/slack.py` | `exoclaw-channel-slack` |
| `nanobot/channels/whatsapp.py` | `exoclaw-channel-whatsapp` |
| `nanobot/channels/email.py` | `exoclaw-channel-email` |
| `nanobot/channels/matrix.py` | `exoclaw-channel-matrix` |
| `nanobot/channels/feishu.py` | `exoclaw-channel-feishu` |
| `nanobot/channels/dingtalk.py` | `exoclaw-channel-dingtalk` |
| `nanobot/channels/mochat.py` | `exoclaw-channel-mochat` |
| `nanobot/channels/qq.py` | `exoclaw-channel-qq` |
| `nanobot/channels/ipc.py` | `exoclaw-channel-ipc` |
| `nanobot/channels/responses_api.py` | `exoclaw-channel-responses-api` |

---

## Intentionally excluded

Nanobot bootstrap/CLI code that is not reusable as a plugin:

- `nanobot/__main__.py`
- `nanobot/cli/commands.py`
- `nanobot/templates/`
- `nanobot/utils/helpers.py` (nanobot-internal utilities)
