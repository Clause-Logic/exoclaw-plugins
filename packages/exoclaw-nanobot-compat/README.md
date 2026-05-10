# exoclaw-nanobot-compat

Compatibility shim that lets HKUDS/nanobot channels run under exoclaw via a codemod-based vendoring pipeline. Sister channels:

- [exoclaw-channel-slack](../exoclaw-channel-slack/)
- [exoclaw-channel-telegram](../exoclaw-channel-telegram/)
- [exoclaw-channel-discord](../exoclaw-channel-discord/)
- [exoclaw-channel-email](../exoclaw-channel-email/)
- [exoclaw-channel-matrix](../exoclaw-channel-matrix/)
- [exoclaw-channel-whatsapp](../exoclaw-channel-whatsapp/)

Each vendored channel imports its `BaseChannel`, bus types, and helpers from this package instead of from upstream `nanobot.*`. The codemod that produces the vendored channel files is the sibling [`exoclaw-channel-codemod`](../exoclaw-channel-codemod/) package.

## What this package re-exports

- `BaseChannel` — drop-in replacement for `nanobot.channels.base.BaseChannel`. Conforms to exoclaw's `Channel` Protocol (`name`, `start(bus)`, `stop()`, `send(msg)`). Inherited by every vendored channel.
- `InboundMessage`, `OutboundMessage`, `MessageBus` — re-exports from `exoclaw.bus.{events,queue}`. The codemod rewrites `from nanobot.bus.events import …` to `from exoclaw_nanobot_compat import …`, so vendored channels see exoclaw's bus types under the names they expect.
- `Base` — pydantic base mirroring nanobot's (`alias_generator=to_camel`, `extra="ignore"`).
- `split_message`, `safe_filename`, `validate_url_target`, `build_help_text`, `redirect_lib_logging` — small standalone helpers ported from `nanobot.utils.*`.
- `get_data_dir(name=None)`, `get_media_dir(name=None)` — workspace path resolvers (see "Workspace layout" below).
- `set_data_dir(path)`, `set_media_dir(path)` — programmatic overrides for hosts that don't use the default `~/.exoclaw/{data,media}` layout.

## Workspace layout

Channels store load-bearing state (matrix E2E keys, slack topic state, email seen-uid tracking, whatsapp bridge token) and downloaded media. Exoclaw deliberately doesn't dictate workspace layout — that's a host concern. Resolution order, highest precedence first:

1. `set_data_dir()` / `set_media_dir()` programmatic override
2. `EXOCLAW_DATA_DIR` / `EXOCLAW_MEDIA_DIR` env vars
3. Default `~/.exoclaw/{data,media}`

Hosts (Luna, picoclaw, anything else) configure once at startup, then channels remain ignorant of where their state lives.

## Why this exists

Exoclaw's `Channel` Protocol is 32 lines (`name`, `start`, `stop`, `send`). HKUDS/nanobot's `BaseChannel` ABC is 200 LOC and bundles allow_from gating, transcription, login, default-config, streaming, etc. — all real functionality, but tightly coupled to nanobot's host shape.

The compat shim is the bridge: it provides a `BaseChannel` superset shaped like nanobot's so vendored channel files inherit unchanged, while routing `_handle_message` through exoclaw's bus and conforming to exoclaw's `Channel` Protocol on the outside.

That lets a channel-codemod vendor channel files from HKUDS upstream with mostly-mechanical transforms, ship them as `exoclaw-channel-<name>` packages, and absorb upstream bug fixes via re-running the codemod against newer commits.

## Maintaining

This compat module is consumed by the codemod's import-rewrite step. When upstream HKUDS adds a new module the channels depend on, add a shim here and add the module name to `COMPAT_MODULES` in `packages/exoclaw-channel-codemod/exoclaw_channel_codemod/codemod.py`, then re-run `packages/exoclaw-channel-codemod/sync.sh <channel> --apply` for each affected channel.
