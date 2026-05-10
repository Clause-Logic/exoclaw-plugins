# exoclaw-channel-codemod

The codemod that vendors HKUDS/nanobot channels into exoclaw — used as a build-time and test-time dependency by every `exoclaw-channel-{slack,telegram,discord,email,matrix,whatsapp}` package.

## What it does

Given a HKUDS/nanobot channel source file (`vendor/upstream.py`) and optional ports (`vendor/upstream_test.py`) snapshotted at a pinned commit (`vendor/SHA`), plus optional channel-specific patches (`patches/00NN-*.patch`), it produces:

- `exoclaw_channel_<name>/channel.py` — the channel module ready to ship
- `tests/test_channel.py` — the upstream test suite re-targeted at the local module

Transforms are mechanical (import rewrites, signature tweaks, bus capture, monkeypatch string-target rewrites). Patches are applied after the codemod for channel-specific tweaks that can't be generalized.

## Public API

```python
from exoclaw_channel_codemod import regenerate

regenerate(pkg_dir)  # one package
```

Pure-Python and idempotent (only writes when content changed).

## Maintainer scripts

- `sync.sh <name> [--apply]` — fetch latest upstream, diff, optionally regenerate
- `scaffold-channel.sh <name>` — bootstrap a new channel package by snapshotting upstream

## Why this is its own package

- **Build isolation**: hatchling builds each `exoclaw-channel-*` wheel in an isolated env. The hatch hook in those packages declares `exoclaw-channel-codemod` as a build dependency so it's importable inside the sandbox.
- **Audit boundary**: the codemod is the trusted transform. Versioning it as a real package makes "what was used to derive this generated file" explicit.

## Audit boundary

Vendored channel packages commit only `vendor/upstream*.py` + `vendor/SHA` + `patches/`. The generated `channel.py` and `test_channel.py` are gitignored — they're derivatives. Anyone reviewing a channel package can verify "this wheel = upstream@SHA + this codemod version + listed patches" by checking out the source and running the test suite (which materializes the codemod output via `conftest.py`).
