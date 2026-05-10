"""Codemod: transform a HKUDS/nanobot file (channel source OR test) for exoclaw.

Deterministic. Same input → same output. Two modes:

  python codemod.py source <path-to-upstream.py> > generated channel module
  python codemod.py test   <path-to-upstream_test.py> --pkg=NAME > generated test

Source-mode transforms:
  1. Rewrite imports — nanobot.* → exoclaw_nanobot_compat
  2. `def __init__(self, config, bus: MessageBus):` → `bus: MessageBus | None = None`
  3. `async def start(self)` → `async def start(self, bus=None)`
  4. Insert `if bus is not None: self.bus = bus` at top of start() body
  5. Insert provenance banner

Test-mode transforms:
  1. Rewrite nanobot.* imports — same as source
  2. Rewrite `from nanobot.channels.<name> import …` → `from <pkg>.channel import …`
  3. Insert provenance banner

Both modes: warn loudly on unsupported nanobot imports so a human notices new
upstream surface before it silently breaks the absorption.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CODEMOD_VERSION = "0.2.0"

COMPAT_MODULES = {
    "nanobot.bus.events",
    "nanobot.channels.base",
    "nanobot.config.schema",
    "nanobot.config.paths",
    "nanobot.utils.helpers",
    "nanobot.security.network",
    "nanobot.bus.queue",
    "nanobot.command.builtin",
    "nanobot.utils.logging_bridge",  # redirect_lib_logging
}


def _rewrite_imports(src: str, channel_name: str | None, pkg: str | None) -> tuple[str, list[str]]:
    """Rewrite `from nanobot.<x> import …` AND `import nanobot.<x> [as Y]` lines.

    `channel_name`/`pkg` are consulted in test mode (to redirect
    `nanobot.channels.<name>` references → `<pkg>.channel`).
    """
    warnings: list[str] = []
    out_lines: list[str] = []
    chan_from_re = re.compile(r"^(\s*)from nanobot\.channels\.(\w+) import (.+)$")
    chan_import_re = re.compile(r"^(\s*)import nanobot\.channels\.(\w+)(\s+as\s+\w+)?\s*$")
    from_re = re.compile(r"^(\s*)from (nanobot\.[\w.]+) import (.+)$")
    import_re = re.compile(r"^(\s*)import (nanobot\.[\w.]+)(\s+as\s+\w+)?\s*$")

    for line in src.splitlines():
        # `from nanobot.channels.<name> import …` (test-mode rewrite)
        m = chan_from_re.match(line)
        if m and pkg:
            indent, name, names = m.groups()
            if channel_name and name != channel_name:
                warnings.append(f"test imports sibling channel ({name}) — left as-is")
                out_lines.append(line)
            else:
                out_lines.append(f"{indent}from {pkg}.channel import {names}")
            continue

        # `import nanobot.channels.<name> as alias` — common late-import in tests
        m = chan_import_re.match(line)
        if m and pkg:
            indent, name, alias = m.groups()
            if channel_name and name != channel_name:
                warnings.append(f"test imports sibling channel ({name}) — left as-is")
                out_lines.append(line)
            else:
                alias = (alias or "").strip()
                if not alias:
                    alias = f"as {name}"  # `import nanobot.channels.X` exposes X
                # `import <pkg>.channel as Y` works; module is real
                out_lines.append(f"{indent}import {pkg}.channel {alias}")
            continue

        # `from nanobot.<x> import …`
        m = from_re.match(line)
        if m:
            indent, module, names = m.group(1), m.group(2), m.group(3)
            if module in COMPAT_MODULES:
                out_lines.append(f"{indent}from exoclaw_nanobot_compat import {names}")
            elif module.startswith("nanobot.channels.") and not pkg:
                warnings.append(f"source imports sibling channel: {module} — commented out")
                out_lines.append(f"{indent}# CODEMOD-DROPPED: from {module} import {names}")
            else:
                warnings.append(f"unsupported nanobot import: {module} ({names}) — commented out")
                out_lines.append(f"{indent}# CODEMOD-DROPPED: from {module} import {names}")
            continue

        # `import nanobot.<x> [as Y]`
        m = import_re.match(line)
        if m:
            indent, module, alias = m.group(1), m.group(2), m.group(3) or ""
            if module in COMPAT_MODULES:
                # Need to import as a module — use compat as the alias target
                # `import nanobot.bus.events as e` → channels rarely do this; warn
                warnings.append(f"compat module imported as bare module: {module} — manual review")
                out_lines.append(f"{indent}# CODEMOD-FLAGGED: import {module}{alias}")
            else:
                warnings.append(f"unsupported nanobot import: {module} — commented out")
                out_lines.append(f"{indent}# CODEMOD-DROPPED: import {module}{alias}")
            continue

        out_lines.append(line)
    return "\n".join(out_lines), warnings


def _rewrite_init_signature(src: str) -> str:
    return re.sub(
        r"(def __init__\(self, config:\s*[\w\[\], ]+?,\s*bus:\s*[\w\[\], ]+?)(\):)",
        r"\1 = None\2",
        src,
    )


def _rewrite_start_signature(src: str) -> str:
    return re.sub(
        r"(async def start\(self)(\)\s*->\s*None:)",
        r"\1, bus=None\2",
        src,
    )


def _add_bus_capture(src: str) -> str:
    pattern = re.compile(
        r"(async def start\(self, bus=None\)\s*->\s*None:\n)(\s+)",
    )

    def repl(m: re.Match[str]) -> str:
        head, indent = m.group(1), m.group(2)
        return f"{head}{indent}if bus is not None:\n{indent}    self.bus = bus\n{indent}"

    return pattern.sub(repl, src, count=1)


def _add_header(src: str, upstream_sha: str, kind: str, channel_name: str) -> str:
    upstream_path = (
        f"nanobot/channels/{channel_name}.py"
        if kind == "source"
        else f"tests/channels/test_{channel_name}_channel.py"
    )
    banner = (
        f'"""GENERATED — exoclaw-channel-{channel_name} ({kind}).\n'
        f"\n"
        f"DO NOT EDIT BY HAND. Regenerate via:\n"
        f"  UPSTREAM=~/hkuds-nanobot bash packages/exoclaw-channel-codemod/sync.sh "
        f"{channel_name} --apply\n"
        f"\n"
        f"Upstream:  https://github.com/HKUDS/nanobot/blob/{upstream_sha}/{upstream_path}\n"
        f"Codemod:   v{CODEMOD_VERSION}\n"
        f'"""\n\n'
    )
    src = re.sub(r'^"""[\s\S]*?"""\s*\n', "", src, count=1)
    return banner + src


def transform_source(src: str, sha: str, channel: str) -> tuple[str, list[str]]:
    src, warnings = _rewrite_imports(src, channel_name=None, pkg=None)
    src = _rewrite_init_signature(src)
    src = _rewrite_start_signature(src)
    src = _add_bus_capture(src)
    src = _add_header(src, sha, "source", channel)
    return src, warnings


def _rewrite_string_targets(src: str, channel: str, pkg: str) -> str:
    """Rewrite string-quoted dotted paths used by `monkeypatch.setattr` etc.

      "nanobot.channels.<channel>.X.Y"  →  "<pkg>.channel.X.Y"
      'nanobot.channels.<channel>.X'    →  '<pkg>.channel.X'

    Pytest's `monkeypatch.setattr(target, value)` resolves the string by
    importing the leftmost importable prefix and walking attrs. We rewrite
    the leftmost prefix to point at our generated module so the resolution
    finds our class/function instead of nanobot's.
    """
    pattern = re.compile(rf'(["\'])nanobot\.channels\.{re.escape(channel)}\.([\w.]+)\1')
    return pattern.sub(rf"\1{pkg}.channel.\2\1", src)


def transform_test(src: str, sha: str, channel: str, pkg: str) -> tuple[str, list[str]]:
    src, warnings = _rewrite_imports(src, channel_name=channel, pkg=pkg)
    src = _rewrite_string_targets(src, channel, pkg)
    src = _add_header(src, sha, "test", channel)
    return src, warnings


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("source", "test"):
        print(f"usage: {argv[0]} (source|test) <path> [--pkg=NAME]", file=sys.stderr)
        return 2
    mode = argv[1]
    path = Path(argv[2])
    pkg = None
    for arg in argv[3:]:
        if arg.startswith("--pkg="):
            pkg = arg.split("=", 1)[1]

    src = path.read_text()
    sha_path = path.parent / "SHA"
    sha = sha_path.read_text().strip() if sha_path.exists() else "unknown"
    # vendor file lives at packages/exoclaw-channel-<name>/vendor/upstream*.py
    channel_name = path.parent.parent.name.removeprefix("exoclaw-channel-")

    if mode == "source":
        out, warnings = transform_source(src, sha, channel_name)
    else:
        if pkg is None:
            pkg = f"exoclaw_channel_{channel_name}"
        out, warnings = transform_test(src, sha, channel_name, pkg)

    for w in warnings:
        print(f"# WARN: {w}", file=sys.stderr)
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
