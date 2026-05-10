"""Smoke tests for the channel codemod.

Asserts the public API stays callable, the transforms are deterministic,
and the regen pipeline writes only when content actually changed.
Per-channel behavior (does the slack codemod output actually pass slack's
upstream tests?) is covered by each `exoclaw-channel-<name>` package.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from exoclaw_channel_codemod import (
    regenerate,
    regenerate_all_channels,
    transform_source,
    transform_test,
)

# Minimal HKUDS-shaped channel source — enough surface for the codemod to
# exercise import rewrite, init signature, start signature, bus capture.
# Kept as syntactically-valid Python (note the `from typing import Any`)
# so anyone running it through `python -c` or similar gets a clean parse.
_FAKE_UPSTREAM = '''"""Test channel."""
from typing import Any

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base


class FakeConfig(Base):
    pass


class FakeChannel(BaseChannel):
    name = "fake"

    def __init__(self, config: Any, bus: MessageBus):
        super().__init__(config, bus)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, msg: OutboundMessage) -> None:
        pass
'''


_FAKE_UPSTREAM_TEST = """from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.fake import FakeChannel


def test_smoke():
    bus = MessageBus()
    ch = FakeChannel({}, bus)
    assert ch.name == "fake"
"""


# ── Pure transforms ─────────────────────────────────────────────────────────


def test_transform_source_rewrites_imports() -> None:
    out, warnings = transform_source(_FAKE_UPSTREAM, "abc1234", "fake")
    assert warnings == []
    assert "from exoclaw_nanobot_compat import OutboundMessage" in out
    assert "from exoclaw_nanobot_compat import MessageBus" in out
    assert "from exoclaw_nanobot_compat import BaseChannel" in out
    assert "from exoclaw_nanobot_compat import Base" in out
    assert "from nanobot." not in out


def test_transform_source_rewrites_signatures() -> None:
    out, _ = transform_source(_FAKE_UPSTREAM, "abc1234", "fake")
    assert "def __init__(self, config: Any, bus: MessageBus = None):" in out
    assert "async def start(self, bus=None) -> None:" in out


def test_transform_source_inserts_bus_capture() -> None:
    out, _ = transform_source(_FAKE_UPSTREAM, "abc1234", "fake")
    assert "if bus is not None:" in out
    assert "self.bus = bus" in out


def test_transform_source_adds_provenance_banner() -> None:
    out, _ = transform_source(_FAKE_UPSTREAM, "abc1234567", "fake")
    assert "GENERATED" in out
    assert "abc1234567" in out
    assert "DO NOT EDIT BY HAND" in out


def test_transform_test_redirects_channel_import() -> None:
    out, warnings = transform_test(_FAKE_UPSTREAM_TEST, "abc1234", "fake", "exoclaw_channel_fake")
    assert warnings == []
    assert "from exoclaw_channel_fake.channel import FakeChannel" in out
    assert "from nanobot.channels.fake" not in out


def test_transform_is_deterministic() -> None:
    """Same input → same output."""
    out1, _ = transform_source(_FAKE_UPSTREAM, "abc1234", "fake")
    out2, _ = transform_source(_FAKE_UPSTREAM, "abc1234", "fake")
    assert out1 == out2


# ── End-to-end regenerate against a synthetic package ───────────────────────


def _build_fake_package(tmp_path: Path) -> Path:
    pkg = tmp_path / "exoclaw-channel-fake"
    (pkg / "exoclaw_channel_fake").mkdir(parents=True)
    (pkg / "tests").mkdir()
    (pkg / "vendor").mkdir()
    (pkg / "vendor" / "upstream.py").write_text(_FAKE_UPSTREAM)
    (pkg / "vendor" / "upstream_test.py").write_text(_FAKE_UPSTREAM_TEST)
    (pkg / "vendor" / "SHA").write_text("abc1234567\n")
    (pkg / "pyproject.toml").write_text(
        '[project]\nname = "exoclaw-channel-fake"\nversion = "0.1.0"\n'
    )
    return pkg


def test_regenerate_writes_channel_and_test(tmp_path: Path) -> None:
    pkg = _build_fake_package(tmp_path)
    result = regenerate(pkg)
    assert result == {"source_patches": 0, "test_patches": 0}
    chan = pkg / "exoclaw_channel_fake" / "channel.py"
    test = pkg / "tests" / "test_channel.py"
    assert chan.is_file() and chan.stat().st_size > 0
    assert test.is_file() and test.stat().st_size > 0
    assert "exoclaw_nanobot_compat" in chan.read_text()


def test_regenerate_is_idempotent(tmp_path: Path) -> None:
    """Repeated calls don't churn file mtimes — important for editor watchers."""
    pkg = _build_fake_package(tmp_path)
    regenerate(pkg)
    chan = pkg / "exoclaw_channel_fake" / "channel.py"
    first_mtime = chan.stat().st_mtime_ns
    # Re-run; content unchanged → mtime unchanged
    regenerate(pkg)
    assert chan.stat().st_mtime_ns == first_mtime


def test_regenerate_missing_upstream_raises(tmp_path: Path) -> None:
    pkg = tmp_path / "exoclaw-channel-empty"
    (pkg / "vendor").mkdir(parents=True)
    (pkg / "pyproject.toml").write_text(
        '[project]\nname = "exoclaw-channel-empty"\nversion = "0.1.0"\n'
    )
    with pytest.raises(FileNotFoundError):
        regenerate(pkg)


def test_regenerate_all_channels_skips_packages_without_vendor(tmp_path: Path) -> None:
    """Hand-written channels (cli/heartbeat/pipe) don't have vendor/ — must
    be skipped, not raise."""
    repo = tmp_path
    pkgs = repo / "packages"
    pkgs.mkdir()
    # One real vendored channel
    real = _build_fake_package(pkgs)
    real.rename(pkgs / "exoclaw-channel-fake")
    # One hand-written channel without vendor/
    hand = pkgs / "exoclaw-channel-cli"
    hand.mkdir()
    (hand / "pyproject.toml").write_text(
        '[project]\nname = "exoclaw-channel-cli"\nversion = "0.1.0"\n'
    )
    # Should not raise
    regenerate_all_channels(repo_root=repo)
    assert (pkgs / "exoclaw-channel-fake" / "exoclaw_channel_fake" / "channel.py").is_file()
