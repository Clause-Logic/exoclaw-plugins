"""Smoke tests for the nanobot→exoclaw compat shim.

These assert the surface every codemod-vendored channel relies on:
the BaseChannel ABC shape, the workspace-path resolution order, and
the small inlined helpers. Channel-level behavior is covered by each
``exoclaw-channel-<name>`` package's ported upstream tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from exoclaw.bus.events import InboundMessage as ExoInbound
from exoclaw.bus.events import OutboundMessage as ExoOutbound
from exoclaw.bus.queue import MessageBus as ExoMessageBus
from exoclaw.channels.protocol import Channel
from exoclaw_nanobot_compat import (
    BaseChannel,
    InboundMessage,
    MessageBus,
    OutboundMessage,
    build_help_text,
    get_data_dir,
    get_media_dir,
    safe_filename,
    set_data_dir,
    set_media_dir,
    split_message,
    validate_url_target,
)

# ── Re-exports point at the real exoclaw classes ────────────────────────────


def test_re_exports_are_exoclaw_classes() -> None:
    """The codemod rewrites `from nanobot.bus.events import …` to `from
    exoclaw_nanobot_compat import …`; channels then use the same name and
    must get the real exoclaw class."""
    assert InboundMessage is ExoInbound
    assert OutboundMessage is ExoOutbound
    assert MessageBus is ExoMessageBus


# ── BaseChannel shape ───────────────────────────────────────────────────────


class _ConcreteChannel(BaseChannel):
    name = "test"

    async def start(self, bus=None):  # noqa: D401
        if bus is not None:
            self.bus = bus
        self._running = True

    async def stop(self):
        self._running = False

    async def send(self, msg):
        pass


def test_base_channel_satisfies_exoclaw_protocol() -> None:
    """A BaseChannel subclass must duck-type as exoclaw.channels.protocol.Channel."""
    ch = _ConcreteChannel(config={"allow_from": ["*"]})
    assert isinstance(ch, Channel), "BaseChannel subclass must satisfy exoclaw Channel Protocol"


def test_base_channel_init_accepts_optional_bus() -> None:
    """Codemod rewrites nanobot's `__init__(self, config, bus)` to make `bus`
    default to None; exoclaw passes the bus on `start(bus)` instead. Both
    constructor shapes must work."""
    ch_no_bus = _ConcreteChannel({"allow_from": ["*"]})
    assert ch_no_bus.bus is None
    bus = MessageBus()
    ch_with_bus = _ConcreteChannel({"allow_from": ["*"]}, bus)
    assert ch_with_bus.bus is bus


@pytest.mark.asyncio
async def test_handle_message_publishes_to_bus_with_allow() -> None:
    bus = MessageBus()
    ch = _ConcreteChannel({"allow_from": ["alice"]}, bus)
    await ch._handle_message("alice", "chat-1", "hello")
    msg = await bus.consume_inbound()
    assert msg.channel == "test"
    assert msg.sender_id == "alice"
    assert msg.content == "hello"
    assert msg.session_key == "test:chat-1"


@pytest.mark.asyncio
async def test_handle_message_drops_unallowed_sender() -> None:
    bus = MessageBus()
    ch = _ConcreteChannel({"allow_from": ["alice"]}, bus)
    await ch._handle_message("eve", "chat-1", "should be dropped")
    assert bus.inbound.empty()


@pytest.mark.asyncio
async def test_handle_message_wildcard_allow() -> None:
    bus = MessageBus()
    ch = _ConcreteChannel({"allow_from": ["*"]}, bus)
    await ch._handle_message("anyone", "chat-1", "x")
    msg = await bus.consume_inbound()
    assert msg.sender_id == "anyone"


# ── Workspace layout — resolution order ─────────────────────────────────────


def test_workspace_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Override > env var > default."""
    # Reset overrides
    set_data_dir(tmp_path / "override")
    assert get_data_dir() == tmp_path / "override"

    # Programmatic override wins over env var
    monkeypatch.setenv("EXOCLAW_DATA_DIR", str(tmp_path / "via_env"))
    assert get_data_dir() == tmp_path / "override"

    # After reset, env var is used
    import exoclaw_nanobot_compat as compat

    compat._data_dir_override = None
    assert get_data_dir() == tmp_path / "via_env"

    # No env, no override → default
    monkeypatch.delenv("EXOCLAW_DATA_DIR")
    assert get_data_dir() == Path.home() / ".exoclaw" / "data"


def test_workspace_subnamespace(tmp_path: Path) -> None:
    """``get_data_dir("matrix")`` returns ``…/data/matrix`` and creates it."""
    set_data_dir(tmp_path)
    p = get_data_dir("matrix")
    assert p == tmp_path / "matrix"
    assert p.is_dir()
    set_media_dir(tmp_path / "media")
    p2 = get_media_dir("slack")
    assert p2 == tmp_path / "media" / "slack"
    assert p2.is_dir()


# ── Helpers ─────────────────────────────────────────────────────────────────


def test_split_message_empty_returns_empty_list() -> None:
    """Channels rely on this distinction to detect media-only messages and
    emit attachment-failure markers (e.g. discord _build_chunks)."""
    assert split_message("") == []
    assert split_message("hi") == ["hi"]


def test_split_message_chunks_under_max_len() -> None:
    text = "para1\n\nlong " * 200
    chunks = split_message(text, max_len=50)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_safe_filename() -> None:
    assert safe_filename("hello world.txt") == "hello_world.txt"
    assert safe_filename("../etc/passwd") == "etc_passwd"
    assert safe_filename("") == "file"
    assert safe_filename("a" * 500, max_len=10) == "aaaaaaaaaa"


def test_validate_url_target_rejects_non_public_ips() -> None:
    # Hostnames + literal loopback
    assert not validate_url_target("http://localhost/x")
    assert not validate_url_target("http://127.0.0.1/x")
    assert not validate_url_target("http://[::1]/x")
    # Private ranges
    assert not validate_url_target("http://10.0.0.1/x")
    assert not validate_url_target("http://192.168.1.1/x")
    assert not validate_url_target("http://172.16.0.1/x")
    # Link-local
    assert not validate_url_target("http://169.254.169.254/x")  # AWS IMDS
    # Unspecified — routes to localhost on many stacks
    assert not validate_url_target("http://0.0.0.0/x")
    # Multicast / reserved
    assert not validate_url_target("http://224.0.0.1/x")
    assert not validate_url_target("http://240.0.0.1/x")
    # Public should pass
    assert validate_url_target("https://example.com/x")
    assert validate_url_target("http://8.8.8.8/x")


def test_build_help_text_returns_neutral_string() -> None:
    """Exoclaw has no command system; channels' /help renders this static
    string instead of nanobot's command-router-derived help."""
    text = build_help_text()
    assert "exoclaw" in text.lower()
