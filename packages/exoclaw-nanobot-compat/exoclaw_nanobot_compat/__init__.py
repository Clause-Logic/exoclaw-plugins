"""Compat base for codemod-vendored HKUDS/nanobot channels.

Provides a `BaseChannel` shaped like nanobot's so vendored channels
inherit unchanged, while routing inbound/outbound through exoclaw's
bus and conforming to exoclaw's `Channel` Protocol
(`name`, `start(bus)`, `stop()`, `send(msg)`).

Codemod transforms applied to vendored channels:
  - `from nanobot.bus.events import …`     → drop (provided here)
  - `from nanobot.bus.queue   import …`     → drop (provided here)
  - `from nanobot.channels.base import …`   → from exoclaw_nanobot_compat import …
  - `from nanobot.config.schema import Base`→ from exoclaw_nanobot_compat import Base
  - `from nanobot.utils.helpers import …`   → from exoclaw_nanobot_compat import …
  - drop methods we route to sibling plugins (transcribe_audio, login default)
"""

from __future__ import annotations

import logging as _stdlib_logging
import os as _os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from exoclaw.bus.events import InboundMessage, OutboundMessage
from exoclaw.bus.protocol import Bus
from exoclaw.bus.queue import MessageBus
from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

# ── Re-export exoclaw bus types under their nanobot names ────────────────
# Channels do `from nanobot.bus.events import OutboundMessage` and use
# `msg.chat_id / .content / .media / .metadata`. Exoclaw's OutboundMessage
# has the same attributes. The codemod rewrites the import path; the
# channels then see exoclaw's class through the original name.
__all__ = [
    "BaseChannel",
    "InboundMessage",
    "OutboundMessage",
    "MessageBus",
    "Base",
    "split_message",
    "safe_filename",
    "get_media_dir",
    "get_data_dir",
    "set_data_dir",
    "set_media_dir",
    "validate_url_target",
    "build_help_text",
    "redirect_lib_logging",
]


# ── nanobot.utils.logging_bridge.redirect_lib_logging shim ───────────────
# Routes stdlib `logging` records (matrix-nio, slack-sdk, etc.) into loguru
# so channel logs don't bypass exoclaw's loguru-based pipeline. Same shape
# as upstream — adds a bridge handler if not present, disables propagation.


class _LoguruBridge(_stdlib_logging.Handler):
    _LEVEL_MAP = {
        _stdlib_logging.DEBUG: "DEBUG",
        _stdlib_logging.INFO: "INFO",
        _stdlib_logging.WARNING: "WARNING",
        _stdlib_logging.ERROR: "ERROR",
        _stdlib_logging.CRITICAL: "CRITICAL",
    }

    def __init__(self, lib_name: str) -> None:
        super().__init__()
        self.lib_name = lib_name

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        level = self._LEVEL_MAP.get(record.levelno, "INFO")
        frame, depth = _stdlib_logging.currentframe(), 2
        while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
            frame, depth = frame.f_back, depth + 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, "[{lib}] {message}", lib=self.lib_name, message=record.getMessage()
        )


def redirect_lib_logging(name: str, level: str | None = None) -> None:
    lib_logger = _stdlib_logging.getLogger(name)
    if not any(isinstance(h, _LoguruBridge) for h in lib_logger.handlers):
        handler = _LoguruBridge(name)
        if level is not None:
            handler.setLevel(getattr(_stdlib_logging, level.upper(), _stdlib_logging.WARNING))
        lib_logger.handlers = [handler]
        lib_logger.propagate = False


# ── nanobot.command.builtin.build_help_text shim ─────────────────────────
# Telegram's /help handler reaches into nanobot's command router for a
# generated help string. Exoclaw delegates command UX to the agent, so
# return a static neutral string. Override via env or by editing this
# function if a richer help is desired.
def build_help_text() -> str:
    return (
        "I'm an exoclaw-powered agent. Send me a message and I'll respond.\n"
        "Commands are handled by the agent itself — try asking what I can do."
    )


# ── nanobot.config.schema.Base shim ───────────────────────────────────────
class Base(BaseModel):
    """Pydantic base mirroring nanobot's: camelCase aliases, ignore extras."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


# ── nanobot.utils.helpers shims (inlined so we don't drag tiktoken etc.) ──
_FILENAME_BAD = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, max_len: int = 200) -> str:
    cleaned = _FILENAME_BAD.sub("_", name).strip("._-") or "file"
    return cleaned[:max_len]


def split_message(text: str, max_len: int = 2000) -> list[str]:
    """Split *text* at paragraph/line boundaries under *max_len* chars.

    Empty/falsy input returns ``[]`` so callers can distinguish "no
    content" from "one empty chunk" — channels rely on this to detect
    media-only messages and emit attachment-failure markers.
    """
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        cut = text.rfind("\n\n", 0, max_len)
        if cut < max_len // 2:
            cut = text.rfind("\n", 0, max_len)
        if cut < max_len // 4:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


# ── Workspace layout — channels' persistent state + media cache ───────────
# Channels store load-bearing state (matrix E2E keys, slack topic state,
# email seen-uid tracking, whatsapp bridge token) and downloaded media.
# Exoclaw deliberately doesn't dictate workspace layout — that's a host
# concern. Resolution order, highest precedence first:
#
#   1. set_data_dir() / set_media_dir() programmatic override
#   2. EXOCLAW_DATA_DIR / EXOCLAW_MEDIA_DIR env vars
#   3. Default ~/.exoclaw/{data,media}
#
# Hosts (Luna, picoclaw, anything else) configure once at startup, then
# channels remain ignorant of where their state lives.
_data_dir_override: Path | None = None
_media_dir_override: Path | None = None


def set_data_dir(path: str | Path) -> None:
    """Override the channel data dir. Call once at host startup before
    any channel's `start()`."""
    global _data_dir_override
    _data_dir_override = Path(path)


def set_media_dir(path: str | Path) -> None:
    """Override the channel media dir. Call once at host startup."""
    global _media_dir_override
    _media_dir_override = Path(path)


def get_data_dir(name: str | None = None) -> Path:
    """Channels store persistent state here. See module-level workspace
    layout comment for resolution order. Optional ``name`` returns a
    per-channel subdir (e.g. ``get_data_dir("matrix") → .../data/matrix``)."""
    if _data_dir_override is not None:
        p = _data_dir_override
    elif env := _os.environ.get("EXOCLAW_DATA_DIR"):
        p = Path(env)
    else:
        p = Path.home() / ".exoclaw" / "data"
    if name:
        p = p / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_media_dir(name: str | None = None) -> Path:
    """Channels cache downloaded media here. See module-level workspace
    layout comment for resolution order. Optional ``name`` returns a
    per-channel subdir."""
    if _media_dir_override is not None:
        p = _media_dir_override
    elif env := _os.environ.get("EXOCLAW_MEDIA_DIR"):
        p = Path(env)
    else:
        p = Path.home() / ".exoclaw" / "media"
    if name:
        p = p / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── nanobot.security.network.validate_url_target shim ─────────────────────
def validate_url_target(url: str) -> bool:
    """Crude SSRF guard — refuses obvious private/loopback hosts.

    Channels that do outbound media fetches use this. Replace with
    exoclaw.security.ssrf when available.
    """
    import ipaddress
    from urllib.parse import urlparse

    try:
        host = urlparse(url).hostname or ""
        if not host:
            return False
        if host in {"localhost", "127.0.0.1", "::1"}:
            return False
        try:
            ip = ipaddress.ip_address(host)
            # Reject any non-public address class. ``is_unspecified``
            # covers 0.0.0.0/:: which on many stacks routes to localhost;
            # ``is_multicast`` and ``is_reserved`` close other classes
            # that shouldn't be hit by an outbound channel fetch.
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_unspecified
                or ip.is_multicast
                or ip.is_reserved
            ):
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


# Note: `OutboundMessage.buttons` was added to exoclaw 0.28.0; vendored
# channels read it directly. No shim needed.


# ── BaseChannel — exoclaw Protocol-conforming, nanobot-shaped ─────────────
class BaseChannel(ABC):
    """Drop-in replacement for nanobot.channels.base.BaseChannel.

    Conforms to exoclaw's Channel Protocol (name, start, stop, send).
    Channels keep using `self.bus.publish_inbound(InboundMessage(…))` via
    the inherited `_handle_message` helper; we translate to exoclaw bus.
    """

    name: str = "base"
    display_name: str = "Base"
    transcription_provider: str = ""
    transcription_api_key: str = ""
    transcription_api_base: str = ""
    transcription_language: str | None = None
    send_progress: bool = True
    send_tool_hints: bool = False

    def __init__(self, config: Any, bus: Bus | None = None) -> None:
        self.config = config
        self.logger = logger.bind(channel=self.name)
        # `bus` is provided by exoclaw via `start(bus)`. Nanobot's
        # constructor required it eagerly; we accept either path.
        self.bus: Bus | None = bus
        self._running = False

    async def transcribe_audio(self, file_path: str | Path) -> str:
        """No-op — transcription belongs to exoclaw-tools-voice."""
        return ""

    async def login(self, force: bool = False) -> bool:
        return True

    @abstractmethod
    async def start(self, bus: Bus | None = None) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None: ...

    async def send_delta(
        self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None
    ) -> None:
        pass

    @property
    def supports_streaming(self) -> bool:
        cfg = self.config
        streaming = (
            cfg.get("streaming", False)
            if isinstance(cfg, dict)
            else getattr(cfg, "streaming", False)
        )
        return bool(streaming) and type(self).send_delta is not BaseChannel.send_delta

    def is_allowed(self, sender_id: str) -> bool:
        if isinstance(self.config, dict):
            allow_list = self.config.get("allow_from", self.config.get("allowFrom", []))
        else:
            allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            self.logger.warning("allow_from is empty — all access denied")
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """Channels call this to push inbound. Translates to exoclaw bus."""
        if not self.is_allowed(sender_id):
            self.logger.warning("Access denied for sender {}", sender_id)
            return
        if self.bus is None:
            self.logger.error("bus not set — start() not yet called?")
            return
        meta = metadata or {}
        if self.supports_streaming:
            meta = {**meta, "_wants_stream": True}
        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=meta,
            session_key_override=session_key,
        )
        await self.bus.publish_inbound(msg)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"enabled": False}

    @property
    def is_running(self) -> bool:
        return self._running


# ── Tiny Field re-export so vendored channels keep working ────────────────
__all__.append("Field")
