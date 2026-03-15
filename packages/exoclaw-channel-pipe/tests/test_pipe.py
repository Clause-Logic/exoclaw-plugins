"""Tests for PipeChannel."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from exoclaw.bus.events import OutboundMessage
from exoclaw_channel_pipe import PipeChannel


def test_channel_name() -> None:
    ch = PipeChannel()
    assert ch.name == "pipe"


def test_custom_chat_id() -> None:
    ch = PipeChannel(chat_id="test-session")
    assert ch._chat_id == "test-session"


@pytest.mark.asyncio
async def test_send_writes_content(capsys: pytest.CaptureFixture[str]) -> None:
    ch = PipeChannel()
    await ch.send(OutboundMessage(channel="pipe", chat_id="x", content="hello world"))
    captured = capsys.readouterr()
    assert "hello world" in captured.out


@pytest.mark.asyncio
async def test_send_empty_content(capsys: pytest.CaptureFixture[str]) -> None:
    ch = PipeChannel()
    await ch.send(OutboundMessage(channel="pipe", chat_id="x", content=""))
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.asyncio
async def test_stop() -> None:
    ch = PipeChannel()
    ch._running = True
    await ch.stop()
    assert ch._running is False
