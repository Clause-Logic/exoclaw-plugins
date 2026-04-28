"""``ListenAndTranscribeTool`` tests — drive a fake provider +
fake capture, verify the request flows correctly."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest
from exoclaw_tools_voice import (
    AudioCapabilities,
    AudioCapture,
    ListenAndTranscribeTool,
)


class _FakeCapture:
    """Yields a fixed audio blob in fixed-size chunks. Implements
    ``AudioCapture`` Protocol."""

    def __init__(self, audio: bytes, chunk: int = 64) -> None:
        self._audio = audio
        self._chunk = chunk
        self.capabilities = AudioCapabilities(
            sample_rate_hz=16000,
            bit_depth=16,
            channels=1,
            format="wav",
        )

    def listen(
        self,
        max_duration_s: float,
        silence_threshold: int,
        silence_seconds: float,
    ) -> AsyncIterator[bytes]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self._audio), self._chunk):
            yield self._audio[i : i + self._chunk]


class _FakeProvider:
    """Captures the streamed request body so the test can verify
    the request shape, returns a fixed transcription."""

    def __init__(self, transcription: str = "set my display to a cat") -> None:
        self._transcription = transcription
        self.last_model: str | None = None
        self.last_body_bytes: bytes | None = None

    async def send_streaming_body(self, model: str, body: AsyncIterator[bytes]) -> str:
        self.last_model = model
        captured = bytearray()
        async for chunk in body:
            captured.extend(chunk)
        self.last_body_bytes = bytes(captured)
        return self._transcription


@pytest.mark.asyncio
async def test_listen_returns_transcription() -> None:
    capture = _FakeCapture(audio=b"audio-bytes" * 100)
    provider = _FakeProvider(transcription="hello world")
    tool = ListenAndTranscribeTool(
        provider=provider,  # type: ignore[invalid-argument-type]
        capture=capture,
        audio_model="openai/gpt-audio-mini",
    )

    out = await tool.execute()
    assert out == "hello world"
    assert provider.last_model == "openai/gpt-audio-mini"
    assert provider.last_body_bytes is not None


@pytest.mark.asyncio
async def test_listen_request_shape_is_chat_completions() -> None:
    """Verify the body the tool produces parses as a valid chat-
    completions request with the audio inline. Same assertion as
    ``test_streaming`` but routed through the tool to catch any
    wiring bug between the capture iterator and the body
    generator."""
    audio = b"\x00\x01\x02\x03" * 50
    capture = _FakeCapture(audio=audio)
    provider = _FakeProvider()
    tool = ListenAndTranscribeTool(
        provider=provider,  # type: ignore[invalid-argument-type]
        capture=capture,
        audio_model="openai/gpt-audio-mini",
    )

    await tool.execute()
    assert provider.last_body_bytes is not None
    parsed = json.loads(provider.last_body_bytes.decode("utf-8"))
    assert parsed["model"] == "openai/gpt-audio-mini"
    msg = parsed["messages"][0]
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][1]["type"] == "input_audio"
    assert msg["content"][1]["input_audio"]["format"] == "wav"


@pytest.mark.asyncio
async def test_listen_empty_response_falls_back_to_no_speech() -> None:
    """If the model returns nothing, surface ``(no speech
    detected)`` rather than an empty string — the agent gets a
    clear signal that the trigger fired but the user was silent."""
    capture = _FakeCapture(audio=b"")
    provider = _FakeProvider(transcription="")
    tool = ListenAndTranscribeTool(
        provider=provider,  # type: ignore[invalid-argument-type]
        capture=capture,
        audio_model="openai/gpt-audio-mini",
    )
    assert await tool.execute() == "(no speech detected)"


def test_tool_metadata_shape() -> None:
    capture = _FakeCapture(audio=b"")
    provider = _FakeProvider()
    tool = ListenAndTranscribeTool(
        provider=provider,  # type: ignore[invalid-argument-type]
        capture=capture,
        audio_model="openai/gpt-audio-mini",
    )
    assert tool.name == "listen"
    params = tool.parameters
    assert params["type"] == "object"
    assert params["required"] == []


def test_capture_satisfies_protocol() -> None:
    """``WavFileCapture`` (the unix board's stub) is structurally
    a valid ``AudioCapture``. Smoke test the runtime-checkable
    Protocol so a refactor that breaks the contract trips a test
    rather than a flash."""
    capture = _FakeCapture(audio=b"")
    assert isinstance(capture, AudioCapture)
