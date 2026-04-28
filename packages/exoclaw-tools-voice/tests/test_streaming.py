"""``stream_audio_request_body`` tests — drain the generator,
verify the produced bytes parse as the chat-completions JSON
shape the audio model expects."""

from __future__ import annotations

import base64
import json
from typing import AsyncIterator

import pytest
from exoclaw_tools_voice.streaming import stream_audio_request_body


async def _drain(gen: AsyncIterator[bytes]) -> bytes:
    out = bytearray()
    async for chunk in gen:
        out.extend(chunk)
    return bytes(out)


async def _audio_chunks(blob: bytes, chunk_size: int = 64) -> AsyncIterator[bytes]:
    for i in range(0, len(blob), chunk_size):
        yield blob[i : i + chunk_size]


@pytest.mark.asyncio
async def test_body_parses_as_chat_completions_request() -> None:
    """The streamed body, joined back together, must parse as
    valid JSON with the OpenAI chat-completions shape: ``model``,
    ``messages[0].content[1].input_audio.data``, etc."""
    audio = b"hello-world-fake-audio-bytes" * 5
    body_gen = stream_audio_request_body(
        model="openai/gpt-audio-mini",
        user_text="Transcribe.",
        audio_chunks=_audio_chunks(audio),
        audio_format="wav",
        max_tokens=128,
        temperature=0.0,
    )
    raw = await _drain(body_gen)
    parsed = json.loads(raw.decode("utf-8"))

    assert parsed["model"] == "openai/gpt-audio-mini"
    assert parsed["stream"] is True
    assert parsed["max_tokens"] == 128
    assert parsed["temperature"] == 0.0

    msg = parsed["messages"][0]
    assert msg["role"] == "user"
    parts = msg["content"]
    assert parts[0] == {"type": "text", "text": "Transcribe."}
    audio_part = parts[1]
    assert audio_part["type"] == "input_audio"
    assert audio_part["input_audio"]["format"] == "wav"
    # Decoded base64 should match the original bytes — proves the
    # streaming encoder didn't lose, duplicate, or pad-mid-stream
    # any data.
    decoded = base64.b64decode(audio_part["input_audio"]["data"])
    assert decoded == audio


@pytest.mark.asyncio
async def test_body_handles_empty_audio() -> None:
    """No audio bytes → empty base64 string. The JSON should still
    be valid; the model will reply with whatever it does for
    silence (probably an empty transcription)."""
    body_gen = stream_audio_request_body(
        model="openai/gpt-audio-mini",
        user_text="Transcribe.",
        audio_chunks=_audio_chunks(b""),
        audio_format="wav",
        max_tokens=64,
        temperature=0.0,
    )
    raw = await _drain(body_gen)
    parsed = json.loads(raw.decode("utf-8"))
    assert parsed["messages"][0]["content"][1]["input_audio"]["data"] == ""


@pytest.mark.asyncio
async def test_body_streams_chunk_per_audio_yield() -> None:
    """Verify the body generator emits multiple chunks rather
    than buffering — proves the chip-side memory budget is
    honored. Count yields and assert it scales with audio
    chunks."""
    big_audio = b"\x00\x01\x02" * 1000  # 3000 bytes, divisible by 3
    body_gen = stream_audio_request_body(
        model="openai/gpt-audio-mini",
        user_text="Transcribe.",
        audio_chunks=_audio_chunks(big_audio, chunk_size=300),
        audio_format="wav",
        max_tokens=64,
        temperature=0.0,
    )
    chunks: list[bytes] = []
    async for c in body_gen:
        chunks.append(c)
    # Expect: head chunk, prefix chunk, ~10 base64 chunks (one per
    # audio yield since chunk_size is divisible by 3 → encoder
    # never carries), close chunk. 13ish total. Anything close to
    # 1 would mean the body buffered.
    assert len(chunks) > 5, "expected streaming, got {} chunks".format(len(chunks))
