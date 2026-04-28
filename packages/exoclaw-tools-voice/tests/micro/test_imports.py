"""MicroPython smoke test for ``exoclaw-tools-voice``.

Pure-Python — no pytest. Driven by the workspace's ``mise run
test-micro`` task on a coverage-variant MicroPython binary.

Verifies the streaming body generator + the chunked base64
encoder + the tool's metadata accessors all import and run
under MP semantics. The actual chip-side ``AudioCapture`` impl
(I2S PDM mic on the E1001) lives in the firmware board tree and
isn't exercised here — this test covers the cross-runtime parts.
"""

import asyncio


def test_top_level_imports():
    from exoclaw_tools_voice import (
        AudioCapabilities,
        AudioCapture,
        ListenAndTranscribeTool,
        stream_audio_request_body,
    )

    assert callable(AudioCapabilities)
    # AudioCapture is a Protocol — runtime checkable but not
    # callable as a constructor; just verify it imported.
    assert AudioCapture is not None
    assert callable(ListenAndTranscribeTool)
    assert callable(stream_audio_request_body)


def test_skill_entry_point_returns_dict():
    from exoclaw_tools_voice.skills import voice

    skill = voice()
    assert isinstance(skill, dict)
    assert skill["name"] == "voice"
    assert "content" in skill
    assert skill["content"]
    assert "path" not in skill


def test_b64_encoder_round_trip():
    """The chunked base64 encoder is on the chip-side hot path
    (every audio chunk goes through it). Verify it produces the
    same output as a one-shot encode regardless of chunk size."""
    try:
        import binascii  # MP unix-port has it; chip MP usually does too

        ref = binascii.b2a_base64(b"hello world from MP", newline=False).decode("ascii")
    except ImportError:
        # Some chip variants don't ship binascii.b2a_base64 — fall
        # back to a known fixture so the test still runs.
        ref = "aGVsbG8gd29ybGQgZnJvbSBNUA=="

    from exoclaw_tools_voice.b64 import B64StreamEncoder

    blob = b"hello world from MP"
    for chunk_size in (1, 3, 4, 7, 100):
        enc = B64StreamEncoder()
        parts = []
        for i in range(0, len(blob), chunk_size):
            parts.append(enc.encode(blob[i : i + chunk_size]))
        parts.append(enc.flush())
        got = "".join(parts)
        assert got == ref, "chunk={}: {!r} != {!r}".format(chunk_size, got, ref)


def test_streaming_body_yields_multiple_chunks():
    """The body generator must emit the JSON envelope, the prefix
    up to the audio data, then base64 chunks, then the closing
    bracket sequence — all separately, never as one string. This
    is the chip-side memory-bound contract: peak heap stays ~one
    chunk."""

    # Class-based async iterator — ``async def`` + ``yield`` +
    # ``async for`` collapses to a sync generator on MP 1.27 that
    # ``async for`` can't drive. Same pattern as
    # ``streaming._AudioBodyIter`` itself.
    class _AudioIter:
        def __init__(self):
            self._chunks = (b"abc", b"def", b"ghi")
            self._i = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._i >= len(self._chunks):
                raise StopAsyncIteration
            c = self._chunks[self._i]
            self._i += 1
            return c

    async def _run():
        from exoclaw_tools_voice.streaming import stream_audio_request_body

        gen = stream_audio_request_body(
            model="openai/gpt-audio-mini",
            user_text="Transcribe.",
            audio_chunks=_AudioIter(),
            audio_format="wav",
            max_tokens=64,
            temperature=0.0,
        )
        chunks = []
        async for c in gen:
            chunks.append(c)
        return chunks

    chunks = asyncio.run(_run())
    assert len(chunks) >= 4, "expected streaming, got {} chunks".format(len(chunks))
    # The whole body should still parse — round-trip via concat +
    # the stdlib JSON parser proves the chunk boundaries didn't
    # corrupt anything.
    import json as _json

    raw = b"".join(chunks).decode("utf-8")
    parsed = _json.loads(raw)
    assert parsed["model"] == "openai/gpt-audio-mini"
    assert parsed["messages"][0]["content"][1]["type"] == "input_audio"
