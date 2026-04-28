"""``AudioCapture`` Protocol — the boundary between hardware-
specific microphone access and the cross-runtime voice tool.

Same shape as ``Display`` in ``exoclaw-screen``: the tool only
sees this Protocol; boards implement it however their hardware
demands. The unix sim provides ``WavFileCapture`` (reads a
pre-staged WAV file); the E1001 chip board provides ``I2SCapture``
(reads PCM from the PDM mic via ``machine.I2S``).

The Protocol is async-iterable so the voice tool can stream PCM
bytes to the LLM as the mic captures them — peak heap stays
bounded by one chunk + base64 encoding overhead.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


class AudioCapabilities:
    """Per-board audio facts the voice tool consults to format the
    LLM payload correctly. Mirror of ``DisplayCapabilities`` in
    ``exoclaw-screen``: concrete fields the LLM API and any silence
    detector need at construction time."""

    def __init__(
        self,
        sample_rate_hz: int,
        bit_depth: int,
        channels: int,
        format: str = "wav",
    ) -> None:
        # Samples per second — typically 16000 for speech.
        self.sample_rate_hz = sample_rate_hz
        # Bits per sample — typically 16 for int16 PCM.
        self.bit_depth = bit_depth
        # 1 (mono) or 2 (stereo). Speech is mono.
        self.channels = channels
        # The format string the LLM API expects in the
        # ``input_audio.format`` field. ``"wav"`` means the bytes
        # produced by ``listen()`` are a complete WAV file (header
        # + PCM data); ``"pcm"`` would mean raw PCM with the
        # sample rate / bit depth / channels conveyed out-of-band.
        # OpenAI's audio-input models accept ``"wav"`` and
        # ``"mp3"``; ``"wav"`` is the path of least friction since
        # we can synthesise a WAV header on the fly.
        self.format = format


@runtime_checkable
class AudioCapture(Protocol):
    """Cross-runtime mic seam.

    ``capabilities`` is a Protocol-level attribute (not a method)
    because the voice tool consults it eagerly to format the LLM
    payload. Boards populate it before the capture is wired into
    ``ListenAndTranscribeTool``.

    ``listen()`` is an async iterator. Each iteration yields a
    chunk of PCM bytes (or, for ``format="wav"`` captures, the
    WAV header on the first yield followed by PCM chunks). The
    iterator terminates when one of these triggers fires:

    - ``max_duration_s`` elapsed since open
    - sustained silence below threshold for ``silence_seconds``
    - hardware-specific external stop (button release, EOF on
      a file-backed capture, etc.)

    Concrete impls decide which triggers apply. The Protocol just
    promises the iterator will eventually terminate so the LLM
    request body generator doesn't hang indefinitely.
    """

    capabilities: AudioCapabilities

    def listen(
        self,
        max_duration_s: float,
        silence_threshold: int,
        silence_seconds: float,
    ) -> AsyncIterator[bytes]:
        """Return an async iterator of audio bytes."""
        ...
