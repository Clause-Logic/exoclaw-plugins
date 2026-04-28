"""Unix-port audio capture.

Two impls in this file:

- ``WavFileCapture`` — reads bytes from a pre-recorded WAV file.
  Useful for deterministic tests; matches the protocol shape the
  voice tool expects.
- ``LiveMicCapture`` — records the system default mic via an
  ``ffmpeg -f avfoundation`` subprocess, then yields the
  resulting WAV bytes. Same shell-out pattern as the screen's
  ``host_render`` — MP unix-port can't load CFFI-wrapped audio
  libs (sounddevice, PyAudio, …) but it can call ``os.system``,
  and ffmpeg ships with most dev environments via Homebrew.

Both implement ``AudioCapture`` (capabilities attr +
async-iterable ``listen``). The chip-side ``I2SCapture`` will
live in ``boards/reterminal_e1001/audio.py`` once we port the
PDM driver.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator

from exoclaw_tools_voice.capture import AudioCapabilities


class WavFileCapture:
    """Yields the bytes of a WAV file in fixed-size chunks.

    The ``silence_threshold`` and ``silence_seconds`` knobs are
    accepted for Protocol parity but ignored — the file's length
    is the natural stop condition.
    """

    def __init__(
        self,
        wav_path: str,
        chunk_bytes: int = 3072,
        chunk_delay_s: float = 0.0,
    ) -> None:
        self._wav_path = wav_path
        self._chunk_bytes = chunk_bytes
        self._chunk_delay_s = chunk_delay_s
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
        return self._iter(max_duration_s)

    async def _iter(self, max_duration_s: float) -> AsyncIterator[bytes]:
        bytes_per_second = (
            self.capabilities.sample_rate_hz
            * self.capabilities.channels
            * (self.capabilities.bit_depth // 8)
        )
        max_bytes = int(max_duration_s * bytes_per_second)
        emitted = 0
        with open(self._wav_path, "rb") as f:
            while True:
                if emitted >= max_bytes:
                    break
                want = min(self._chunk_bytes, max_bytes - emitted)
                buf = f.read(want)
                if not buf:
                    break
                yield buf
                emitted += len(buf)
                if self._chunk_delay_s > 0:
                    await asyncio.sleep(self._chunk_delay_s)


class LiveMicCapture:
    """Live-mic capture via a CPython subprocess.

    Each ``listen()`` call shells out to ``capture.py`` (a
    board-sibling script staged at ``.stage/capture.py``) which
    uses ``sounddevice`` + ``numpy`` for real VAD silence-detect.
    Same pattern as ``display.py`` shelling out to ``render.py``
    for Pillow rendering — MP unix-port can't load CFFI-wrapped
    libs directly, so the host CPython does the work.

    The subprocess records from the system default mic, stops
    when ``silence_seconds`` of quiet elapse after speech (or
    ``max_duration_s`` hard cap), writes a WAV file, and exits.
    This capture then yields the WAV bytes in chunks.
    """

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        bit_depth: int = 16,
        channels: int = 1,
    ) -> None:
        self.capabilities = AudioCapabilities(
            sample_rate_hz=sample_rate_hz,
            bit_depth=bit_depth,
            channels=channels,
            format="wav",
        )

    def listen(
        self,
        max_duration_s: float,
        silence_threshold: int,
        silence_seconds: float,
    ) -> AsyncIterator[bytes]:
        # Phase 1: capture to disk SYNCHRONOUSLY via os.system.
        # This blocks the asyncio loop but that's fine — the user
        # is speaking, nothing else should run. The critical thing
        # is that capture finishes BEFORE we return the iterator,
        # because the caller opens an HTTP connection and starts
        # sending the body immediately when it iterates. If we
        # delayed capture into _iter(), the HTTP connection would
        # sit idle for 10s waiting for the mic → server EPIPE.
        try:
            stamp = time.time_ns()
        except AttributeError:
            stamp = int(time.time() * 1_000_000_000)
        tmp_dir = os.getenv("TMPDIR") or "/tmp"
        self._wav_path = "{}/exoclaw-voice-{}.wav".format(tmp_dir.rstrip("/"), stamp)

        host_python = os.getenv("EXOCLAW_HOST_PYTHON") or "python3"
        argv = [
            host_python,
            "-P",
            "capture.py",
            self._wav_path,
            str(self.capabilities.sample_rate_hz),
            str(self.capabilities.channels),
            str(max_duration_s),
            str(silence_threshold),
            str(silence_seconds),
        ]
        cmd = " ".join('"' + a + '"' for a in argv)
        rc = os.system(cmd)
        if rc != 0:
            raise RuntimeError("capture.py failed (rc={}); cmd: {}".format(rc, cmd))

        # Phase 2: return an iterator that streams from the
        # already-written file. By the time the HTTP body
        # generator pulls from this, the data is on disk and
        # yields instantly — no idle gap on the wire.
        return self._stream_from_file()

    async def _stream_from_file(self) -> AsyncIterator[bytes]:
        wav_path = self._wav_path
        # ``capture.py`` prints ``SILENCE`` and exits 0 without
        # writing a WAV when no speech was detected. Yield nothing
        # so the listener returns ``(no speech detected)`` cleanly.
        try:
            f = open(wav_path, "rb")
        except OSError:
            return

        try:
            while True:
                buf = f.read(3072)
                if not buf:
                    break
                yield buf
        finally:
            f.close()
            try:
                os.remove(wav_path)
            except OSError:
                pass
