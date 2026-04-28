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
    """Live-mic capture via an ``ffmpeg`` subprocess.

    Each ``listen()`` call:

    1. Picks a temp WAV path under ``$TMPDIR``.
    2. Runs ``ffmpeg -f avfoundation -i :<device>`` to record from
       the system default mic. ``silenceremove`` filter trims
       leading silence + stops on ``silence_seconds`` of trailing
       silence, capped by ``-t max_duration_s``.
    3. After ffmpeg exits, opens the resulting WAV file and yields
       its bytes in chunks via the same path as ``WavFileCapture``.

    Why ffmpeg over a Python audio lib: the unix-port runs under
    MicroPython for runtime parity with the chip. MP can't load
    CFFI-wrapped libs (sounddevice, PyAudio). ``os.system`` is
    available on MP unix-port and ffmpeg ships with most dev
    setups via Homebrew. Same shell-out pattern as the screen
    package's ``host_render`` for Pillow.
    """

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        bit_depth: int = 16,
        channels: int = 1,
        ffmpeg_bin: str = "ffmpeg",
        avfoundation_device: str = ":0",
    ) -> None:
        self.capabilities = AudioCapabilities(
            sample_rate_hz=sample_rate_hz,
            bit_depth=bit_depth,
            channels=channels,
            format="wav",
        )
        self._ffmpeg_bin = ffmpeg_bin
        # ``:0`` is the macOS avfoundation default audio input.
        # Override with ``EXOCLAW_AVFOUNDATION_DEVICE`` env var if
        # the system has multiple inputs and you want a specific
        # one (run ``ffmpeg -f avfoundation -list_devices true -i ""``
        # to enumerate).
        self._device = os.getenv("EXOCLAW_AVFOUNDATION_DEVICE") or avfoundation_device

    def listen(
        self,
        max_duration_s: float,
        silence_threshold: int,
        silence_seconds: float,
    ) -> AsyncIterator[bytes]:
        return self._iter(max_duration_s, silence_threshold, silence_seconds)

    async def _iter(
        self,
        max_duration_s: float,
        silence_threshold: int,
        silence_seconds: float,
    ) -> AsyncIterator[bytes]:
        # Map int16 ``silence_threshold`` (0..32767) to ffmpeg's
        # dB notation. -30 dBFS ≈ amplitude ~1000 — reasonable
        # default for indoor speech vs. fan hum. Caller can
        # override via the threshold knob.
        # Convert: amplitude → dBFS = 20 * log10(amp / 32767).
        # For threshold=500 → ~-36 dBFS. For 1000 → ~-30 dBFS.
        if silence_threshold > 0:
            import math

            db = 20.0 * math.log10(max(1, silence_threshold) / 32767.0)
            stop_threshold_db = "{:.1f}dB".format(db)
        else:
            stop_threshold_db = "-30dB"

        # Pick a temp WAV path. ``time.time_ns`` exists on both
        # CPython and MP; falls back to ``time.time`` if not.
        try:
            stamp = time.time_ns()
        except AttributeError:
            stamp = int(time.time() * 1_000_000_000)
        tmp_dir = os.getenv("TMPDIR") or "/tmp"
        wav_path = "{}/exoclaw-voice-{}.wav".format(tmp_dir.rstrip("/"), stamp)

        # ``-y`` overwrite, ``-loglevel error`` quiet, ``-nostdin``
        # so the child doesn't fight with the SerialChannel for
        # stdin. The ``silenceremove`` filter strips leading
        # silence (start_periods=1 + start_silence=0.5 — wait up
        # to half a second for speech), then stops once
        # ``silence_seconds`` of trailing silence below
        # ``stop_threshold_db`` elapse.
        cmd = (
            '{ffmpeg} -nostdin -y -loglevel error -f avfoundation -i "{dev}" '
            "-ac {ch} -ar {sr} -sample_fmt s16 -t {max_dur} "
            '-af "silenceremove=start_periods=1:start_silence=0.5:'
            "start_threshold={thr}:stop_periods=1:stop_silence={ss}:"
            'stop_threshold={thr}" {out}'
        ).format(
            ffmpeg=self._ffmpeg_bin,
            dev=self._device,
            ch=self.capabilities.channels,
            sr=self.capabilities.sample_rate_hz,
            max_dur=int(max_duration_s),
            thr=stop_threshold_db,
            ss=silence_seconds,
            out=wav_path,
        )

        rc = os.system(cmd)
        if rc != 0:
            # Surface the failure as a synthetic empty WAV so the
            # listener returns "(no speech detected)" rather than
            # blowing up the agent loop.
            raise RuntimeError("ffmpeg recording failed (rc={}); cmd: {}".format(rc, cmd))

        try:
            f = open(wav_path, "rb")
        except OSError as e:
            raise RuntimeError("failed to read recorded WAV: {}".format(e)) from e

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
