"""CPython-only live mic capture CLI.

Invoked by the unix board's ``LiveMicCapture`` via
``os.system`` so MicroPython unix-port (which can't load
sounddevice, a CFFI-wrapped C extension) can still capture
real audio from the laptop mic when the user types ``/talk``.

Same shell-out pattern as ``render.py`` for Pillow: the MP
process shells out to a CPython subprocess; this CLI does the
platform-specific work, exits.

CLI:
    python3 -P capture.py \\
        <out_path> <sample_rate> <channels> <max_duration_s> \\
        <silence_threshold> <silence_seconds>

All args are positional scalars — no JSON, no shlex quoting
needed on the MP side.

Output: writes a WAV file to ``<out_path>``. Prints
``CAPTURED <bytes>`` on success, ``SILENCE`` if no speech was
detected before the silence timeout, or ``ERROR <msg>`` on
failure.
"""

from __future__ import annotations

import struct
import sys
import time


def _wav_header(sample_rate: int, bit_depth: int, channels: int, data_bytes: int) -> bytes:
    byte_rate = sample_rate * channels * bit_depth // 8
    block_align = channels * bit_depth // 8
    return (
        b"RIFF"
        + struct.pack("<I", data_bytes + 36)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", channels)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", bit_depth)
        + b"data"
        + struct.pack("<I", data_bytes)
    )


def main(argv: list[str]) -> int:
    if len(argv) < 7:
        sys.stderr.write(
            "usage: capture.py out_path sample_rate channels "
            "max_duration_s silence_threshold silence_seconds\n"
        )
        return 2

    out_path = argv[1]
    sample_rate = int(argv[2])
    channels = int(argv[3])
    max_duration_s = float(argv[4])
    silence_threshold = int(argv[5])
    silence_seconds = float(argv[6])

    # Remove cwd from sys.path so the MP-stub ``datetime.py``
    # / ``typing.py`` in ``.stage/`` don't shadow real stdlib
    # modules that numpy depends on.
    import os as _os

    cwd = _os.getcwd()
    sys.path = [p for p in sys.path if p not in ("", ".", cwd)]

    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        print("ERROR: {}".format(e))
        return 1

    chunk_ms = 64
    frames_per_chunk = max(1, sample_rate * chunk_ms // 1000)
    max_frames = int(max_duration_s * sample_rate)

    all_frames: list[bytes] = []
    total_frames = 0
    silent_streak_ms = 0.0
    seen_voice = False

    def _callback(indata, frames, time_info, status) -> None:
        nonlocal total_frames, silent_streak_ms, seen_voice

        buf = bytes(indata)
        all_frames.append(buf)
        total_frames += frames

        samples = np.frombuffer(buf, dtype=np.int16)
        peak = int(np.max(np.abs(samples)))

        # Log every 8th chunk (~500ms) so the user can see
        # actual mic levels and tune the threshold.
        if total_frames % (frames_per_chunk * 8) < frames_per_chunk:
            sys.stderr.write(
                "capture: peak={} thr={} voice={} silent={:.0f}ms\n".format(
                    peak, silence_threshold, seen_voice, silent_streak_ms
                )
            )
            sys.stderr.flush()

        if peak >= silence_threshold:
            seen_voice = True
            silent_streak_ms = 0.0
        else:
            silent_streak_ms += chunk_ms

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        blocksize=frames_per_chunk,
        callback=_callback,
    )

    t0 = time.monotonic()
    with stream:
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= max_duration_s:
                break
            if total_frames >= max_frames:
                break
            if seen_voice and silent_streak_ms / 1000.0 >= silence_seconds:
                break
            time.sleep(0.05)

    if not all_frames or not seen_voice:
        print("SILENCE")
        return 0

    pcm = b"".join(all_frames)
    header = _wav_header(sample_rate, 16, channels, len(pcm))
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(pcm)
    print("CAPTURED {}".format(len(header) + len(pcm)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
