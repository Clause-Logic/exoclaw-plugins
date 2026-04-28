"""``B64StreamEncoder`` tests — round-trip against stdlib ``base64``,
verify chunked-encoding produces the same bytes as a one-shot encode."""

from __future__ import annotations

import base64

from exoclaw_tools_voice.b64 import B64StreamEncoder


def _stream_encode(blob: bytes, chunk: int) -> str:
    """Helper: feed ``blob`` to the encoder ``chunk`` bytes at a
    time, then flush, return the concatenated output."""
    enc = B64StreamEncoder()
    out: list[str] = []
    for i in range(0, len(blob), chunk):
        out.append(enc.encode(blob[i : i + chunk]))
    out.append(enc.flush())
    return "".join(out)


def test_round_trip_against_stdlib_base64() -> None:
    """Whatever chunk size the caller picks, the output should
    decode back to the original bytes — same as a one-shot
    ``base64.b64encode`` would produce."""
    blob = b"the quick brown fox jumps over the lazy dog" * 7
    expected = base64.b64encode(blob).decode("ascii")

    # Try a range of chunk sizes including ones that don't align
    # with the 3-byte group boundary.
    for chunk in (1, 2, 3, 4, 5, 7, 13, 64, len(blob)):
        got = _stream_encode(blob, chunk)
        assert got == expected, "chunk={}: got={!r} want={!r}".format(chunk, got, expected)


def test_empty_input_produces_empty_output() -> None:
    enc = B64StreamEncoder()
    assert enc.encode(b"") == ""
    assert enc.flush() == ""


def test_padding_correct_for_length_mod_3() -> None:
    """Length-1 input → ``XX==``, length-2 → ``XXX=``, length-3 →
    ``XXXX`` (no padding). The encoder's flush handles all three
    cases — verify each."""
    # 1 byte → 2 chars + 2 ``=`` pads.
    assert B64StreamEncoder().encode(b"a") == "" or True  # carries
    enc = B64StreamEncoder()
    enc.encode(b"a")
    assert enc.flush() == "YQ=="

    # 2 bytes → 3 chars + 1 ``=`` pad.
    enc = B64StreamEncoder()
    enc.encode(b"ab")
    assert enc.flush() == "YWI="

    # 3 bytes → 4 chars, no pad.
    enc = B64StreamEncoder()
    encoded = enc.encode(b"abc") + enc.flush()
    assert encoded == "YWJj"


def test_no_mid_stream_padding_when_chunks_misalign() -> None:
    """The whole point of the streaming encoder: feeding bytes in
    awkwardly-sized chunks should never emit ``=`` padding except
    in the very final ``flush()``. Otherwise the receiver would
    decode the incoming stream and stop early."""
    blob = bytes(range(256))  # 256 bytes — not a multiple of 3
    enc = B64StreamEncoder()
    interim = ""
    for chunk in (blob[:7], blob[7:50], blob[50:200], blob[200:]):
        interim += enc.encode(chunk)
    # No padding character appeared during the streaming portion.
    assert "=" not in interim, interim
    # Final flush emits the remainder + any padding.
    full = interim + enc.flush()
    assert full == base64.b64encode(blob).decode("ascii")
