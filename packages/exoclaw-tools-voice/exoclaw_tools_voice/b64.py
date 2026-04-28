"""Chunked base64 encoder — pure Python, no stdlib ``base64``
dependency on chip MP (it ships, but using our own gives us
control over chunk alignment).

The encoder maintains a 0-2 byte ``carry`` of input bytes that
didn't divide evenly into the previous chunk's groups of three.
Each ``encode(chunk)`` call concatenates carry+chunk, emits as
many full 4-char output groups as possible, and stores the
remaining 0-2 bytes for the next call. ``flush()`` emits the
final group with ``=`` padding.

Why this matters for the voice path: the LLM request body is a
streaming JSON body where the audio is a single base64 string.
We can't pad mid-stream — the decoder would treat ``=`` as
end-of-data. So we accumulate exactly the bytes that don't fit
into a full group, then flush at the very end before closing
the JSON string.
"""

from __future__ import annotations

# Standard base64 alphabet (RFC 4648 §4). MP has ``ubinascii``
# / ``base64`` in some builds but not all; this is six lines of
# code and saves a dependency check.
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


class B64StreamEncoder:
    """Stateful encoder. Call ``encode(chunk)`` zero or more
    times, then ``flush()`` once at the end."""

    __slots__ = ("_carry",)

    def __init__(self) -> None:
        self._carry: bytes = b""

    def encode(self, chunk: bytes) -> str:
        """Encode another chunk of input bytes. Returns the
        base64 chars produced; may be empty if the chunk + carry
        is shorter than 3 bytes."""
        if not chunk:
            return ""
        data = self._carry + chunk
        # Number of complete 3-byte groups in ``data``. Anything
        # past that is carry for next time.
        full = (len(data) // 3) * 3
        encoded = self._encode_full(data[:full])
        self._carry = data[full:]
        return encoded

    def flush(self) -> str:
        """Emit the final group + padding. After this the encoder
        is exhausted; further ``encode`` calls won't produce
        valid output mid-stream (the padding has been written)."""
        carry = self._carry
        self._carry = b""
        if not carry:
            return ""
        if len(carry) == 1:
            b0 = carry[0]
            return _ALPHABET[b0 >> 2] + _ALPHABET[(b0 & 0x03) << 4] + "=="
        # len == 2
        b0, b1 = carry[0], carry[1]
        return (
            _ALPHABET[b0 >> 2]
            + _ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)]
            + _ALPHABET[(b1 & 0x0F) << 2]
            + "="
        )

    @staticmethod
    def _encode_full(data: bytes) -> str:
        """Encode a buffer whose length is an exact multiple of
        3, producing 4 chars per 3 bytes with no padding."""
        out: list[str] = []
        for i in range(0, len(data), 3):
            b0, b1, b2 = data[i], data[i + 1], data[i + 2]
            out.append(_ALPHABET[b0 >> 2])
            out.append(_ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)])
            out.append(_ALPHABET[((b1 & 0x0F) << 2) | (b2 >> 6)])
            out.append(_ALPHABET[b2 & 0x3F])
        return "".join(out)
