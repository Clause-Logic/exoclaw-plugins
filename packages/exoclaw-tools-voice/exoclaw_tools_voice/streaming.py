"""Streaming chat-completions request body for audio input.

The agent records audio bytes from an ``AudioCapture`` and we
need to send them to an OpenAI-compatible audio model. Because
the chip target has ~50KB of free heap and a real recording can
be 100+ KB once base64-encoded, we cannot materialise the JSON
body in memory. Instead the body is yielded chunk-by-chunk:

1. Envelope head: ``{"model":"...","stream":true,...,"messages":[``
2. The user message prefix ending right before the audio data:
   ``{"role":"user","content":[{"type":"text","text":"..."},``
   ``{"type":"input_audio","input_audio":{"format":"wav","data":"``
3. Base64-encoded chunks of audio bytes pulled from the capture
   iterator as they're produced — one ``B64StreamEncoder.encode``
   per chunk, then a single ``flush()`` for the final padding
4. Closing: ``"}}]}],...}``

Implementation note: this is a class with ``__aiter__`` /
``__anext__`` rather than ``async def`` + ``yield``. On
MicroPython 1.27, ``async def`` with a body that yields AND
iterates a nested async-for collapses to a sync generator that
``async for`` can't drive (``'generator' object has no attribute
'__aiter__'``). Class-based iterators avoid the pitfall and
work identically on CPython.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from exoclaw._compat import aiter_compat

from exoclaw_tools_voice.b64 import B64StreamEncoder

# Sentinel state values for the streaming generator's mini state
# machine. Plain integers (not ``enum``) because MP's ``enum``
# coverage is patchy and we only have four states.
_S_HEAD = 0
_S_PREFIX = 1
_S_AUDIO = 2
_S_FLUSH = 3
_S_TAIL = 4
_S_DONE = 5


class _AudioBodyIter:
    """Async iterator that yields the JSON request body in
    chunks. See module docstring for the four-stage layout.

    Each ``__anext__`` call advances the state machine by one
    stage and returns one chunk of bytes:

    - ``_S_HEAD`` → envelope opening + ``"messages":[``
    - ``_S_PREFIX`` → message prefix up to the audio data quote
    - ``_S_AUDIO`` → one base64-encoded audio chunk (loops here
      until the source iterator is exhausted)
    - ``_S_FLUSH`` → final base64 padding for whatever bytes the
      encoder was carrying
    - ``_S_TAIL`` → closing the JSON
    - ``_S_DONE`` → ``StopAsyncIteration``
    """

    def __init__(
        self,
        *,
        model: str,
        user_text: str,
        audio_chunks: Any,
        audio_format: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self._head: dict[str, Any] = {
            "model": model,
            "stream": True,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        self._user_text = user_text
        self._audio_format = audio_format
        # ``aiter_compat`` adapts both CPython async generators
        # and MP plain generators so the caller can pass either
        # shape without thinking about runtime.
        self._audio = aiter_compat(audio_chunks)
        self._encoder = B64StreamEncoder()
        self._state = _S_HEAD

    def __aiter__(self) -> "_AudioBodyIter":
        return self

    async def __anext__(self) -> bytes:
        if self._state == _S_HEAD:
            self._state = _S_PREFIX
            head_json = json.dumps(self._head)
            # Splice ``,"messages":[`` in just before the closing
            # ``}`` of the head object.
            return (head_json[:-1] + ',"messages":[').encode("utf-8")

        if self._state == _S_PREFIX:
            self._state = _S_AUDIO
            prefix = (
                '{"role":"user","content":['
                + json.dumps({"type": "text", "text": self._user_text})
                + ',{"type":"input_audio","input_audio":{"format":'
                + json.dumps(self._audio_format)
                + ',"data":"'
            )
            return prefix.encode("utf-8")

        if self._state == _S_AUDIO:
            # Loop until we get a chunk that produces non-empty
            # encoded output. Tiny chunks below 3 bytes might
            # carry without emitting anything; keep pulling until
            # we have something to yield.
            while True:
                try:
                    pcm = await self._audio.__anext__()
                except StopAsyncIteration:
                    self._state = _S_FLUSH
                    return await self.__anext__()
                encoded = self._encoder.encode(pcm)
                if encoded:
                    return encoded.encode("ascii")

        if self._state == _S_FLUSH:
            self._state = _S_TAIL
            tail = self._encoder.flush()
            if tail:
                return tail.encode("ascii")
            # No flush bytes — fall through to tail directly.
            return await self.__anext__()

        if self._state == _S_TAIL:
            self._state = _S_DONE
            return b'"}}]}]}'

        raise StopAsyncIteration


def stream_audio_request_body(
    *,
    model: str,
    user_text: str,
    audio_chunks: AsyncIterator[bytes],
    audio_format: str,
    max_tokens: int,
    temperature: float,
) -> "_AudioBodyIter":
    """Build the streaming request body. Returns an async-
    iterable; iterate with ``async for chunk in body:``.

    Class-based (not ``async def`` + ``yield``) so it works on
    both CPython and MicroPython 1.27.
    """
    return _AudioBodyIter(
        model=model,
        user_text=user_text,
        audio_chunks=audio_chunks,
        audio_format=audio_format,
        max_tokens=max_tokens,
        temperature=temperature,
    )
