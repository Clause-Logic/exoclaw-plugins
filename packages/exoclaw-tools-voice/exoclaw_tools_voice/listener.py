"""``MicListener`` — captures audio + transcribes via the same
streaming-base64-in-JSON pipeline ``ListenAndTranscribeTool``
uses, but exposed as a SerialChannel ``line_interceptor``.

Voice as a channel-trigger rather than an agent-callable tool:
the user types ``/talk`` (or whatever ``trigger_token`` is set
to), the SerialChannel hands the line to the interceptor, and
the interceptor opens the mic, captures one utterance with
silence-detect, transcribes it via the audio model, and returns
the transcribed text. The SerialChannel publishes that text as
an inbound message — same shape as if the user had typed it.

Drop-in for any board with an ``AudioCapture``: the unix sim's
live-mic ``LiveMicCapture`` and the chip's ``I2SCapture`` both
satisfy the Protocol, so the listener doesn't care which one
it's holding.
"""

from __future__ import annotations

from typing import Any

from exoclaw._compat import get_logger

from exoclaw_tools_voice.capture import AudioCapture
from exoclaw_tools_voice.streaming import stream_audio_request_body

logger = get_logger()

_DEFAULT_TOKEN = "/talk"
_DEFAULT_PROMPT = "Transcribe the audio. Reply with only the words the user said, no commentary."


class MicListener:
    """Voice-input shim for SerialChannel.

    The listener wraps an ``AudioCapture`` + an LLM provider's
    audio-capable deployment. Its ``intercept(line)`` method is
    the SerialChannel hook: returns the transcribed utterance on
    the trigger token, ``None`` (pass through) for everything
    else.
    """

    def __init__(
        self,
        provider: Any,
        capture: AudioCapture,
        audio_model: str,
        trigger_token: str = _DEFAULT_TOKEN,
        prompt: str = _DEFAULT_PROMPT,
        max_duration_s: float = 15.0,
        silence_threshold: int = 500,
        silence_seconds: float = 1.5,
        max_tokens: int = 512,
    ) -> None:
        self._provider = provider
        self._capture = capture
        self._audio_model = audio_model
        self._trigger_token = trigger_token
        self._prompt = prompt
        self._max_duration_s = max_duration_s
        self._silence_threshold = silence_threshold
        self._silence_seconds = silence_seconds
        self._max_tokens = max_tokens

    async def intercept(self, line: str) -> "str | None":
        """SerialChannel ``line_interceptor`` callback.

        Returns the transcribed utterance when ``line`` matches
        the trigger token; ``None`` to pass the line through
        unchanged.
        """
        if line.strip() != self._trigger_token:
            return None
        return await self.listen_and_transcribe()

    async def listen_and_transcribe(self) -> str:
        """Open the mic, capture one utterance, return the text."""
        # Surface "listening" feedback before the capture blocks
        # on mic input — otherwise the user types ``/talk`` and
        # sits in apparent silence wondering whether anything is
        # happening.
        try:
            import sys

            sys.stdout.write("[listening...]\n")
            sys.stdout.flush()
        except (AttributeError, OSError):
            pass

        try:
            chunks = self._capture.listen(
                max_duration_s=self._max_duration_s,
                silence_threshold=self._silence_threshold,
                silence_seconds=self._silence_seconds,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("mic_listener_capture_open_failed", **{"error": str(e)})
            return "(mic error: {})".format(e)

        body = stream_audio_request_body(
            model=self._audio_model,
            user_text=self._prompt,
            audio_chunks=chunks,
            audio_format=self._capture.capabilities.format,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        send = getattr(self._provider, "send_streaming_body", None)
        if send is None:
            logger.error(
                "mic_listener_provider_missing_send_streaming_body",
                **{"llm.model": self._audio_model},
            )
            return "(voice unavailable: provider missing send_streaming_body)"
        try:
            transcription = await send(model=self._audio_model, body=body)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "mic_listener_failed",
                **{"llm.model": self._audio_model, "error": str(e)},
            )
            return "(transcription failed: {})".format(e)

        if not transcription:
            return ""
        return transcription
