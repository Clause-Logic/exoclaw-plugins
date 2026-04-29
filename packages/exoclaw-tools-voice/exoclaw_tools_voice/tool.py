"""``ListenAndTranscribeTool`` — the agent-facing voice surface.

When the agent calls this tool:

1. The tool opens an ``AudioCapture`` (board-supplied — unix sim
   uses ``WavFileCapture``, the E1001 chip uses ``I2SCapture``).
2. PCM bytes flow chunk-by-chunk into a streaming chat-completions
   request body, base64-encoded on the fly so the chip never
   materialises the whole recording in memory.
3. The audio model returns a transcription as plain text.
4. The tool returns that transcription as the tool result.

The agent then sees the text and proceeds normally — no audio
knowledge in the main loop. Same pattern as ``WebSearchTool``:
one dedicated model handles the modality-specific work, the
agent's main chat model stays cheap and text-only.
"""

from __future__ import annotations

from typing import Any

from exoclaw._compat import get_logger
from exoclaw.agent.tools.protocol import ToolBase
from exoclaw.providers.protocol import LLMProvider

from exoclaw_tools_voice.capture import AudioCapture
from exoclaw_tools_voice.streaming import stream_audio_request_body

logger = get_logger()

# Default prompt — the audio model sees both this text and the
# audio. Phrased as a transcription request so the model returns
# a clean text rendering of what the user said, without
# editorialising. The agent then handles intent.
_DEFAULT_PROMPT = "Transcribe the audio. Reply with only the words the user said, no commentary."


class ListenAndTranscribeTool(ToolBase):
    """Capture audio + transcribe via an audio-capable LLM."""

    def __init__(
        self,
        provider: LLMProvider,
        capture: AudioCapture,
        audio_model: str,
        max_duration_s: float = 15.0,
        silence_threshold: int = 500,
        silence_seconds: float = 1.5,
        max_tokens: int = 512,
        prompt: str = _DEFAULT_PROMPT,
    ) -> None:
        self._provider = provider
        self._capture = capture
        self._audio_model = audio_model
        self._max_duration_s = max_duration_s
        self._silence_threshold = silence_threshold
        self._silence_seconds = silence_seconds
        self._max_tokens = max_tokens
        self._prompt = prompt

    @property
    def name(self) -> str:
        return "listen"

    @property
    def description(self) -> str:
        return (
            "Listen to the user's microphone and return what they "
            "said as text. Use this when the user wants to speak "
            "instead of type, or when they ask you to listen. "
            "Recording stops automatically after a short silence "
            "or when the maximum duration is reached."
        )

    @property
    def parameters(self) -> "dict[str, Any]":
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        # Pull bytes from the capture and stream them through the
        # request body. The provider's audio-model deployment was
        # registered alongside the chat model on construction
        # (same pattern as ``web_search_model`` for OpenRouter).
        try:
            chunks = self._capture.listen(
                max_duration_s=self._max_duration_s,
                silence_threshold=self._silence_threshold,
                silence_seconds=self._silence_seconds,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("listen_capture_open_failed", **{"error": str(e)})
            return "Error: failed to open microphone: {}".format(e)

        body = stream_audio_request_body(
            model=self._audio_model,
            user_text=self._prompt,
            audio_chunks=chunks,
            audio_format=self._capture.capabilities.format,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        # ``send_streaming_body`` is the provider's escape hatch
        # for callers that build their own request body. Lives on
        # ``OpenAIStreamingProvider`` rather than the generic
        # ``LLMProvider`` Protocol since most providers' chat()
        # API is enough; only the audio path needs raw streaming
        # control.
        send = getattr(self._provider, "send_streaming_body", None)
        if send is None:
            logger.error(
                "listen_provider_missing_send_streaming_body",
                **{"llm.model": self._audio_model},
            )
            return (
                "Error: provider does not support streaming audio "
                "requests. Update exoclaw-provider-openai to a "
                "version that exposes ``send_streaming_body``."
            )
        try:
            transcription = await send(model=self._audio_model, body=body)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "listen_failed",
                **{"llm.model": self._audio_model, "error": str(e)},
            )
            return "Error: listen failed: {}".format(e)

        if not transcription:
            return "(no speech detected)"
        return transcription
