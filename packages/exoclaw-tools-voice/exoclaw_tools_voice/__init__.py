"""Public surface for ``exoclaw-tools-voice``.

Exports:

- ``AudioCapture`` / ``AudioCapabilities`` — the Protocol seam
  boards implement. Unix sim ships ``WavFileCapture`` (in the
  firmware's ``boards/unix/`` tree); the E1001 chip ships
  ``I2SCapture``.
- ``ListenAndTranscribeTool`` — agent-callable tool that pulls
  PCM from an ``AudioCapture`` and routes it through an audio-
  capable LLM for transcription. Returns plain text so the
  agent's main chat model stays text-only.
- ``stream_audio_request_body`` — the streaming JSON body
  generator. Public so other tools (e.g. a future "voice
  conversation" path) can reuse it.
"""

from exoclaw_tools_voice.capture import AudioCapabilities, AudioCapture
from exoclaw_tools_voice.listener import MicListener
from exoclaw_tools_voice.streaming import stream_audio_request_body
from exoclaw_tools_voice.tool import ListenAndTranscribeTool

__all__ = [
    "AudioCapabilities",
    "AudioCapture",
    "ListenAndTranscribeTool",
    "MicListener",
    "stream_audio_request_body",
]
