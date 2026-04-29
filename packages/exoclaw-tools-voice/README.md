# exoclaw-tools-voice

Voice input for exoclaw — capture microphone audio and transcribe via an
audio-capable LLM. Cross-runtime (CPython + MicroPython).

## What it does

`ListenAndTranscribeTool` is the agent surface. When the agent calls
`listen()`:

1. Opens a board-supplied `AudioCapture` (file-backed on the unix sim,
   I2S PDM mic on the reTerminal E1001).
2. Streams PCM bytes through a chunked base64 encoder into a
   chat-completions request body — never materialising the whole
   recording in memory.
3. Routes the request to a dedicated audio-capable model (e.g.
   `openai/gpt-audio-mini` on OpenRouter), separate from the agent's
   main chat model.
4. Returns the transcription as plain text.

The agent's main loop never sees audio — it just gets text back from a
tool call, the same way `web_search` returns grounded text.

## Architecture

The package is a Protocol seam:

- `AudioCapture` Protocol — what each board implements
- `ListenAndTranscribeTool` — cross-runtime tool that consumes the
  Protocol
- `stream_audio_request_body` — chunked JSON body generator
- `B64StreamEncoder` — chunked base64 with mid-stream-padding-safe
  semantics

Concrete `AudioCapture` impls live in the firmware board tree:

- `boards/unix/audio.py` — `WavFileCapture` (reads a pre-staged WAV)
- `boards/reterminal_e1001/audio.py` — `I2SCapture` (PDM mic via
  `machine.I2S`, mic-power-enable on GPIO38, button K0 trigger)

## Configuration

The firmware wires the tool when `OPENAI_AUDIO_MODEL` is set:

```toml
# mise.local.toml
[env]
OPENAI_AUDIO_MODEL = "openai/gpt-audio-mini"
```

Same pattern as `OPENAI_SEARCH_MODEL` for `web_search`. Unset → no
`listen` tool surface.
