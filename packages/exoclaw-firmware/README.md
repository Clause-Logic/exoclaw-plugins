# exoclaw-firmware

Deployable exoclaw image for MicroPython boards. Bundles core
`exoclaw` + the MP-compatible plugin set (conversation, OpenAI
provider) + board-specific boot wrappers, so flashing one tree
gets you a chip that can hold a conversation with an LLM.

This is the equivalent of `exoclaw-nanobot` for the chip target —
where nanobot is "what you run on a server", `exoclaw-firmware`
is "what you flash to a board".

## Supported boards

| Board                    | RAM       | Status      | Notes                                           |
|--------------------------|-----------|-------------|-------------------------------------------------|
| ESP32-S3 (8 MB PSRAM)    | 8 MB      | Supported   | Reference target; SD card recommended.          |
| ESP32 (4 MB)             | 4 MB      | Should work | Tighter; turn off any optional plugins.         |
| Raspberry Pi Pico W      | 264 KB    | Unlikely    | RAM budget too small for an LLM agent loop.     |

Adding a new board: copy `boards/esp32_s3/` to `boards/<your_board>/`,
adapt the `boot.py` for that platform's WiFi / SD / clock APIs.

## Quickstart (ESP32-S3)

1. **Wire up an SD card** (optional but recommended for session
   storage). Default pins in `boards/esp32_s3/boot.py`:
   `sclk=12, mosi=11, miso=13, cs=10`.
2. **Flash MicroPython 1.27+** to the board (out of scope for
   this package — see the [MicroPython downloads page]).
3. **Copy `boards/esp32_s3/secrets.py.example`** to
   `boards/esp32_s3/secrets.py` and fill in `WIFI_SSID`,
   `WIFI_PASSWORD`, `OPENAI_API_KEY`.
4. **Stage and flash:**
   ```sh
   cd packages/exoclaw-firmware
   mise run flash
   ```
5. **Reset the board** (or `mise run repl` to watch output). On
   first boot you should see `boot: SD mounted`, `boot: WiFi up`,
   `boot: clock synced`, then the demo prompt's response.

## What's in the flash image

The `mise run stage` task assembles the deployable tree under
`.stage/`:

```
.stage/
├── boot.py                    # WiFi + SD + NTP setup
├── main.py                    # runs run_demo() once
├── secrets.py                 # gitignored; your credentials
├── exoclaw/                   # core (with _cpython.py stripped)
├── exoclaw_conversation/      # MP-compatible conversation plugin
├── exoclaw_provider_openai/   # MP-compatible OpenAI provider
└── exoclaw_firmware/          # this package's app.py
```

Files matching `_cpython.py` are stripped at stage time — they
hold CPython-only implementations that the MP runtime never
imports. Same trick the core MP coverage runner uses.

## Beyond the demo

`main.py` runs a single hardcoded prompt and exits. Real use needs
a channel layer:

- **HTTP webhook** — `asyncio.start_server` listening on port 80
  with a JSON `{message: ...}` POST handler that drives the agent.
- **MQTT subscriber** — `umqtt.simple` for incoming messages,
  publish replies on a paired topic.
- **Polling queue** — periodic `exoclaw.http.HTTPClient.stream_post`
  to a server-side queue endpoint that returns the next message.
- **Serial REPL** — debug-only; reads `input()` and prints replies.

Each of these is straightforward to layer on top of
`exoclaw_firmware.app.build_agent`, which gives you a
`(provider, conversation)` pair ready to drive turns.

## mise tasks

| Task             | What it does                                              |
|------------------|-----------------------------------------------------------|
| `mise run stage` | Assemble `.stage/` from core + plugins + board files.     |
| `mise run flash` | Copy `.stage/` onto the attached board via `mpremote`.    |
| `mise run repl`  | Drop into a REPL on the board.                            |
| `mise run run`   | Exec `main.py` on the board without flashing.             |
| `mise run wipe`  | Erase the board's filesystem (destructive).               |

[MicroPython downloads page]: https://micropython.org/download/
