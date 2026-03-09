# exoclaw-channel-cli

Interactive CLI REPL channel implementing the exoclaw `Channel` protocol — rich markdown output, persistent input history, and clean terminal handling.

## Install

```
pip install exoclaw-channel-cli
```

## Usage

```python
from exoclaw_channel_cli.channel import CLIChannel

cli = CLIChannel(
    chat_id="direct",
    render_markdown=True,
)

await cli.start(bus)  # runs the interactive REPL until the user types exit or Ctrl+C
```

`CLIChannel` reads from stdin using `prompt_toolkit` (with file history at `~/.exoclaw/history/cli_history`), prints responses with `rich`, and publishes/consumes messages on the exoclaw `Bus`.
