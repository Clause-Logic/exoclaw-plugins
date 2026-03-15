# exoclaw-channel-pipe

Non-interactive pipe channel for exoclaw. Reads lines from stdin, writes responses to stdout. No terminal required.

## Usage

```bash
# One-shot
echo "what is 2+2?" | python app.py

# Heredoc
python app.py <<< "summarize this repo"

# Multi-turn from file
cat prompts.txt | python app.py

# In scripts
echo "fetch top HN stories" | python -c "
import asyncio
from exoclaw_nanobot import create
from exoclaw_channel_pipe import PipeChannel

async def main():
    bot = await create(enable_cli=False, extra_channels=[PipeChannel()])
    await bot.run()

asyncio.run(main())
"
```

Progress messages go to stderr. Responses go to stdout. Clean for piping.
