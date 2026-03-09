"""Entry point: python -m exoclaw_nanobot or exoclaw CLI command."""

import asyncio

from exoclaw_nanobot.app import create


async def _main() -> None:
    bot = await create()
    await bot.run()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
