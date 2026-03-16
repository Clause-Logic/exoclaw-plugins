"""Entry point: python -m exoclaw_github"""

import asyncio

from exoclaw_github.app import run


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
