"""Materialize codemod outputs before pytest collection.

Channel.py and test_channel.py are gitignored — derivatives of vendor/.
This conftest runs the codemod on import so `pytest packages/exoclaw-channel-X/tests/`
works from a fresh source checkout without a separate build step.
"""

from pathlib import Path

from exoclaw_channel_codemod import regenerate

regenerate(Path(__file__).resolve().parent)
