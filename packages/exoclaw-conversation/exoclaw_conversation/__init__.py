"""File-backed conversation implementation for exoclaw."""

from exoclaw_conversation.context import (
    compact_tool_results,
    drop_oldest_half,
)

__all__ = ["compact_tool_results", "drop_oldest_half"]
