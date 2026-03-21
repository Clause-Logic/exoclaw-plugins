"""File-backed conversation implementation for exoclaw."""

from exoclaw_conversation.context import (
    compact_tool_results,
    drop_oldest_half,
)
from exoclaw_conversation.protocols import ConsolidationPolicy

__all__ = ["ConsolidationPolicy", "compact_tool_results", "drop_oldest_half"]
