"""File-backed conversation implementation for exoclaw."""

from exoclaw_conversation.context import (
    compact_tool_results,
    drop_oldest_half,
)
from exoclaw_conversation.protocols import ConsolidationPolicy
from exoclaw_conversation.summarizing_policy import SummarizingConsolidationPolicy

__all__ = [
    "ConsolidationPolicy",
    "SummarizingConsolidationPolicy",
    "compact_tool_results",
    "drop_oldest_half",
]
