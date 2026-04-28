"""File-backed conversation implementation for exoclaw."""

from exoclaw_conversation.context import (
    compact_tool_results,
    drop_oldest_half,
)
from exoclaw_conversation.load_skill_tool import LoadSkillTool
from exoclaw_conversation.protocols import ConsolidationPolicy
from exoclaw_conversation.skills import LOAD_SKILL_TOOL_DEF, AgentHook, LoadSkillResult
from exoclaw_conversation.summarizing_policy import SummarizingConsolidationPolicy

__all__ = [
    "AgentHook",
    "ConsolidationPolicy",
    "LOAD_SKILL_TOOL_DEF",
    "LoadSkillResult",
    "LoadSkillTool",
    "SummarizingConsolidationPolicy",
    "compact_tool_results",
    "drop_oldest_half",
]
