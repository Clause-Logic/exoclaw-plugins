"""LoadSkillTool — bridges the LOAD_SKILL_TOOL_DEF schema with SkillsLoader.activate_skill."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exoclaw_conversation.skills import SkillsLoader

from exoclaw_conversation import LOAD_SKILL_TOOL_DEF

_schema = LOAD_SKILL_TOOL_DEF["function"]


class LoadSkillTool:
    """Tool that lets the agent dynamically activate skills listed in <skills>."""

    name = _schema["name"]
    description = _schema["description"]
    parameters = _schema["parameters"]

    def __init__(self, skills: SkillsLoader, active_tools: set[str]) -> None:
        self._skills = skills
        self._active_tools = active_tools

    async def execute(self, *, name: str, **_: object) -> str:
        result = self._skills.activate_skill(name)
        # Merge newly activated tool names so they're visible on subsequent LLM calls
        self._active_tools.update(result.tool_names)
        return result.content
