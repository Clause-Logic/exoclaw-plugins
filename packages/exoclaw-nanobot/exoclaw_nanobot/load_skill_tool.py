"""LoadSkillTool — bridges the LOAD_SKILL_TOOL_DEF schema with SkillsLoader.activate_skill."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exoclaw_conversation.context import ContextBuilder

from exoclaw_conversation import LOAD_SKILL_TOOL_DEF

_schema = LOAD_SKILL_TOOL_DEF["function"]


class LoadSkillTool:
    """Tool that lets the agent dynamically activate skills listed in <skills>."""

    name = _schema["name"]
    description = _schema["description"]
    parameters = _schema["parameters"]

    def __init__(self, prompt: ContextBuilder) -> None:
        self._prompt = prompt

    async def execute(self, *, name: str, **_: object) -> str:
        result = self._prompt.skills.activate_skill(name)
        # Merge newly activated tool names so they're visible on subsequent LLM calls
        self._prompt._active_optional_tools.update(result.tool_names)
        return result.content
