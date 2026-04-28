"""LoadSkillTool — bridges the LOAD_SKILL_TOOL_DEF schema with
``SkillsLoader.activate_skill``.

Lives in ``exoclaw-conversation`` so any consumer (server-side
nanobot, MicroPython firmware, future deployments) can pull the
canonical implementation from one place. The previous home was
``exoclaw-nanobot``; that nanobot copy now re-exports from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exoclaw_conversation.skills import SkillsLoader

from exoclaw_conversation.skills import LOAD_SKILL_TOOL_DEF

_schema = LOAD_SKILL_TOOL_DEF["function"]


class LoadSkillTool:
    """Tool that lets the agent dynamically activate skills listed in
    the system prompt's ``<skills>`` block.

    Activating a skill merges its content into context AND merges any
    tool names the skill declares into the agent's active-tools set,
    so subsequent LLM calls see those tools available."""

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
