"""Spawn tool for creating background subagents."""

import json
from typing import Any, Protocol, runtime_checkable

from exoclaw.agent.tools.protocol import ToolBase, ToolContext


@runtime_checkable
class SpawnManager(Protocol):
    """Protocol for subagent lifecycle management."""

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        batch: str | None = None,
        skills: list[str] | None = None,
        model: str | None = None,
    ) -> str: ...

    def get_status(self) -> dict: ...
    def list_results(self, limit: int = 20) -> list[dict[str, str]]: ...


class SpawnTool(ToolBase):
    """Tool to spawn a subagent for background task execution.

    ``allowed_models`` optionally restricts the set of models the agent
    may request via the ``model`` parameter. When set, the allowlist is
    advertised directly in the tool schema as an ``enum`` so the LLM
    sees valid choices up front, and any request outside the list is
    rejected at the tool boundary. ``None`` disables the check and
    allows any model string to pass through (backwards compatible).
    """

    def __init__(
        self,
        manager: SpawnManager,
        allowed_models: list[str] | None = None,
    ):
        self._manager = manager
        self._allowed_models = allowed_models
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._parent_skills: list[str] | None = None

    def set_context(
        self,
        channel: str,
        chat_id: str,
        session_key: str | None = None,
        skills: list[str] | None = None,
    ) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = session_key or f"{channel}:{chat_id}"
        self._parent_skills = skills

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done. "
            "Set action to 'status' to check running subagents, or 'results' to list completed results."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema: dict[str, Any] = {
            "type": "string",
            "description": (
                "Optional model override for this subagent. "
                "Omit to use the manager's default model."
            ),
        }
        if self._allowed_models is not None:
            model_schema["enum"] = list(self._allowed_models)
            model_schema["description"] = (
                "Optional model override for this subagent. Must be one of: "
                + ", ".join(self._allowed_models)
                + ". Omit to use the manager's default model."
            )

        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["spawn", "status", "results"],
                    "description": "Action to perform. 'spawn' (default) to start a subagent, "
                    "'status' to check running subagents and batch progress, "
                    "'results' to list completed subagent result files.",
                },
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete (required for 'spawn')",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
                "batch": {
                    "type": "string",
                    "description": "Optional batch ID. When set, results are held until all "
                    "subagents with the same batch ID complete, then announced together. "
                    "Use the same batch value for related parallel tasks.",
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional skill names to load in the subagent. "
                    "If not provided, the subagent inherits the parent's active skills.",
                },
                "model": model_schema,
            },
            "required": [],
        }

    def _validate_model(self, model: str | None) -> str | None:
        """Return an error string if ``model`` violates the allowlist, else None."""
        if model is None or self._allowed_models is None:
            return None
        if model not in self._allowed_models:
            allowed = ", ".join(self._allowed_models)
            return f"Error: model '{model}' is not in the allowlist. Allowed models: {allowed}."
        return None

    async def execute_with_context(
        self,
        ctx: ToolContext,
        action: str = "spawn",
        task: str | None = None,
        label: str | None = None,
        batch: str | None = None,
        skills: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute spawn tool with context."""
        if action == "status":
            return json.dumps(self._manager.get_status(), indent=2)
        if action == "results":
            return json.dumps(self._manager.list_results(), indent=2)
        if not task:
            return "Error: 'task' is required for spawn action."
        error = self._validate_model(model)
        if error is not None:
            return error
        resolved_skills = skills if skills is not None else self._parent_skills
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=ctx.channel,
            origin_chat_id=ctx.chat_id,
            session_key=ctx.session_key,
            batch=batch,
            skills=resolved_skills,
            model=model,
        )

    async def execute(
        self,
        action: str = "spawn",
        task: str | None = None,
        label: str | None = None,
        batch: str | None = None,
        skills: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute spawn tool."""
        if action == "status":
            return json.dumps(self._manager.get_status(), indent=2)
        if action == "results":
            return json.dumps(self._manager.list_results(), indent=2)
        if not task:
            return "Error: 'task' is required for spawn action."
        error = self._validate_model(model)
        if error is not None:
            return error
        resolved_skills = skills if skills is not None else self._parent_skills
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
            batch=batch,
            skills=resolved_skills,
            model=model,
        )
