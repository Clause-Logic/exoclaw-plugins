"""Wires the exoclaw stack with GitHubChannel for GitHub Actions."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, cast

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.utils import create_isolated_task
from exoclaw_conversation.context import ContextBuilder
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_conversation.load_skill_tool import LoadSkillTool
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_tools_workspace.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from exoclaw_tools_workspace.shell import ExecTool

from exoclaw_github.channel import GitHubChannel
from exoclaw_github.tools import (
    GitHubChecksTool,
    GitHubFileTool,
    GitHubIssueTool,
    GitHubLabelTool,
    GitHubPRDiffTool,
    GitHubReactionTool,
    GitHubReviewTool,
    GitHubSearchTool,
)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_csv(key: str) -> tuple[str, ...] | None:
    value = os.environ.get(key)
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


async def create(
    model: str | None = None,
    state_dir: Path | None = None,
    repo_dir: Path | None = None,
    trigger: str | None = ...,  # type: ignore[assignment]
    respond_to_issues_opened: bool = True,
    respond_to_prs_opened: bool = False,
    max_tokens: int = 8192,
    max_iterations: int = 40,
    allowed_events: tuple[str, ...] | None = None,
    issue_label: str | None = None,
    issue_author_associations: tuple[str, ...] | None = None,
    skills_dir: Path | None = None,
    allowed_skills: tuple[str, ...] | None = None,
    allowed_tools: tuple[str, ...] | None = None,
) -> tuple[AgentLoop, GitHubChannel, MessageBus]:
    """
    Create a fully wired exoclaw stack for GitHub Actions.

    Args:
        model: LLM model string (default: EXOCLAW_MODEL env var or claude-sonnet-4-5).
        state_dir: Where sessions and memory are persisted (default: ~/.nanobot/workspace).
            In a GitHub Actions workflow, check out the bot-state branch here before
            running and commit it back afterwards.
        repo_dir: Root of the checked-out repository for file/shell tools
            (default: GITHUB_WORKSPACE env var or current directory).
        trigger: Word that must appear in comments to trigger the bot.
            Defaults to EXOCLAW_TRIGGER env var, then "@exoclaw". Pass None to respond
            to all comments.
        respond_to_issues_opened: Whether to respond when an issue is opened.
        respond_to_prs_opened: Whether to respond when a PR is opened.
        max_tokens: Maximum tokens per LLM response.
        max_iterations: Maximum tool-call iterations per turn.
        allowed_events: GitHub event names accepted by the channel.
        issue_label: Exact label required on opened issues.
        issue_author_associations: Author associations accepted for opened issues.
        skills_dir: Optional directory of deployment skills, relative to repo_dir.
        allowed_skills: Skill names visible to the agent.
        allowed_tools: Tool names registered with the agent.
        Unset filters preserve the existing behavior. Empty allowlists deny all.
    """
    model = model or _env("EXOCLAW_MODEL", "claude-sonnet-4-5")

    state_dir = state_dir or Path(_env("EXOCLAW_STATE_DIR", "~/.nanobot/workspace")).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = repo_dir or Path(_env("GITHUB_WORKSPACE") or os.getcwd())

    # trigger: sentinel ... means "read from env"
    if trigger is ...:
        env_val = _env("EXOCLAW_TRIGGER", "@exoclaw")
        trigger = env_val if env_val else None

    if allowed_events is None:
        allowed_events = _env_csv("EXOCLAW_ALLOWED_EVENTS")
    if issue_label is None:
        issue_label = os.environ.get("EXOCLAW_ISSUE_LABEL")
    if issue_author_associations is None:
        issue_author_associations = _env_csv("EXOCLAW_ISSUE_AUTHOR_ASSOCIATIONS")
    if skills_dir is None and (configured_dir := os.environ.get("EXOCLAW_SKILLS_DIR")):
        skills_dir = Path(configured_dir).expanduser()
    if skills_dir is not None and not skills_dir.is_absolute():
        skills_dir = repo_dir / skills_dir
    if allowed_skills is None:
        allowed_skills = _env_csv("EXOCLAW_ALLOWED_SKILLS")
    if allowed_tools is None:
        allowed_tools = _env_csv("EXOCLAW_ALLOWED_TOOLS")

    provider = LiteLLMProvider(default_model=model)

    bus = MessageBus()

    conversation = DefaultConversation.create(
        workspace=state_dir,
        provider=provider,
        model=model,
        builtin_skills_dir=skills_dir,
        allowed_skills=list(allowed_skills) if allowed_skills is not None else None,
    )
    prompt = cast(ContextBuilder, conversation.prompt)
    if allowed_skills is not None:
        discovered = {skill["name"]: skill for skill in prompt.skills.list_skills()}
        missing = set(allowed_skills) - discovered.keys()
        if missing:
            raise ValueError(f"Allowed skills not found: {', '.join(sorted(missing))}")
        if skills_dir is not None:
            shadowed = {name for name in allowed_skills if discovered[name]["source"] != "builtin"}
            if shadowed:
                raise ValueError(
                    "Allowed skills are not loaded from the configured skills directory: "
                    + ", ".join(sorted(shadowed))
                )

    tools: list[Any] = [
        ReadFileTool(workspace=repo_dir),
        WriteFileTool(workspace=repo_dir),
        EditFileTool(workspace=repo_dir),
        ListDirTool(workspace=repo_dir),
        ExecTool(working_dir=str(repo_dir)),
        GitHubReviewTool(),
        GitHubLabelTool(),
        GitHubPRDiffTool(),
        GitHubIssueTool(),
        GitHubReactionTool(),
        GitHubFileTool(),
        GitHubChecksTool(),
        GitHubSearchTool(),
    ]
    if skills_dir is not None or allowed_skills is not None:
        tools.insert(
            0,
            LoadSkillTool(
                skills=prompt.skills,
                active_tools=prompt._active_optional_tools,
            ),
        )
    if allowed_tools is not None:
        available = {tool.name for tool in tools}
        unknown = set(allowed_tools) - available
        if unknown:
            raise ValueError(f"Unknown GitHub agent tools: {', '.join(sorted(unknown))}")
        if allowed_skills and "load_skill" not in allowed_tools:
            raise ValueError("load_skill is required when allowed skills are configured")
        permitted = set(allowed_tools)
        tools = [tool for tool in tools if tool.name in permitted]

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=conversation,
        model=model,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        tools=tools,
    )

    channel = GitHubChannel(
        trigger=trigger,
        respond_to_issues_opened=respond_to_issues_opened,
        respond_to_prs_opened=respond_to_prs_opened,
        allowed_events=allowed_events,
        issue_label=issue_label,
        issue_author_associations=issue_author_associations,
    )

    return agent_loop, channel, bus


async def run() -> None:
    """Create the stack and run one GitHub Actions turn."""
    agent_loop, channel, bus = await create()
    loop_task = create_isolated_task(agent_loop.run())
    try:
        await channel.start(bus)
    finally:
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)
