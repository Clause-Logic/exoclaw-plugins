"""Wires the exoclaw stack with GitHubChannel for GitHub Actions."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.utils import create_isolated_task
from exoclaw_conversation.conversation import DefaultConversation
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


def _env_bool(key: str, default: bool = False) -> bool:
    """Read an explicit boolean environment setting without string truthiness."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(key)
    if value is None:
        return default
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or default


_MODEL_ISSUE_AUTHOR_ASSOCIATIONS = (
    "CONTRIBUTOR",
    "MEMBER",
    "OWNER",
    "COLLABORATOR",
)
_MODEL_ISSUE_SKILLS = ("decisionbench-add-model",)


async def create(
    model: str | None = None,
    state_dir: Path | None = None,
    repo_dir: Path | None = None,
    trigger: str | None = ...,  # type: ignore[assignment]
    respond_to_issues_opened: bool = True,
    respond_to_prs_opened: bool = False,
    max_tokens: int = 8192,
    max_iterations: int = 40,
    model_issue_gate: bool | None = None,
    model_issue_label: str | None = None,
    model_issue_author_associations: tuple[str, ...] | None = None,
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
        model_issue_gate: Opt into accepting only opened issues with the
            configured label and trusted author association. Defaults to
            EXOCLAW_MODEL_ISSUE_GATE (off when unset).
        model_issue_label: Exact issue label required by the model gate.
        model_issue_author_associations: Exact author associations accepted by
            the model gate.
    """
    model = model or _env("EXOCLAW_MODEL", "claude-sonnet-4-5")

    state_dir = state_dir or Path(_env("EXOCLAW_STATE_DIR", "~/.nanobot/workspace")).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = repo_dir or Path(_env("GITHUB_WORKSPACE") or os.getcwd())

    # trigger: sentinel ... means "read from env"
    if trigger is ...:
        env_val = _env("EXOCLAW_TRIGGER", "@exoclaw")
        trigger = env_val if env_val else None

    if model_issue_gate is None:
        model_issue_gate = _env_bool("EXOCLAW_MODEL_ISSUE_GATE")
    model_issue_label = model_issue_label or _env("EXOCLAW_MODEL_ISSUE_LABEL", "model")
    model_issue_author_associations = model_issue_author_associations or _env_csv(
        "EXOCLAW_MODEL_ISSUE_AUTHOR_ASSOCIATIONS", _MODEL_ISSUE_AUTHOR_ASSOCIATIONS
    )

    provider = LiteLLMProvider(default_model=model)

    bus = MessageBus()

    conversation = DefaultConversation.create(
        workspace=state_dir,
        provider=provider,
        model=model,
        # The agent's durable state is separate from the checked-out repo.
        # Project skills belong to the latter so they are versioned with the
        # workflow target and cannot be supplied through agent state.
        builtin_skills_dir=repo_dir / ".agents" / "skills",
        # The gated profile exposes its single reviewed workflow skill. Other
        # GitHub-channel deployments retain the full discovered skill surface.
        allowed_skills=list(_MODEL_ISSUE_SKILLS) if model_issue_gate else None,
    )

    tools: list[Any] = [
        ReadFileTool(workspace=repo_dir),
        WriteFileTool(workspace=repo_dir),
        EditFileTool(workspace=repo_dir),
        ListDirTool(workspace=repo_dir),
    ]
    if not model_issue_gate:
        tools.extend(
            [
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
        )

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
        model_issue_gate=model_issue_gate,
        model_issue_label=model_issue_label,
        model_issue_author_associations=model_issue_author_associations,
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
