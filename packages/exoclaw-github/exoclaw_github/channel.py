"""GitHub Actions channel for exoclaw."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from exoclaw.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from exoclaw.bus.protocol import Bus


@dataclass
class GitHubEvent:
    kind: str                       # "issue", "pr", "dispatch"
    number: int                     # issue/PR number; 0 for dispatch
    sender: str                     # GitHub username
    body: str                       # message body to send to the agent
    repo: str                       # "owner/repo"
    title: str = ""
    reply_to_comment_id: int | None = None  # set for review comment replies
    comment_id: int | None = None           # id of the triggering comment (for reactions)
    comment_kind: str = "issue"             # "issue" or "pr_review"
    head_sha: str | None = None             # PR head commit SHA (for checks)


class GitHubChannel:
    """
    GitHub Actions channel for exoclaw.

    Reads the current GitHub Actions event from the environment, publishes it
    as an InboundMessage, then posts the agent's response as a GitHub comment.

    Implements the exoclaw Channel protocol without inheriting from any
    exoclaw class.

    Configuration via constructor args or environment variables:
      GITHUB_TOKEN       — required for posting comments
      GITHUB_EVENT_NAME  — set automatically by GitHub Actions
      GITHUB_EVENT_PATH  — set automatically by GitHub Actions
      GITHUB_REPOSITORY  — set automatically by GitHub Actions
    """

    name = "github"

    def __init__(
        self,
        token: str | None = None,
        trigger: str | None = "@exoclaw",
        respond_to_issues_opened: bool = True,
        respond_to_prs_opened: bool = False,
    ):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._trigger = trigger
        self._respond_to_issues_opened = respond_to_issues_opened
        self._respond_to_prs_opened = respond_to_prs_opened
        self._pending_event: GitHubEvent | None = None
        self._response_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Event parsing
    # ------------------------------------------------------------------

    def _parse_event(self) -> GitHubEvent | None:
        event_name = os.environ.get("GITHUB_EVENT_NAME", "")
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")
        repo = os.environ.get("GITHUB_REPOSITORY", "")

        if not event_path or not Path(event_path).exists():
            logger.warning("GITHUB_EVENT_PATH not set or file missing")
            return None

        with open(event_path) as f:
            data: dict[str, Any] = json.load(f)

        if event_name == "issues":
            return self._parse_issues_event(data, repo)
        elif event_name == "issue_comment":
            return self._parse_issue_comment_event(data, repo)
        elif event_name == "pull_request":
            return self._parse_pr_event(data, repo)
        elif event_name == "pull_request_review_comment":
            return self._parse_review_comment_event(data, repo)
        elif event_name == "workflow_dispatch":
            return self._parse_dispatch_event(data, repo)
        else:
            logger.info("Unsupported event type: {}", event_name)
            return None

    def _parse_issues_event(self, data: dict[str, Any], repo: str) -> GitHubEvent | None:
        if data.get("action") != "opened":
            return None
        if not self._respond_to_issues_opened:
            return None
        issue = data["issue"]
        body = issue.get("body") or issue.get("title", "")
        return GitHubEvent(
            kind="issue",
            number=issue["number"],
            sender=issue["user"]["login"],
            body=body,
            repo=repo,
            title=issue.get("title", ""),
        )

    def _parse_issue_comment_event(self, data: dict[str, Any], repo: str) -> GitHubEvent | None:
        if data.get("action") != "created":
            return None
        comment = data["comment"]
        body = comment.get("body", "")
        if self._trigger and self._trigger not in body:
            logger.info("Comment missing trigger '{}', skipping", self._trigger)
            return None
        issue = data["issue"]
        return GitHubEvent(
            kind="issue",
            number=issue["number"],
            sender=comment["user"]["login"],
            body=body,
            repo=repo,
            title=issue.get("title", ""),
            comment_id=comment.get("id"),
            comment_kind="issue",
        )

    def _parse_pr_event(self, data: dict[str, Any], repo: str) -> GitHubEvent | None:
        if data.get("action") != "opened":
            return None
        if not self._respond_to_prs_opened:
            return None
        pr = data["pull_request"]
        body = pr.get("body") or pr.get("title", "")
        return GitHubEvent(
            kind="pr",
            number=pr["number"],
            sender=pr["user"]["login"],
            body=body,
            repo=repo,
            title=pr.get("title", ""),
            head_sha=pr.get("head", {}).get("sha"),
        )

    def _parse_review_comment_event(self, data: dict[str, Any], repo: str) -> GitHubEvent | None:
        if data.get("action") != "created":
            return None
        comment = data["comment"]
        body = comment.get("body", "")
        if self._trigger and self._trigger not in body:
            logger.info("Review comment missing trigger '{}', skipping", self._trigger)
            return None
        pr = data["pull_request"]
        path = comment.get("path", "")
        line = comment.get("line") or comment.get("original_line", "")
        context = f"[Diff comment on `{path}` line {line}]\n{body}" if path else body
        return GitHubEvent(
            kind="pr",
            number=pr["number"],
            sender=comment["user"]["login"],
            body=context,
            repo=repo,
            title=pr.get("title", ""),
            reply_to_comment_id=comment["id"],
            comment_id=comment.get("id"),
            comment_kind="pr_review",
            head_sha=pr.get("head", {}).get("sha"),
        )

    def _parse_dispatch_event(self, data: dict[str, Any], repo: str) -> GitHubEvent | None:
        inputs = data.get("inputs") or {}
        message = inputs.get("message") or "Workflow dispatched"
        return GitHubEvent(
            kind="dispatch",
            number=0,
            sender="workflow_dispatch",
            body=message,
            repo=repo,
        )

    # ------------------------------------------------------------------
    # GitHub API
    # ------------------------------------------------------------------

    async def _post_comment(self, event: GitHubEvent, content: str) -> None:
        if event.kind == "dispatch":
            logger.info("Response (dispatch): {}", content)
            return

        if event.reply_to_comment_id is not None:
            url = (
                f"https://api.github.com/repos/{event.repo}"
                f"/pulls/comments/{event.reply_to_comment_id}/replies"
            )
        else:
            url = f"https://api.github.com/repos/{event.repo}/issues/{event.number}/comments"

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={"body": content}, headers=headers)
            resp.raise_for_status()
        logger.info("Posted comment to {}/{}", event.repo, event.number)

    # ------------------------------------------------------------------
    # Channel protocol
    # ------------------------------------------------------------------

    async def start(self, bus: Bus) -> None:
        event = self._parse_event()
        if event is None:
            logger.info("No actionable GitHub event — exiting")
            return

        self._pending_event = event
        self._response_event = asyncio.Event()

        await bus.publish_inbound(InboundMessage(
            channel=self.name,
            sender_id=event.sender,
            chat_id=str(event.number),
            content=event.body,
            session_key_override=f"github:{event.kind}:{event.number}",
            metadata={
                "repo": event.repo,
                "title": event.title,
                "kind": event.kind,
                "number": event.number,
                "comment_id": event.comment_id,
                "comment_kind": event.comment_kind,
                "head_sha": event.head_sha,
            },
        ))

        logger.info("Waiting for response to {} #{}", event.kind, event.number)
        await self._response_event.wait()

    async def stop(self) -> None:
        if self._response_event:
            self._response_event.set()

    async def send(self, msg: OutboundMessage) -> None:
        if msg.metadata.get("_progress"):
            return
        if not msg.content:
            return
        if self._pending_event:
            await self._post_comment(self._pending_event, msg.content)
        if self._response_event:
            self._response_event.set()
