"""GitHub-specific tools for exoclaw agents."""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger


def _gh_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


from exoclaw.agent.tools.protocol import ToolBase

if TYPE_CHECKING:
    from exoclaw.bus.events import InboundMessage


class GitHubReviewTool(ToolBase):
    """
    Submit a review on a GitHub pull request.

    Supports APPROVE, REQUEST_CHANGES, and COMMENT actions, with optional
    inline diff comments on specific file lines.

    Implements the exoclaw Tool protocol. Uses on_inbound() to capture the
    current PR number from the triggering event so the LLM doesn't have to
    specify it for the common case.
    """

    def __init__(self, token: str | None = None, repo: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
        self._pr_number: int | None = None

    def on_inbound(self, msg: InboundMessage) -> None:
        """Capture PR number from the triggering GitHub event."""
        if msg.metadata.get("kind") == "pr":
            self._pr_number = int(msg.chat_id)

    def system_context(self) -> str | None:
        if self._pr_number and self._repo:
            return f"Current pull request: #{self._pr_number} in {self._repo}"
        return None

    @property
    def name(self) -> str:
        return "github_review"

    @property
    def description(self) -> str:
        return (
            "Submit a review on the current GitHub pull request. "
            "Use to approve, request changes, or leave a review with inline comments. "
            "For simple replies, prefer posting a regular comment instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "event": {
                    "type": "string",
                    "enum": ["APPROVE", "REQUEST_CHANGES", "COMMENT"],
                    "description": "Review action to submit",
                },
                "body": {
                    "type": "string",
                    "description": "Overall review summary comment",
                },
                "pull_number": {
                    "type": "integer",
                    "description": "PR number — omit to use the current PR",
                },
                "comments": {
                    "type": "array",
                    "description": "Optional inline comments on specific diff lines",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path relative to repo root",
                            },
                            "line": {
                                "type": "integer",
                                "description": "Line number in the file",
                            },
                            "body": {
                                "type": "string",
                                "description": "Comment text for this line",
                            },
                        },
                        "required": ["path", "line", "body"],
                    },
                },
            },
            "required": ["event", "body"],
        }

    async def execute(
        self,
        event: str,
        body: str,
        pull_number: int | None = None,
        comments: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        number = pull_number or self._pr_number
        if not number:
            return "Error: no pull request number — trigger from a PR event or pass pull_number"
        if not self._repo:
            return "Error: GITHUB_REPOSITORY not set"

        url = f"https://api.github.com/repos/{self._repo}/pulls/{number}/reviews"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload: dict[str, Any] = {"event": event, "body": body}
        if comments:
            payload["comments"] = comments

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()

        n_inline = len(comments) if comments else 0
        logger.info("Submitted {} review on PR #{} ({} inline comments)", event, number, n_inline)
        detail = f" with {n_inline} inline comment(s)" if n_inline else ""
        return f"Submitted {event} review on PR #{number}{detail}"


class GitHubLabelTool(ToolBase):
    """
    Add or remove labels on the current issue or PR, or list available labels.

    Implements the exoclaw Tool protocol.
    """

    def __init__(self, token: str | None = None, repo: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
        self._number: int | None = None

    def on_inbound(self, msg: "InboundMessage") -> None:
        number = msg.metadata.get("number")
        if number is not None:
            self._number = int(number)

    @property
    def name(self) -> str:
        return "github_label"

    @property
    def description(self) -> str:
        return "Add or remove labels on the current issue or PR. Also lists all available repo labels."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "list"],
                    "description": "Action to perform",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Label names (for add/remove)",
                },
                "number": {
                    "type": "integer",
                    "description": "Issue/PR number — omit to use current",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        labels: list[str] | None = None,
        number: int | None = None,
        **kwargs: Any,
    ) -> str:
        n = number or self._number
        base = f"https://api.github.com/repos/{self._repo}"

        if action == "list":
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base}/labels", headers=_gh_headers(self._token))
                resp.raise_for_status()
            names = [lb["name"] for lb in resp.json()]
            return f"Available labels: {', '.join(names)}" if names else "No labels defined"

        if not n:
            return "Error: no issue/PR number"

        if action == "add":
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{base}/issues/{n}/labels",
                    json={"labels": labels or []},
                    headers=_gh_headers(self._token),
                )
                resp.raise_for_status()
            return f"Added {labels} to #{n}"

        if action == "remove":
            async with httpx.AsyncClient() as client:
                for label in labels or []:
                    resp = await client.delete(
                        f"{base}/issues/{n}/labels/{label}",
                        headers=_gh_headers(self._token),
                    )
                    resp.raise_for_status()
            return f"Removed {labels} from #{n}"

        return f"Unknown action: {action}"


class GitHubPRDiffTool(ToolBase):
    """
    Fetch the full unified diff for a pull request.

    Implements the exoclaw Tool protocol.
    """

    _MAX_DIFF_CHARS = 50_000

    def __init__(self, token: str | None = None, repo: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
        self._pr_number: int | None = None

    def on_inbound(self, msg: "InboundMessage") -> None:
        if msg.metadata.get("kind") == "pr":
            self._pr_number = int(msg.chat_id)

    @property
    def name(self) -> str:
        return "github_pr_diff"

    @property
    def description(self) -> str:
        return "Fetch the full unified diff for a pull request. Use before reviewing code changes."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pull_number": {
                    "type": "integer",
                    "description": "PR number — omit to use current PR",
                },
            },
        }

    async def execute(self, pull_number: int | None = None, **kwargs: Any) -> str:
        n = pull_number or self._pr_number
        if not n:
            return "Error: no PR number — trigger from a PR event or pass pull_number"

        url = f"https://api.github.com/repos/{self._repo}/pulls/{n}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github.diff",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        diff = resp.text
        if len(diff) > self._MAX_DIFF_CHARS:
            diff = diff[: self._MAX_DIFF_CHARS] + "\n... (diff truncated)"
        return diff or "(empty diff)"


class GitHubIssueTool(ToolBase):
    """
    Create, update, close, or get GitHub issues.

    Implements the exoclaw Tool protocol.
    """

    def __init__(self, token: str | None = None, repo: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repo = repo or os.environ.get("GITHUB_REPOSITORY", "")

    @property
    def name(self) -> str:
        return "github_issue"

    @property
    def description(self) -> str:
        return "Create, update, close, or get GitHub issues in this repository."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "close", "get"],
                    "description": "Action to perform",
                },
                "number": {
                    "type": "integer",
                    "description": "Issue number (required for update/close/get)",
                },
                "title": {"type": "string", "description": "Issue title (for create/update)"},
                "body": {"type": "string", "description": "Issue body (for create/update)"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to apply (for create)",
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed"],
                    "description": "Issue state (for update)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        number: int | None = None,
        title: str | None = None,
        body: str | None = None,
        labels: list[str] | None = None,
        state: str | None = None,
        **kwargs: Any,
    ) -> str:
        base = f"https://api.github.com/repos/{self._repo}/issues"

        if action == "create":
            payload: dict[str, Any] = {"title": title or "", "body": body or ""}
            if labels:
                payload["labels"] = labels
            async with httpx.AsyncClient() as client:
                resp = await client.post(base, json=payload, headers=_gh_headers(self._token))
                resp.raise_for_status()
                data = resp.json()
            return f"Created issue #{data['number']}: {data['html_url']}"

        if not number:
            return "Error: number is required for update/close/get"

        if action == "get":
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base}/{number}", headers=_gh_headers(self._token))
                resp.raise_for_status()
                data = resp.json()
            return f"#{data['number']} [{data['state']}] {data['title']}\n\n{data.get('body') or ''}"

        if action in ("update", "close"):
            patch: dict[str, Any] = {}
            if title:
                patch["title"] = title
            if body:
                patch["body"] = body
            patch["state"] = state if action == "update" and state else "closed"
            async with httpx.AsyncClient() as client:
                resp = await client.patch(
                    f"{base}/{number}", json=patch, headers=_gh_headers(self._token)
                )
                resp.raise_for_status()
            return f"Updated issue #{number}"

        return f"Unknown action: {action}"


class GitHubReactionTool(ToolBase):
    """
    Add a reaction emoji to the triggering comment.

    Useful for acknowledging a comment before a long-running response.
    Implements the exoclaw Tool protocol.
    """

    def __init__(self, token: str | None = None, repo: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
        self._comment_id: int | None = None
        self._comment_kind: str = "issue"

    def on_inbound(self, msg: "InboundMessage") -> None:
        comment_id = msg.metadata.get("comment_id")
        if comment_id is not None:
            self._comment_id = int(comment_id)
            self._comment_kind = str(msg.metadata.get("comment_kind", "issue"))

    @property
    def name(self) -> str:
        return "github_reaction"

    @property
    def description(self) -> str:
        return (
            "Add a reaction emoji to the triggering comment. "
            "Use 'eyes' to acknowledge you've seen a request, '+1' to agree."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "enum": ["+1", "-1", "laugh", "confused", "heart", "hooray", "rocket", "eyes"],
                    "description": "Reaction emoji to add",
                },
                "comment_id": {
                    "type": "integer",
                    "description": "Comment ID — omit to react to the triggering comment",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        comment_id: int | None = None,
        **kwargs: Any,
    ) -> str:
        cid = comment_id or self._comment_id
        if not cid:
            return "Error: no comment ID — not triggered by a comment event"

        if self._comment_kind == "pr_review":
            url = f"https://api.github.com/repos/{self._repo}/pulls/comments/{cid}/reactions"
        else:
            url = f"https://api.github.com/repos/{self._repo}/issues/comments/{cid}/reactions"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url, json={"content": content}, headers=_gh_headers(self._token)
            )
            resp.raise_for_status()
        return f"Added {content} reaction to comment {cid}"


class GitHubFileTool(ToolBase):
    """
    Read a file from the repository at any branch, tag, or commit SHA via the GitHub API.

    Unlike the workspace ReadFileTool, this works without checking out the repo
    and can access any ref. Implements the exoclaw Tool protocol.
    """

    _MAX_CHARS = 50_000

    def __init__(self, token: str | None = None, repo: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repo = repo or os.environ.get("GITHUB_REPOSITORY", "")

    @property
    def name(self) -> str:
        return "github_file"

    @property
    def description(self) -> str:
        return "Read a file from the repository at any branch, tag, or commit SHA."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root",
                },
                "ref": {
                    "type": "string",
                    "description": "Branch name, tag, or commit SHA (default: HEAD)",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str, ref: str | None = None, **kwargs: Any) -> str:
        url = f"https://api.github.com/repos/{self._repo}/contents/{path}"
        params: dict[str, str] = {}
        if ref:
            params["ref"] = ref

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=_gh_headers(self._token))
            if resp.status_code == 404:
                return f"File not found: {path}"
            resp.raise_for_status()
            data = resp.json()

        if data.get("type") != "file":
            return f"{path} is not a file (type: {data.get('type')})"

        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        if len(content) > self._MAX_CHARS:
            content = content[: self._MAX_CHARS] + "\n... (truncated)"
        return content


class GitHubChecksTool(ToolBase):
    """
    Read CI check run results for the current PR or a specific commit.

    Implements the exoclaw Tool protocol.
    """

    def __init__(self, token: str | None = None, repo: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
        self._head_sha: str | None = None
        self._pr_number: int | None = None

    def on_inbound(self, msg: "InboundMessage") -> None:
        head_sha = msg.metadata.get("head_sha")
        if head_sha:
            self._head_sha = str(head_sha)
        if msg.metadata.get("kind") == "pr":
            self._pr_number = int(msg.chat_id)

    @property
    def name(self) -> str:
        return "github_checks"

    @property
    def description(self) -> str:
        return "Read CI check run results for the current PR or a specific commit SHA."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Commit SHA or branch name (default: PR head)",
                },
            },
        }

    async def execute(self, ref: str | None = None, **kwargs: Any) -> str:
        sha = ref or self._head_sha

        if not sha and self._pr_number:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"https://api.github.com/repos/{self._repo}/pulls/{self._pr_number}",
                    headers=_gh_headers(self._token),
                )
                resp.raise_for_status()
                sha = resp.json()["head"]["sha"]

        if not sha:
            return "Error: no commit ref — trigger from a PR event or provide ref"

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.github.com/repos/{self._repo}/commits/{sha}/check-runs",
                headers=_gh_headers(self._token),
            )
            resp.raise_for_status()
            data = resp.json()

        runs = data.get("check_runs", [])
        if not runs:
            return f"No check runs found for {sha[:8]}"

        lines = [
            f"- {r['name']}: {r['status']} / {r.get('conclusion') or 'pending'}"
            for r in runs
        ]
        return f"Check runs for {sha[:8]}:\n" + "\n".join(lines)


class GitHubSearchTool(ToolBase):
    """
    Search issues or code within this repository.

    Implements the exoclaw Tool protocol.
    """

    def __init__(self, token: str | None = None, repo: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repo = repo or os.environ.get("GITHUB_REPOSITORY", "")

    @property
    def name(self) -> str:
        return "github_search"

    @property
    def description(self) -> str:
        return (
            "Search issues or code within this repository. "
            "Supports GitHub search syntax, e.g. 'is:open label:bug'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms (GitHub search syntax, repo: is added automatically)",
                },
                "kind": {
                    "type": "string",
                    "enum": ["issues", "code"],
                    "description": "What to search",
                },
            },
            "required": ["query", "kind"],
        }

    async def execute(self, query: str, kind: str = "issues", **kwargs: Any) -> str:
        full_query = f"{query} repo:{self._repo}"
        url = f"https://api.github.com/search/{kind}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                params={"q": full_query, "per_page": 10},
                headers=_gh_headers(self._token),
            )
            resp.raise_for_status()
            data = resp.json()

        items = data.get("items", [])
        total = data.get("total_count", 0)

        if not items:
            return f"No {kind} found for: {query}"

        if kind == "issues":
            lines = [f"- #{i['number']} [{i['state']}] {i['title']}" for i in items]
        else:
            lines = [f"- {i['path']}" for i in items]

        return f"Found {total} result(s) (showing {len(items)}):\n" + "\n".join(lines)
