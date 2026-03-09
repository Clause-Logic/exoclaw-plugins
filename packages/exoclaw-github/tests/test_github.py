"""Tests for exoclaw-github channel."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typing import Any

from exoclaw.bus.events import InboundMessage, OutboundMessage
from exoclaw_github.channel import GitHubChannel, GitHubEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_event(tmp_path: Path, data: Any) -> str:
    p = tmp_path / "event.json"
    p.write_text(json.dumps(data))
    return str(p)


def _make_channel(
    trigger: str | None = "@exoclaw",
    respond_to_issues_opened: bool = True,
    respond_to_prs_opened: bool = False,
) -> GitHubChannel:
    return GitHubChannel(
        token="test-token",
        trigger=trigger,
        respond_to_issues_opened=respond_to_issues_opened,
        respond_to_prs_opened=respond_to_prs_opened,
    )


# ---------------------------------------------------------------------------
# _parse_event: issues
# ---------------------------------------------------------------------------

def test_parse_issues_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "opened",
        "issue": {"number": 42, "title": "Bug report", "body": "It crashes", "user": {"login": "alice"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issues")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    ch = _make_channel()
    event = ch._parse_event()
    assert event is not None
    assert event.kind == "issue"
    assert event.number == 42
    assert event.sender == "alice"
    assert event.body == "It crashes"


def test_parse_issues_opened_skipped_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "opened",
        "issue": {"number": 1, "title": "T", "body": "B", "user": {"login": "alice"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issues")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    ch = _make_channel(respond_to_issues_opened=False)
    assert ch._parse_event() is None


def test_parse_issues_non_opened_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "closed",
        "issue": {"number": 1, "title": "T", "body": "B", "user": {"login": "alice"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issues")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    assert _make_channel()._parse_event() is None


# ---------------------------------------------------------------------------
# _parse_event: issue_comment
# ---------------------------------------------------------------------------

def test_parse_issue_comment_with_trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "created",
        "issue": {"number": 7, "title": "Some issue"},
        "comment": {"body": "Hey @exoclaw fix this", "user": {"login": "bob"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    event = _make_channel(trigger="@exoclaw")._parse_event()
    assert event is not None
    assert event.number == 7
    assert event.sender == "bob"


def test_parse_issue_comment_without_trigger_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "created",
        "issue": {"number": 7, "title": "T"},
        "comment": {"body": "Just a regular comment", "user": {"login": "bob"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    assert _make_channel(trigger="@exoclaw")._parse_event() is None


def test_parse_issue_comment_no_trigger_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "created",
        "issue": {"number": 3, "title": "T"},
        "comment": {"body": "Any comment", "user": {"login": "carol"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    event = _make_channel(trigger=None)._parse_event()
    assert event is not None
    assert event.sender == "carol"


# ---------------------------------------------------------------------------
# _parse_event: pull_request
# ---------------------------------------------------------------------------

def test_parse_pr_opened_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "opened",
        "pull_request": {"number": 99, "title": "Add feature", "body": "Details", "user": {"login": "dave"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    event = _make_channel(respond_to_prs_opened=True)._parse_event()
    assert event is not None
    assert event.kind == "pr"
    assert event.number == 99


def test_parse_pr_opened_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "opened",
        "pull_request": {"number": 99, "title": "T", "body": "B", "user": {"login": "dave"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    assert _make_channel()._parse_event() is None


# ---------------------------------------------------------------------------
# _parse_event: workflow_dispatch
# ---------------------------------------------------------------------------

def test_parse_dispatch_with_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"inputs": {"message": "Run the daily summary"}}
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    event = _make_channel()._parse_event()
    assert event is not None
    assert event.kind == "dispatch"
    assert event.body == "Run the daily summary"
    assert event.number == 0


def test_parse_dispatch_default_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data: dict[str, object] = {"inputs": {}}
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    event = _make_channel()._parse_event()
    assert event is not None
    assert event.body == "Workflow dispatched"


# ---------------------------------------------------------------------------
# _parse_event: unsupported / missing
# ---------------------------------------------------------------------------

def test_unsupported_event_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_event(tmp_path, {})
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    assert _make_channel()._parse_event() is None


def test_missing_event_path_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issues")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)

    assert _make_channel()._parse_event() is None


# ---------------------------------------------------------------------------
# start(): no event → returns immediately
# ---------------------------------------------------------------------------

async def test_start_no_event_returns_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    bus = MagicMock()
    ch = _make_channel()
    await ch.start(bus)  # should return without blocking
    bus.publish_inbound.assert_not_called()


# ---------------------------------------------------------------------------
# start(): publishes inbound and waits; send() resolves and posts comment
# ---------------------------------------------------------------------------

async def test_start_publishes_inbound_and_send_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {
        "action": "opened",
        "issue": {"number": 5, "title": "Help", "body": "Please help", "user": {"login": "user1"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issues")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    bus = MagicMock()
    bus.publish_inbound = AsyncMock()

    ch = _make_channel()

    posted: list[str] = []

    async def fake_post_comment(event: GitHubEvent, content: str) -> None:
        posted.append(content)

    ch._post_comment = fake_post_comment  # type: ignore[method-assign]

    outbound = OutboundMessage(channel="github", chat_id="5", content="Here is my answer")

    async def send_after_start() -> None:
        await asyncio.sleep(0.05)
        await ch.send(outbound)

    await asyncio.gather(ch.start(bus), send_after_start())

    bus.publish_inbound.assert_called_once()
    msg: InboundMessage = bus.publish_inbound.call_args[0][0]
    assert msg.channel == "github"
    assert msg.sender_id == "user1"
    assert msg.session_key_override == "github:issue:5"

    assert posted == ["Here is my answer"]


# ---------------------------------------------------------------------------
# send(): progress messages are ignored
# ---------------------------------------------------------------------------

async def test_send_ignores_progress_messages() -> None:
    ch = _make_channel()
    ch._pending_event = GitHubEvent(kind="issue", number=1, sender="u", body="b", repo="r")
    ch._response_event = asyncio.Event()

    with patch.object(ch, "_post_comment", new_callable=AsyncMock) as mock_post:
        await ch.send(OutboundMessage(
            channel="github", chat_id="1", content="thinking...",
            metadata={"_progress": True},
        ))
        mock_post.assert_not_called()

    assert not ch._response_event.is_set()


# ---------------------------------------------------------------------------
# send(): dispatch events log instead of posting
# ---------------------------------------------------------------------------

async def test_send_dispatch_logs_not_posts() -> None:
    ch = _make_channel()
    ch._pending_event = GitHubEvent(kind="dispatch", number=0, sender="u", body="b", repo="r")
    ch._response_event = asyncio.Event()

    with patch("httpx.AsyncClient") as mock_client:
        await ch.send(OutboundMessage(channel="github", chat_id="0", content="Done"))
        mock_client.assert_not_called()

    assert ch._response_event.is_set()


# ---------------------------------------------------------------------------
# stop(): sets response event so start() unblocks
# ---------------------------------------------------------------------------

async def test_stop_unblocks_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "opened",
        "issue": {"number": 1, "title": "T", "body": "B", "user": {"login": "u"}},
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issues")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    ch = _make_channel()
    ch._post_comment = AsyncMock()  # type: ignore[method-assign]

    async def stop_after_start() -> None:
        await asyncio.sleep(0.05)
        await ch.stop()

    await asyncio.gather(ch.start(bus), stop_after_start())


# ---------------------------------------------------------------------------
# _parse_event: pull_request_review_comment
# ---------------------------------------------------------------------------

def test_parse_review_comment_with_trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "created",
        "pull_request": {"number": 12, "title": "My PR"},
        "comment": {
            "id": 999,
            "body": "@exoclaw what does this do?",
            "user": {"login": "reviewer"},
            "path": "src/foo.py",
            "line": 42,
        },
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_review_comment")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    event = _make_channel(trigger="@exoclaw")._parse_event()
    assert event is not None
    assert event.kind == "pr"
    assert event.number == 12
    assert event.sender == "reviewer"
    assert event.reply_to_comment_id == 999
    assert "src/foo.py" in event.body
    assert "42" in event.body


def test_parse_review_comment_without_trigger_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "action": "created",
        "pull_request": {"number": 12, "title": "My PR"},
        "comment": {
            "id": 999,
            "body": "LGTM",
            "user": {"login": "reviewer"},
            "path": "src/foo.py",
            "line": 1,
        },
    }
    path = _write_event(tmp_path, data)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_review_comment")
    monkeypatch.setenv("GITHUB_EVENT_PATH", path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    assert _make_channel(trigger="@exoclaw")._parse_event() is None


# ---------------------------------------------------------------------------
# _post_comment: review comment uses pulls/comments reply endpoint
# ---------------------------------------------------------------------------

async def test_post_comment_reply_uses_review_endpoint() -> None:
    ch = _make_channel()
    event = GitHubEvent(
        kind="pr", number=5, sender="u", body="b", repo="owner/repo",
        reply_to_comment_id=999,
    )
    captured_url: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self
        async def __aexit__(self, *_: object) -> None:
            pass
        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            captured_url.append(url)
            return FakeResponse()

    with patch("httpx.AsyncClient", return_value=FakeClient()):
        await ch._post_comment(event, "Nice code")

    assert "pulls/comments/999/replies" in captured_url[0]


async def test_post_comment_issue_uses_issues_endpoint() -> None:
    ch = _make_channel()
    event = GitHubEvent(kind="issue", number=7, sender="u", body="b", repo="owner/repo")
    captured_url: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self
        async def __aexit__(self, *_: object) -> None:
            pass
        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            captured_url.append(url)
            return FakeResponse()

    with patch("httpx.AsyncClient", return_value=FakeClient()):
        await ch._post_comment(event, "Here is my answer")

    assert "issues/7/comments" in captured_url[0]


# ---------------------------------------------------------------------------
# GitHubReviewTool
# ---------------------------------------------------------------------------

async def test_review_tool_submits_approve() -> None:
    from exoclaw_github.tools import GitHubReviewTool

    tool = GitHubReviewTool(token="tok", repo="owner/repo")
    tool._pr_number = 5
    captured: list[dict[str, Any]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self
        async def __aexit__(self, *_: object) -> None:
            pass
        async def post(self, url: str, json: dict[str, Any], **kwargs: object) -> FakeResponse:
            captured.append({"url": url, "json": json})
            return FakeResponse()

    with patch("httpx.AsyncClient", return_value=FakeClient()):
        result = await tool.execute(event="APPROVE", body="Looks great!")

    assert "APPROVE" in result
    assert "#5" in result
    assert captured[0]["json"]["event"] == "APPROVE"
    assert "pulls/5/reviews" in captured[0]["url"]


async def test_review_tool_no_pr_number_returns_error() -> None:
    from exoclaw_github.tools import GitHubReviewTool

    tool = GitHubReviewTool(token="tok", repo="owner/repo")
    result = await tool.execute(event="APPROVE", body="LGTM")
    assert "Error" in result


async def test_review_tool_on_inbound_captures_pr_number() -> None:
    from exoclaw.bus.events import InboundMessage
    from exoclaw_github.tools import GitHubReviewTool

    tool = GitHubReviewTool(token="tok", repo="owner/repo")
    assert tool._pr_number is None
    msg = InboundMessage(
        channel="github", sender_id="u", chat_id="42", content="hi",
        metadata={"kind": "pr", "number": 42},
    )
    tool.on_inbound(msg)
    assert tool._pr_number == 42


async def test_review_tool_system_context() -> None:
    from exoclaw_github.tools import GitHubReviewTool

    tool = GitHubReviewTool(token="tok", repo="owner/repo")
    tool._pr_number = 7
    ctx = tool.system_context()
    assert ctx is not None
    assert "#7" in ctx
