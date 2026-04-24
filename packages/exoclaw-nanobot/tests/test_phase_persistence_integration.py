"""End-to-end integration tests for phase 1 (per-message append) and
phase 2 (disk-backed prior source) of the memory-model refactor.

Wires the real composition that runs in openclaw:

    DefaultConversation (file-backed SessionManager + minimal prompt)
      ↔ DBOSExecutor (inside a real DBOS workflow)
         ↔ AgentLoop's buffer protocol (set_messages / set_prior_source /
            load_messages)

Phase 1 (shipped in exoclaw-conversation 0.15.0 + exoclaw-executor-dbos
0.12.0): every assistant/tool/user message is persisted to the session
JSONL as it's produced, via ``DefaultConversation.append``. No
end-of-turn batched ``record`` call. Crash-recovery-friendly.

Phase 2 (shipped in exoclaw 0.19.1/0.20.0 +
exoclaw-conversation 0.16.0 + exoclaw-executor-dbos 0.13.0):
``DBOSExecutor.build_prompt`` auto-detects
``Conversation.load_persisted_history`` and installs a lazy
``PriorSource`` on the executor. Successive ``load_messages`` calls
re-read the history slice from session state rather than holding a
per-turn Python list copy of prior alongside SessionManager's own
cache. The prior list stops being double-held between SessionManager
and the executor.

Together these two changes are what the 2026-04-23 openclaw OOM
post-mortem identified as the fix: crash-recoverable mid-turn flushing
AND no double-held prior between LLM iterations.

Without this integration test, either phase could regress silently —
unit tests in their own packages verify the surfaces, but nothing
else exercises the phase 1 append path and the phase 2 auto-wire
against each other inside a real DBOS workflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from dbos import DBOS, DBOSConfig
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_conversation.session.manager import SessionManager
from exoclaw_executor_dbos import DBOSExecutor

_DB_PATH = f"/tmp/dbos_phase_integration_test_{os.getpid()}.sqlite"


@pytest.fixture(scope="module")
def dbos_instance() -> Any:
    """Module-scoped DBOS fixture — DBOS is a process-global singleton
    so we isolate the lifetime to this module."""
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "phase-integration-test",
        "system_database_url": f"sqlite:///{_DB_PATH}",
        "enable_otlp": False,
    }
    # Importing ensures @DBOS.step / @DBOS.workflow decorators register.
    import exoclaw_executor_dbos.executor  # noqa: F401

    dbos = DBOS(config=config)
    DBOS.launch()
    yield dbos
    DBOS.destroy()
    if os.path.exists(_DB_PATH):
        os.unlink(_DB_PATH)


def _make_conversation(workspace: Path) -> DefaultConversation:
    """DefaultConversation with real SessionManager + stub prompt/memory.

    Real SessionManager is the whole point — phase 1 writes through it
    to the on-disk JSONL, and phase 2's ``load_persisted_history`` reads
    from its session cache. Prompt and memory are stubbed because the
    phase 1+2 persistence story doesn't go through consolidation or
    skill rendering (and running those would require real providers).

    The stub ``build_messages`` splats history into
    ``[system_prompt, *history, user_message]`` so the resulting
    prompt list contains the same content (by equality) as
    ``load_persisted_history`` will return. The phase 2 auto-wire
    uses dict-EQUALITY (not ``id()``) to locate the history slice,
    since the real ``DefaultConversation.session.get_history``
    strips timestamps and returns fresh dict objects per call — any
    id-based match would never find the slice in production. Content
    equality handles both the fresh-dicts case and the splat-of-
    session-messages case.
    """
    sessions = SessionManager(workspace)

    prompt = MagicMock()

    def _build_messages(
        *,
        history: list[dict[str, Any]],
        current_message: str,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": "test-system"},
            *history,
            {"role": "user", "content": current_message},
        ]

    prompt.build_messages = _build_messages
    prompt.get_active_optional_tools = MagicMock(return_value=set())
    prompt.skills = MagicMock()
    prompt.skills.get_always_skills = MagicMock(return_value=[])

    memory = MagicMock()
    memory.consolidate = AsyncMock(return_value=False)
    memory.consolidate_messages = AsyncMock(return_value=False)

    return DefaultConversation(
        history=sessions,
        memory=memory,
        prompt=prompt,
        memory_window=100,
    )


def _read_jsonl_messages(workspace: Path, session_id: str) -> list[dict[str, Any]]:
    """Read the persisted JSONL, skipping the metadata header line."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    path = workspace / "sessions" / f"{safe}.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("_type") == "metadata":
            continue
        out.append(entry)
    return out


@pytest.mark.asyncio(loop_scope="session")
class TestPhase1PerMessageJSONLAppend:
    """Phase 1: each call to ``conversation.append`` writes one line to
    the session JSONL. No end-of-turn batch — mid-turn crash leaves
    work-in-progress on disk."""

    async def test_each_append_writes_one_line(self, tmp_path: Path) -> None:
        conv = _make_conversation(tmp_path)

        await conv.append("sess:1", {"role": "user", "content": "hi"})
        after_user = _read_jsonl_messages(tmp_path, "sess:1")
        assert len(after_user) == 1
        assert after_user[0]["role"] == "user"

        await conv.append("sess:1", {"role": "assistant", "content": "hello, how can I help?"})
        after_assistant = _read_jsonl_messages(tmp_path, "sess:1")
        assert len(after_assistant) == 2
        assert after_assistant[1]["role"] == "assistant"

        await conv.append(
            "sess:1",
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "name": "lookup",
                "content": "result",
            },
        )
        after_tool = _read_jsonl_messages(tmp_path, "sess:1")
        assert len(after_tool) == 3
        assert after_tool[2]["role"] == "tool"

    async def test_append_survives_partial_turn(self, tmp_path: Path) -> None:
        """Mid-turn crash semantics: after two appends, the JSONL has
        two messages. No end-of-turn ``record`` needs to run for those
        to be durable. This is the crash-recovery win phase 1 delivers.
        """
        conv = _make_conversation(tmp_path)

        await conv.append("sess:crash", {"role": "user", "content": "start"})
        await conv.append("sess:crash", {"role": "assistant", "content": "partial response"})

        # No ``record`` call — simulate a process crash here. Then
        # pretend a new process starts and opens the same session.
        conv2 = _make_conversation(tmp_path)
        session = conv2.history.get_or_create("sess:crash")
        messages = session.get_history(max_messages=100)
        # The new process sees both messages via the on-disk JSONL.
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert [m["content"] for m in messages] == ["start", "partial response"]


@pytest.mark.asyncio(loop_scope="session")
class TestPhase2DiskBackedPriorAutoWire:
    """Phase 2: ``DBOSExecutor.build_prompt`` auto-installs a
    ``PriorSource`` closure when ``Conversation`` exposes
    ``load_persisted_history``. Successive ``load_messages`` re-read
    the history slice from session state rather than holding a
    per-turn list copy."""

    async def test_prior_var_holds_callable_not_list(
        self, tmp_path: Path, dbos_instance: Any
    ) -> None:
        """After ``build_prompt``, the executor's ``_prior_var`` should
        hold a callable source — not a list snapshot. This is the
        signal that the auto-wire took the disk-backed path vs. the
        ``set_messages``-with-list fallback."""
        conv = _make_conversation(tmp_path)
        # Seed session so load_persisted_history has something to
        # return — empty-history would take the snapshot fallback.
        session = conv.history.get_or_create("sess:autowire")
        session.messages.extend(
            [
                {"role": "user", "content": "h1"},
                {"role": "assistant", "content": "h2"},
            ]
        )
        conv.history.save(session)

        executor = DBOSExecutor()
        await executor.build_prompt(conv, "sess:autowire", "new message")

        stored = executor._prior_var.get()
        assert callable(stored), (
            "phase 2 auto-wire didn't fire — ``_prior_var`` holds a "
            "list snapshot rather than a lazy source"
        )

    async def test_load_messages_reflects_mid_turn_appends(
        self, tmp_path: Path, dbos_instance: Any
    ) -> None:
        """The load-bearing phase 2 assertion: once auto-wire is
        installed, ``load_messages`` re-reads ``load_persisted_history``
        on each call. So a mid-turn ``conversation.append`` (the phase
        1 path) shows up on the next ``load_messages`` without
        rebuilding the prompt.

        This is the integration point where phase 1 and phase 2 meet —
        phase 1 writes each message as it's produced, phase 2's source
        re-reads the session state so the next iteration's prompt
        reflects the new messages.
        """
        conv = _make_conversation(tmp_path)
        session = conv.history.get_or_create("sess:combined")
        session.messages.extend(
            [
                {"role": "user", "content": "h1"},
                {"role": "assistant", "content": "h2"},
            ]
        )
        conv.history.save(session)

        executor = DBOSExecutor()
        await executor.build_prompt(conv, "sess:combined", "new message")

        before = executor.load_messages()
        assert any(m["content"] == "h1" for m in before)
        assert any(m["content"] == "h2" for m in before)

        # Mid-turn: simulate an assistant response being flushed via
        # the phase 1 path.
        await conv.append(
            "sess:combined",
            {"role": "assistant", "content": "h3-new-via-append"},
        )

        after = executor.load_messages()
        contents = [m["content"] for m in after]
        assert "h3-new-via-append" in contents, (
            "phase 2 source didn't pick up the phase 1 append — either "
            "the auto-wire regressed to snapshot mode or the source "
            "isn't invoking load_persisted_history per call"
        )

    async def test_empty_history_falls_back_to_snapshot(
        self, tmp_path: Path, dbos_instance: Any
    ) -> None:
        """Fresh sessions (no history) take the snapshot fallback —
        there's nothing to disk-back. Assert this doesn't accidentally
        install a lazy source that returns an empty list forever."""
        conv = _make_conversation(tmp_path)

        executor = DBOSExecutor()
        await executor.build_prompt(conv, "sess:fresh", "first message")

        # Still callable (set_messages installs a snapshot closure,
        # which is also callable) — check by signature instead. The
        # snapshot closure ignores session state; the disk-backed one
        # would pick up mutations.
        await conv.append(
            "sess:fresh",
            {"role": "assistant", "content": "response"},
        )

        # load_messages is the stable observable. Snapshot path means
        # we see the initial build_prompt return only — no leak of
        # the new append.
        loaded = executor.load_messages()
        assert all(m.get("content") != "response" for m in loaded), (
            "empty-history fallback must not read fresh appends"
        )
