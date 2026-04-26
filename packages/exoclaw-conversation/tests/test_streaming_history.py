"""End-to-end coverage for ``SessionManager(streaming_history=True)``.

The deployed multi-tenant bot can't afford to hold N concurrent
sessions × per-session unconsolidated tail in RAM. ``streaming_history``
(memory-model.md Step C) makes the unconsolidated tail live only on
disk between turns; ``read_history`` reads it on demand. These tests
pin the contract: ``session.messages`` stays empty across N turns,
``read_history`` still returns the correct slice, and the
``total_messages`` / ``last_consolidated`` bookkeeping advances
identically to the cached path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from exoclaw_conversation.session.manager import SessionManager


class TestStreamingHistoryFlag:
    def test_load_does_not_populate_session_messages(self, tmp_path: Path) -> None:
        """A session loaded under streaming has empty in-memory messages,
        even when the JSONL on disk has a long tail."""
        # Seed a session via a non-streaming manager (cheaper than
        # routing through the conversation layer for this assertion).
        seed_mgr = SessionManager(tmp_path, streaming_history=False)
        s = seed_mgr.get_or_create("ch:1")
        for i in range(20):
            s.add_message("user" if i % 2 == 0 else "assistant", f"msg-{i}")
        seed_mgr.save(s)

        streaming_mgr = SessionManager(tmp_path, streaming_history=True)
        s2 = streaming_mgr.get_or_create("ch:1")
        assert s2.messages == []
        # But the metadata still loaded — total_messages / offsets are intact.
        assert s2.total_messages == 20

    def test_read_history_returns_tail_from_disk(self, tmp_path: Path) -> None:
        """With session.messages empty, read_history reads the
        unconsolidated tail directly from JSONL."""
        seed_mgr = SessionManager(tmp_path, streaming_history=False)
        s = seed_mgr.get_or_create("ch:1")
        for i in range(10):
            s.add_message("user" if i % 2 == 0 else "assistant", f"msg-{i}")
        s.last_consolidated = 4
        seed_mgr.save(s)

        streaming_mgr = SessionManager(tmp_path, streaming_history=True)
        history = streaming_mgr.read_history("ch:1", max_messages=500)
        # last_consolidated=4 → tail starts at msg-4 (a user message,
        # so the leading-non-user drop is a no-op).
        assert len(history) == 6
        assert history[0]["content"] == "msg-4"
        assert history[-1]["content"] == "msg-9"

    def test_session_messages_stays_empty_across_many_turns(self, tmp_path: Path) -> None:
        """Drive the conversation surface (build_prompt + record-style
        save_append) for many turns. session.messages must remain empty
        — that's the entire RAM win."""
        from datetime import datetime

        mgr = SessionManager(tmp_path, streaming_history=True)
        s = mgr.get_or_create("ch:1")

        for turn in range(200):
            # Mirror what _prepare_turn does under streaming: increment
            # total_messages, write to disk, do NOT append to session.messages.
            entries = [
                {
                    "role": "user",
                    "content": f"u-{turn}",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": f"a-{turn}",
                    "timestamp": datetime.now().isoformat(),
                },
            ]
            s._total_messages += 2
            mgr.save_append(s, entries)

        # The headline assertion: 200 turns of bookkeeping, zero in-memory tail.
        assert s.messages == []
        assert s.total_messages == 400

        # And the streaming reader still gets the full unconsolidated tail.
        history = mgr.read_history("ch:1", max_messages=500)
        assert len(history) == 400
        assert history[0]["content"] == "u-0"
        assert history[-1]["content"] == "a-199"

    def test_save_preserves_disk_tail_under_streaming(self, tmp_path: Path) -> None:
        """``save()`` is a full rewrite. Under streaming the in-memory
        list is empty, so a naive rewrite would wipe the JSONL. The
        tail must come from disk instead."""
        seed_mgr = SessionManager(tmp_path, streaming_history=False)
        s = seed_mgr.get_or_create("ch:1")
        for i in range(5):
            s.add_message("user" if i % 2 == 0 else "assistant", f"msg-{i}")
        seed_mgr.save(s)

        streaming_mgr = SessionManager(tmp_path, streaming_history=True)
        s2 = streaming_mgr.get_or_create("ch:1")
        assert s2.messages == []  # Confirm empty before save.

        # Rewrite — common after metadata-only changes (consolidation
        # advancing last_consolidated). Tail messages must survive.
        s2.metadata["touched"] = True
        streaming_mgr.save(s2)

        # Reload via a fresh non-streaming manager and verify the tail
        # is intact.
        check_mgr = SessionManager(tmp_path, streaming_history=False)
        s3 = check_mgr.get_or_create("ch:1")
        assert s3.total_messages == 5
        assert len(s3.messages) == 5
        assert s3.messages[0]["content"] == "msg-0"
        assert s3.messages[-1]["content"] == "msg-4"
        assert s3.metadata.get("touched") is True

    def test_default_is_non_streaming(self, tmp_path: Path) -> None:
        """The flag defaults False — existing callers that rely on
        ``session.messages`` being populated by ``_load`` keep working
        without code changes. Flip is opt-in per backend instance."""
        mgr = SessionManager(tmp_path)
        assert mgr.streaming_history is False

        s = mgr.get_or_create("ch:1")
        s.add_message("user", "hi")
        mgr.save(s)

        s2 = SessionManager(tmp_path).get_or_create("ch:1")
        # Default mgr populates session.messages from disk.
        assert len(s2.messages) == 1
        assert s2.messages[0]["content"] == "hi"
