"""Tests for exoclaw-conversation package."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from exoclaw_conversation.context import _RUNTIME_CONTEXT_TAG, ContextBuilder
from exoclaw_conversation.conversation import _RUNTIME_CONTEXT_TAG as CONV_TAG
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_conversation.helpers import detect_image_mime, ensure_dir, safe_filename
from exoclaw_conversation.memory import MemoryStore
from exoclaw_conversation.protocols import HistoryStore, MemoryBackend, PromptBuilder
from exoclaw_conversation.session.manager import Session, SessionManager
from exoclaw_conversation.skills import SkillsLoader

# ---------------------------------------------------------------------------
# helpers.py
# ---------------------------------------------------------------------------


class TestEnsureDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "new" / "nested"
        result = ensure_dir(d)
        assert result.exists()
        assert result == d

    def test_returns_path(self, tmp_path: Path) -> None:
        result = ensure_dir(tmp_path)
        assert result == tmp_path


class TestSafeFilename:
    def test_replaces_unsafe_chars(self) -> None:
        assert safe_filename('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"

    def test_safe_string_unchanged(self) -> None:
        assert safe_filename("hello_world-123") == "hello_world-123"

    def test_strips_whitespace(self) -> None:
        assert safe_filename("  hello  ") == "hello"


class TestDetectImageMime:
    def test_png(self) -> None:
        assert detect_image_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10) == "image/png"

    def test_jpeg(self) -> None:
        assert detect_image_mime(b"\xff\xd8\xff" + b"\x00" * 10) == "image/jpeg"

    def test_gif87(self) -> None:
        assert detect_image_mime(b"GIF87a" + b"\x00" * 10) == "image/gif"

    def test_gif89(self) -> None:
        assert detect_image_mime(b"GIF89a" + b"\x00" * 10) == "image/gif"

    def test_webp(self) -> None:
        assert detect_image_mime(b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP") == "image/webp"

    def test_unknown(self) -> None:
        assert detect_image_mime(b"\x00\x01\x02\x03") is None


# ---------------------------------------------------------------------------
# session/manager.py
# ---------------------------------------------------------------------------


class TestSession:
    def test_add_message(self) -> None:
        s = Session(key="test")
        s.add_message("user", "hello")
        assert len(s.messages) == 1
        assert s.messages[0]["role"] == "user"
        assert s.messages[0]["content"] == "hello"

    def test_add_message_total_messages_no_double_count(self) -> None:
        """add_message must not double-count total_messages (off-by-one regression)."""
        s = Session(key="test")
        assert s.total_messages == 0
        s.add_message("user", "one")
        assert s.total_messages == 1
        s.add_message("assistant", "two")
        assert s.total_messages == 2
        assert len(s.messages) == 2

    def test_add_message_total_with_offset(self) -> None:
        """add_message with pre-existing offset must track total correctly."""
        s = Session(key="test")
        s._messages_offset = 100
        s._total_messages = 105
        s.messages = [{"role": "user", "content": str(i)} for i in range(5)]
        s.add_message("user", "new")
        assert s.total_messages == 106
        assert len(s.messages) == 6

    def test_get_history_empty(self) -> None:
        s = Session(key="test")
        assert s.get_history() == []

    def test_get_history_aligns_to_user(self) -> None:
        s = Session(key="test")
        s.messages = [
            {"role": "assistant", "content": "stale"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        history = s.get_history()
        assert history[0]["role"] == "user"

    def test_get_history_returns_full_log(self) -> None:
        """``Session.get_history`` returns the entire message log now;
        view-windowing belongs to the consolidation policy via its
        sidecar, not to ``Session``."""
        s = Session(key="test")
        for i in range(10):
            s.messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": str(i)})
        history = s.get_history()
        assert [m["content"] for m in history] == [str(i) for i in range(10)]

    def test_get_history_strips_extra_keys(self) -> None:
        s = Session(key="test")
        s.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01"}]
        history = s.get_history()
        assert "timestamp" not in history[0]

    def test_get_history_preserves_tool_calls(self) -> None:
        s = Session(key="test")
        s.messages = [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "done", "tool_call_id": "1", "name": "exec"},
        ]
        history = s.get_history()
        assert history[1]["tool_calls"] == [{"id": "1"}]
        assert history[2]["tool_call_id"] == "1"

    def test_get_history_repairs_orphan_tool_results(self) -> None:
        """Sessions persisted before the pair-split fix have orphan tool_results
        in the kept tail. get_history() must strip them so the request shape is
        valid for strict providers (MiniMax).
        """
        s = Session(key="test")
        # Simulates a tail loaded from disk: first message is a tool_result
        # whose tool_call was archived. No user message exists so the
        # leading-non-user peel can't save us.
        s.messages = [
            {"role": "tool", "content": "orphan result", "tool_call_id": "T7"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "T8"}]},
            {"role": "tool", "content": "ok", "tool_call_id": "T8"},
            {"role": "assistant", "content": "done"},
        ]
        history = s.get_history()
        tool_ids_in_history = {m.get("tool_call_id") for m in history if m.get("role") == "tool"}
        assert "T7" not in tool_ids_in_history
        assert "T8" in tool_ids_in_history

    def test_get_history_strips_orphan_tool_calls_from_assistant(self) -> None:
        """Dangling in the other direction: assistant declares tool_calls but
        the matching tool_result is archived. Drop those tool_calls entries.
        """
        s = Session(key="test")
        s.messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [{"id": "unanswered"}, {"id": "answered"}],
            },
            {"role": "tool", "content": "r", "tool_call_id": "answered"},
        ]
        history = s.get_history()
        asst = [m for m in history if m.get("role") == "assistant"][0]
        assert asst["tool_calls"] == [{"id": "answered"}]

    def test_get_history_drops_assistant_with_only_orphan_tool_calls(self) -> None:
        """If an assistant message has no content and all its tool_calls are
        orphaned, drop the message entirely rather than send an empty shell.
        """
        s = Session(key="test")
        s.messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "orphan"}]},
            {"role": "assistant", "content": "recovered"},
        ]
        history = s.get_history()
        # The orphan-only assistant message is gone; only the user and the
        # "recovered" assistant remain.
        assert len(history) == 2
        assert history[0]["content"] == "go"
        assert history[1]["content"] == "recovered"

    def test_clear(self) -> None:
        s = Session(key="test")
        s.add_message("user", "hi")
        s.clear()
        assert s.messages == []
        assert s.total_messages == 0


class TestSessionManager:
    def test_get_or_create_new(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:123")
        assert s.key == "ch:123"

    def test_get_or_create_cached(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        s1 = mgr.get_or_create("ch:123")
        s2 = mgr.get_or_create("ch:123")
        assert s1 is s2

    def test_save_and_load(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:123")
        s.add_message("user", "hello")
        mgr.save(s)

        mgr2 = SessionManager(tmp_path)
        s2 = mgr2.get_or_create("ch:123")
        assert s2.total_messages == 1
        assert len(s2.messages) == 1
        assert s2.messages[0]["content"] == "hello"

    def test_load_keeps_full_log_in_ram(self, tmp_path: Path) -> None:
        """Non-streaming mode loads the entire log into ``session.messages``.
        Consolidation no longer windows the in-RAM tail — that's the
        policy's job, applied via ``transform``."""
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:123")
        s.add_message("user", "old")
        s.add_message("assistant", "old reply")
        s.add_message("user", "new")
        mgr.save(s)

        mgr2 = SessionManager(tmp_path)
        s2 = mgr2.get_or_create("ch:123")
        assert s2.total_messages == 3
        assert len(s2.messages) == 3
        assert s2.messages[2]["content"] == "new"

    def test_load_range(self, tmp_path: Path) -> None:
        """load_range reads specific message ranges from disk."""
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:123")
        for i in range(5):
            s.add_message("user" if i % 2 == 0 else "assistant", str(i))
        mgr.save(s)

        msgs = mgr.load_range("ch:123", 1, 3)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "1"
        assert msgs[1]["content"] == "2"

    def test_save_append_is_append_only(self, tmp_path: Path) -> None:
        """save_append must not read the full file — only append."""
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:123")
        s.add_message("user", "first")
        mgr.save(s)

        # Append a second message
        new_msg = {"role": "assistant", "content": "second", "timestamp": "2026-01-01T00:00"}
        s.messages.append(new_msg)
        mgr.save_append(s, [new_msg])

        # Reload and verify both messages present
        mgr2 = SessionManager(tmp_path)
        s2 = mgr2.get_or_create("ch:123")
        assert s2.total_messages == 2
        assert s2.messages[0]["content"] == "first"
        assert s2.messages[1]["content"] == "second"

    def test_streaming_load_keeps_messages_empty(self, tmp_path: Path) -> None:
        """``streaming_history=True`` deliberately keeps
        ``session.messages`` empty — callers go through ``reader()``
        for on-demand disk reads. Non-streaming keeps the full log in
        memory; streaming doesn't."""
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:big")
        for i in range(100):
            s.add_message("user" if i % 2 == 0 else "assistant", f"msg-{i}")
        mgr.save(s)

        streaming = SessionManager(tmp_path, streaming_history=True)
        s2 = streaming.get_or_create("ch:big")
        assert s2.total_messages == 100
        assert len(s2.messages) == 0  # streaming holds nothing in RAM
        # But the reader streams the full log on demand.
        msgs = streaming.load_range("ch:big", 0, 100)
        assert len(msgs) == 100
        assert msgs[0]["content"] == "msg-0"
        assert msgs[-1]["content"] == "msg-99"

    def test_invalidate(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        mgr.get_or_create("ch:123")
        mgr.invalidate("ch:123")
        assert "ch:123" not in mgr._cache

    def test_list_sessions(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:abc")
        mgr.save(s)
        sessions = mgr.list_sessions()
        assert any(sess["key"] == "ch:abc" for sess in sessions)

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        bad = tmp_path / "sessions" / "bad.jsonl"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("not json\n")
        s = mgr.get_or_create("bad")
        assert s.messages == []

    def test_list_sessions_skips_corrupt(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        bad = tmp_path / "sessions" / "bad.jsonl"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("not json\n")
        sessions = mgr.list_sessions()
        assert isinstance(sessions, list)


# ---------------------------------------------------------------------------
# memory.py
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def test_read_write_long_term(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.read_long_term() == ""
        store.write_long_term("facts")
        assert store.read_long_term() == "facts"

    def test_append_history(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.append_history("[2024-01-01] USER: hello")
        content = store.history_file.read_text()
        assert "[2024-01-01]" in content

    def test_get_memory_context_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert store.get_memory_context() == ""

    def test_get_memory_context_with_content(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.write_long_term("I know things")
        ctx = store.get_memory_context()
        assert "I know things" in ctx
        assert "Long-term Memory" in ctx

    async def test_summarize_no_provider(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        assert await store.summarize(msgs, archive_all=True) is None

    async def test_summarize_empty(self, tmp_path: Path) -> None:
        provider = MagicMock()
        store = MemoryStore(tmp_path, provider=provider, model="test-model")
        # Empty input is a no-op success — returns "" rather than None so
        # callers can distinguish from a failed LLM call.
        assert await store.summarize([], archive_all=False) == ""

    async def test_summarize_calls_provider(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = {"history_entry": "[2024-01-01] summary", "memory_update": "facts"}
        response.tool_calls = [tc]

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        msgs = [
            {"role": "user", "content": "hello", "timestamp": "2024-01-01T00:00"},
            {"role": "assistant", "content": "hi", "timestamp": "2024-01-01T00:01"},
        ]
        entry = await store.summarize(msgs, archive_all=True)
        assert entry == "[2024-01-01] summary"
        assert "facts" in store.read_long_term()

    async def test_summarize_no_tool_call(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = False

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        assert await store.summarize(msgs, archive_all=True) is None

    async def test_summarize_provider_exception(self, tmp_path: Path) -> None:
        provider = MagicMock()
        provider.chat = AsyncMock(side_effect=Exception("boom"))
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        assert await store.summarize(msgs, archive_all=True) is None

    async def test_summarize_args_as_string(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = json.dumps({"history_entry": "entry", "memory_update": "update"})
        response.tool_calls = [tc]

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        assert await store.summarize(msgs, archive_all=True) == "entry"

    async def test_summarize_args_as_list(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = [{"history_entry": "entry", "memory_update": "update"}]
        response.tool_calls = [tc]

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        assert await store.summarize(msgs, archive_all=True) == "entry"

    async def test_summarize_args_unexpected_type(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = 42
        response.tool_calls = [tc]

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        assert await store.summarize(msgs, archive_all=True) is None


# ---------------------------------------------------------------------------
# skills.py
# ---------------------------------------------------------------------------


class TestSkillsLoader:
    def _make_skill(self, workspace: Path, name: str, content: str = "# Skill") -> None:
        skill_dir = workspace / "skills" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content)

    def test_list_skills_empty(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        assert loader.list_skills() == []

    def test_list_skills_finds_skill(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "myscill")
        loader = SkillsLoader(tmp_path)
        skills = loader.list_skills()
        assert any(s["name"] == "myscill" for s in skills)

    def test_load_skill_found(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "myskill", "# My Skill\ncontent")
        loader = SkillsLoader(tmp_path)
        assert loader.load_skill("myskill") == "# My Skill\ncontent"

    def test_load_skill_not_found(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        assert loader.load_skill("missing") is None

    def test_load_skills_for_context(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "s1", "content1")
        loader = SkillsLoader(tmp_path)
        result = loader.load_skills_for_context(["s1", "missing"])
        assert "content1" in result
        assert "s1" in result

    def test_load_skills_for_context_empty(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        assert loader.load_skills_for_context([]) == ""

    def test_build_skills_summary_empty(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        assert loader.build_skills_summary() == ""

    def test_build_skills_summary(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "myskill")
        loader = SkillsLoader(tmp_path)
        summary = loader.build_skills_summary()
        assert "<skills>" in summary
        assert "myskill" in summary

    def test_skill_with_frontmatter(self, tmp_path: Path) -> None:
        content = '---\ndescription: "Does stuff"\n---\n# Skill body'
        self._make_skill(tmp_path, "described", content)
        loader = SkillsLoader(tmp_path)
        desc = loader._get_skill_description("described")
        assert desc == "Does stuff"

    def test_skill_missing_description(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "nodesc", "# body")
        loader = SkillsLoader(tmp_path)
        assert loader._get_skill_description("nodesc") == "nodesc"

    def test_get_skill_metadata_no_frontmatter(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "plain", "# no frontmatter")
        loader = SkillsLoader(tmp_path)
        assert loader.get_skill_metadata("plain") is None

    def test_get_skill_metadata_missing_skill(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        assert loader.get_skill_metadata("missing") is None

    def test_check_requirements_no_requires(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        assert loader._check_requirements({}) is True

    def test_check_requirements_missing_bin(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        assert (
            loader._check_requirements({"requires": {"bins": ["definitely_not_installed_xyz"]}})
            is False
        )

    def test_check_requirements_missing_env(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        assert (
            loader._check_requirements({"requires": {"env": ["DEFINITELY_NOT_SET_XYZ_ABC"]}})
            is False
        )

    def test_parse_exoclaw_metadata_valid_json(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        result = loader._parse_exoclaw_metadata('{"exoclaw": {"always": true}}')
        assert result == {"always": True}

    def test_parse_exoclaw_metadata_nanobot_key(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        result = loader._parse_exoclaw_metadata('{"nanobot": {"always": true}}')
        assert result == {"always": True}

    def test_parse_exoclaw_metadata_invalid_json(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        assert loader._parse_exoclaw_metadata("not json") == {}

    def test_parse_exoclaw_metadata_non_dict(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        assert loader._parse_exoclaw_metadata("[1, 2, 3]") == {}

    def test_get_always_skills(self, tmp_path: Path) -> None:
        content = '---\nmetadata: {"exoclaw": {"always": true}}\n---\n# skill'
        self._make_skill(tmp_path, "always_skill", content)
        loader = SkillsLoader(tmp_path)
        always = loader.get_always_skills()
        assert "always_skill" in always

    def test_get_bootstrap_injections_empty(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        assert loader.get_bootstrap_injections() == []

    def test_get_bootstrap_injections(self, tmp_path: Path) -> None:
        hook_path = tmp_path / "skills" / "myhook" / "hooks" / "exoclaw" / "bootstrap.md"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("# Bootstrap content")
        loader = SkillsLoader(tmp_path)
        injections = loader.get_bootstrap_injections()
        assert any("Bootstrap content" in i for i in injections)

    def test_get_skill_hook_scripts_empty(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        assert loader.get_skill_hook_scripts("pre_turn") == []


# ---------------------------------------------------------------------------
# context.py
# ---------------------------------------------------------------------------


class TestContextBuilder:
    def test_build_system_prompt_minimal(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        prompt = builder.build_system_prompt()
        assert "exoclaw" in prompt

    def test_build_system_prompt_with_soul(self, tmp_path: Path) -> None:
        (tmp_path / "SOUL.md").write_text("You are special.")
        builder = ContextBuilder(tmp_path)
        prompt = builder.build_system_prompt()
        assert "You are special." in prompt

    def test_build_system_prompt_with_memory(self, tmp_path: Path) -> None:
        memory = MagicMock()
        memory.get_memory_context.return_value = "I know things"
        builder = ContextBuilder(tmp_path, memory=memory)
        prompt = builder.build_system_prompt()
        assert "I know things" in prompt

    def test_build_system_prompt_with_extra_context(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        prompt = builder.build_system_prompt(extra_context="extra stuff")
        assert "extra stuff" in prompt

    def test_build_runtime_context(self) -> None:
        ctx = ContextBuilder._build_runtime_context("telegram", "456")
        assert _RUNTIME_CONTEXT_TAG in ctx
        assert "telegram" in ctx
        assert "456" in ctx

    def test_build_runtime_context_no_channel(self) -> None:
        ctx = ContextBuilder._build_runtime_context(None, None)
        assert _RUNTIME_CONTEXT_TAG in ctx
        assert "Current Time" in ctx

    def test_build_messages_text(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        msgs = builder.build_messages(history=[], current_message="hello")
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert "hello" in str(msgs[-1]["content"])

    def test_isolated_system_prompt_skips_persona_memory_and_skills_menu(
        self, tmp_path: Path
    ) -> None:
        """Isolated mode strips the whole persona/memory/menu envelope so
        small open-weight models don't get swamped by context that
        contradicts the per-call skill. The only content that survives is
        a tiny worker preamble plus the active skill bodies.
        """
        (tmp_path / "SOUL.md").write_text("You are Luna and you love cats.")
        (tmp_path / "USER.md").write_text("Stephen prefers short replies.")
        memory = MagicMock()
        memory.get_memory_context.return_value = "Previous session notes: ..."
        builder = ContextBuilder(tmp_path, memory=memory)

        normal = builder.build_system_prompt()
        isolated = builder.build_system_prompt(isolated=True)

        # Envelope bits that MUST disappear in isolated mode.
        for phrase in (
            "You are Luna",
            "Stephen prefers",
            "Previous session notes",
            "# exoclaw",  # identity preamble
        ):
            assert phrase in normal, f"baseline should contain: {phrase!r}"
            assert phrase not in isolated, (
                f"isolated mode leaked: {phrase!r}\n---\n{isolated[:500]}"
            )

        # And the short worker preamble IS there.
        assert "worker" in isolated.lower()
        # Isolated envelope stays small — <500 chars when no skills active.
        assert len(isolated) < 500, f"isolated prompt too large: {len(isolated)} chars\n{isolated}"

    def test_isolated_preserves_caller_supplied_extra_and_turn_context(
        self, tmp_path: Path
    ) -> None:
        """Isolated mode only strips *implicit* envelope (persona, memory,
        history). Caller-supplied ``extra_context`` and ``turn_context``
        are explicit inputs the caller wants the model to see — they
        must flow through."""
        builder = ContextBuilder(tmp_path)
        msgs = builder.build_messages(
            history=[],
            current_message="task body",
            extra_context="retrieved doc X",
            turn_context=["note A", "note B"],
            isolated=True,
        )
        system = str(msgs[0]["content"])
        user = str(msgs[-1]["content"])
        # extra_context ends up in the system prompt.
        assert "retrieved doc X" in system
        # turn_context is prepended to the user message (docstring contract).
        assert "note A" in user
        assert "note B" in user
        assert "task body" in user

    def test_isolated_build_messages_drops_history_and_runtime_context(
        self, tmp_path: Path
    ) -> None:
        """Isolated mode also strips session history and the runtime
        metadata prefix so the LLM sees only ``[system, user]`` with the
        caller-provided task as the entire user content.
        """
        builder = ContextBuilder(tmp_path)
        prior_history = [
            {"role": "user", "content": "prior turn 1"},
            {"role": "assistant", "content": "prior reply 1"},
        ]
        msgs = builder.build_messages(
            history=prior_history,
            current_message="current task",
            channel="ipc",
            chat_id="x",
            isolated=True,
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "current task"
        # Runtime-context preamble is gone.
        assert "Runtime Context" not in str(msgs[1]["content"])
        assert "ipc" not in str(msgs[1]["content"])

    def test_build_messages_with_history(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        history = [{"role": "user", "content": "prev"}]
        msgs = builder.build_messages(history=history, current_message="new")
        assert msgs[1]["content"] == "prev"

    def test_build_messages_with_media(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        builder = ContextBuilder(tmp_path)
        msgs = builder.build_messages(history=[], current_message="look", media=[str(img)])
        user_content = msgs[-1]["content"]
        assert isinstance(user_content, list)
        assert any(c.get("type") == "image_url" for c in user_content)

    def test_build_messages_media_nonexistent(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        msgs = builder.build_messages(
            history=[], current_message="look", media=["/nonexistent.png"]
        )
        assert isinstance(msgs[-1]["content"], str)

    def test_add_tool_result(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        msgs: list[dict[str, Any]] = []
        builder.add_tool_result(msgs, "call_1", "exec", "output")
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["content"] == "output"

    def test_add_assistant_message(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        msgs: list[dict[str, Any]] = []
        builder.add_assistant_message(msgs, "hello", tool_calls=[{"id": "1"}])
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"] == [{"id": "1"}]


# ---------------------------------------------------------------------------
# protocols.py
# ---------------------------------------------------------------------------


class TestProtocols:
    def test_session_manager_satisfies_history_store(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        assert isinstance(mgr, HistoryStore)

    def test_memory_store_satisfies_memory_backend(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        assert isinstance(store, MemoryBackend)

    def test_context_builder_satisfies_prompt_builder(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        assert isinstance(builder, PromptBuilder)


# ---------------------------------------------------------------------------
# conversation.py
# ---------------------------------------------------------------------------


class _MockReader:
    """Minimal in-memory ``SessionReader`` for unit tests. Streams from
    a list — fine for tests, not for production paths that need disk
    streaming."""

    def __init__(self, key: str, source: list[dict[str, Any]]) -> None:
        self._key = key
        self._source = source

    @property
    def key(self) -> str:
        return self._key

    async def count(self) -> int:
        return len(self._source)

    def stream(self, *, start: int = 0, end: int | None = None):
        async def _gen():
            stop = end if end is not None else len(self._source)
            for msg in self._source[start:stop]:
                yield msg

        return _gen()

    async def at(self, index: int):
        if 0 <= index < len(self._source):
            return self._source[index]
        return None


def _make_mock_history(session: Session | None = None) -> MagicMock:
    s = session or Session(key="test:1")
    h = MagicMock(
        spec=[
            "get_or_create",
            "save",
            "save_append",
            "load_range",
            "reader",
            "invalidate",
            "list_sessions",
            "sessions_dir",
        ]
    )
    h.get_or_create.return_value = s
    h.list_sessions.return_value = [{"key": "test:1"}]
    h.load_range.return_value = []
    # Reader streams from session.messages — keeps tests that seed the
    # in-memory list working without per-test reader wiring.
    h.reader.side_effect = lambda key: _MockReader(key, list(s.messages))
    # Default to no on-disk sidecar location so clear() doesn't try to
    # delete files. Tests that exercise sidecar interaction set this
    # explicitly.
    h.sessions_dir = None
    return h


def _make_mock_memory() -> MagicMock:
    m = MagicMock(spec=["get_memory_context", "summarize"])
    m.get_memory_context.return_value = ""
    m.summarize = AsyncMock(return_value="[summary]")
    return m


def _make_mock_prompt() -> MagicMock:
    p = MagicMock(spec=["build_messages"])
    p.build_messages.return_value = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    return p


class TestDefaultConversation:
    async def test_build_prompt_basic(self) -> None:
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        msgs = await conv.build_prompt("test:1", "hello")
        assert msgs[0]["role"] == "system"

    async def test_build_prompt_with_plugin_context(self) -> None:
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        await conv.build_prompt("test:1", "hello", plugin_context=["extra"])
        call_kwargs = conv.prompt.build_messages.call_args[1]  # type: ignore[union-attr]
        assert "extra" in call_kwargs["extra_context"]

    async def test_build_prompt_isolated_rejects_non_bool(self) -> None:
        """``isolated`` must be a real bool — ``bool("false")`` is True, so
        accepting strings would silently enable isolation on typos. Raise
        instead so the caller fixes the bug."""
        import pytest as _pytest

        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        with _pytest.raises(TypeError, match="'isolated' must be a bool"):
            await conv.build_prompt("test:1", "hello", isolated="false")

    async def test_build_prompt_isolated_forwarded_to_prompt_builder(self) -> None:
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        await conv.build_prompt("test:1", "hello", isolated=True)
        call_kwargs = conv.prompt.build_messages.call_args[1]  # type: ignore[union-attr]
        assert call_kwargs["isolated"] is True

    async def test_post_turn_runs_policy_maintenance(self) -> None:
        """Maintenance moved from ``build_prompt`` to ``post_turn`` —
        the policy's ``on_turn_complete`` runs in a background task
        after each turn, not during prompt assembly. ``build_prompt``
        is now strictly read."""
        session = Session(key="test:1")

        class _SpyPolicy:
            def __init__(self) -> None:
                self.on_turn_complete_calls = 0

            def transform(self, reader, *, budget=None):
                async def _gen():
                    async for m in reader.stream():
                        yield m

                return _gen()

            async def on_turn_complete(self, reader) -> None:
                self.on_turn_complete_calls += 1

        policy = _SpyPolicy()
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            consolidation_policy=policy,  # type: ignore[arg-type]
            memory_window=100,
        )
        # build_prompt does not run on_turn_complete.
        await conv.build_prompt("test:1", "hello")
        await asyncio.sleep(0.05)
        assert policy.on_turn_complete_calls == 0
        # post_turn schedules maintenance.
        await conv.post_turn("test:1")
        await asyncio.sleep(0.05)
        assert policy.on_turn_complete_calls == 1

    async def test_record_saves_turn(self) -> None:
        session = Session(key="test:1")
        history = _make_mock_history(session)
        conv = DefaultConversation(
            history=history,
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        await conv.record("test:1", [{"role": "user", "content": "hi"}])
        history.save_append.assert_called_once()

    async def test_record_strips_runtime_tag(self) -> None:
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        msg = {"role": "user", "content": f"{CONV_TAG}\nmetadata\n\nactual message"}
        await conv.record("test:1", [msg])
        assert session.messages[0]["content"] == "actual message"

    async def test_record_drops_empty_user_after_tag(self) -> None:
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        msg = {"role": "user", "content": f"{CONV_TAG}\nonly metadata"}
        await conv.record("test:1", [msg])
        assert len(session.messages) == 0

    async def test_record_truncates_large_tool_result(self) -> None:
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        big = "x" * 2000
        await conv.record("test:1", [{"role": "tool", "content": big}])
        assert len(session.messages[0]["content"]) < 600
        assert "truncated" in session.messages[0]["content"]

    async def test_record_skips_empty_assistant(self) -> None:
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        await conv.record("test:1", [{"role": "assistant", "content": None}])
        assert len(session.messages) == 0

    async def test_record_strips_base64_images(self) -> None:
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "describe"},
            ],
        }
        await conv.record("test:1", [msg])
        saved = session.messages[0]["content"]
        assert isinstance(saved, list)
        assert any(c["type"] == "text" and c["text"] == "[image]" for c in saved)

    async def test_record_runtime_tag_in_list_content(self) -> None:
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{CONV_TAG}\nmetadata"},
                {"type": "text", "text": "actual"},
            ],
        }
        await conv.record("test:1", [msg])
        saved = session.messages[0]["content"]
        assert all(c["text"] != f"{CONV_TAG}\nmetadata" for c in saved)

    def test_no_load_persisted_history(self) -> None:
        """``load_persisted_history`` is removed — the executor's
        prior-source pattern reaches into ``conversation.history.reader``
        directly now, so the sync convenience method on ``DefaultConversation``
        no longer exists. The executor's ``getattr(... None)`` lookup
        already handles its absence."""
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        assert not hasattr(conv, "load_persisted_history")

    # ------------------------------------------------------------------
    # AppendableConversation surface (added for exoclaw>=0.19.0)
    # ------------------------------------------------------------------

    async def test_append_flushes_single_message(self) -> None:
        """``append`` writes one message to disk via the existing
        ``save_append`` path — one call, one message. The agent loop
        uses this after each assistant/tool/user message instead of
        batching at end-of-turn."""
        session = Session(key="test:1")
        history = _make_mock_history(session)
        conv = DefaultConversation(
            history=history,
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )

        await conv.append("test:1", {"role": "user", "content": "hi"})

        history.save_append.assert_called_once()
        # save_append gets the single-element prepared list.
        saved_msgs = history.save_append.call_args.args[1]
        assert len(saved_msgs) == 1
        assert saved_msgs[0]["role"] == "user"

    async def test_append_applies_per_message_prepare(self) -> None:
        """``append`` routes through ``_prepare_turn`` so the
        per-message transformations (tool-result truncation,
        runtime-context tag stripping, empty-assistant skip) still
        apply — the append path must produce the same on-disk shape
        as the legacy ``record`` path."""
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )

        big = "x" * 2000
        await conv.append("test:1", {"role": "tool", "content": big})

        assert len(session.messages) == 1
        assert len(session.messages[0]["content"]) < 600
        assert "truncated" in session.messages[0]["content"]

    async def test_append_skips_write_for_dropped_message(self) -> None:
        """Messages that ``_prepare_turn`` drops (empty assistant,
        runtime-tag-only user) mustn't trigger a ``save_append`` call
        — a no-op write on a fresh session would otherwise create the
        session file with only the metadata header."""
        session = Session(key="test:1")
        history = _make_mock_history(session)
        conv = DefaultConversation(
            history=history,
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )

        await conv.append("test:1", {"role": "assistant", "content": None})

        history.save_append.assert_not_called()
        assert len(session.messages) == 0

    async def test_append_does_not_fire_hooks(self) -> None:
        """Hooks belong to ``post_turn`` — a per-message append
        firing hooks would run end-of-turn callbacks after every
        tool result."""
        session = Session(key="test:1")
        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            bus=bus,
        )

        await conv.append("test:1", {"role": "user", "content": "hi"})

        bus.publish_inbound.assert_not_called()

    async def test_post_turn_delegates_to_fire_agent_hooks(self) -> None:
        """``post_turn`` owns the hook-firing half of the legacy
        ``record`` — assert the delegation rather than the nested
        bus-publish so this test doesn't have to mock the skills
        loader's hook discovery too."""
        session = Session(key="test:1")
        bus = MagicMock()
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            bus=bus,
        )
        hooks = AsyncMock()
        conv._fire_agent_hooks = hooks  # type: ignore[method-assign]

        await conv.post_turn("test:1")

        hooks.assert_awaited_once_with("test:1")

    async def test_post_turn_skips_hook_turn(self) -> None:
        """Hook turns (``channel="hook"``) must not re-fire hooks —
        prevents recursion when an agent_end hook calls back into
        the bot."""
        session = Session(key="test:1")
        bus = MagicMock()
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            bus=bus,
        )
        conv._turn_channel = "hook"
        hooks = AsyncMock()
        conv._fire_agent_hooks = hooks  # type: ignore[method-assign]

        await conv.post_turn("test:1")

        hooks.assert_not_called()

    async def test_post_turn_skips_when_no_bus(self) -> None:
        """No bus means no hook dispatch — same guard shape as the
        legacy ``record``."""
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        hooks = AsyncMock()
        conv._fire_agent_hooks = hooks  # type: ignore[method-assign]

        await conv.post_turn("test:1")

        hooks.assert_not_called()

    async def test_clear_success(self) -> None:
        session = Session(key="test:1")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        result = await conv.clear("test:1")
        assert result is True

    async def test_clear_empty_session(self) -> None:
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        result = await conv.clear("test:1")
        assert result is True

    async def test_clear_does_not_archive(self) -> None:
        """``clear`` no longer auto-summarizes the session into
        long-term memory before deleting it. Archival is the caller's
        responsibility — call ``policy.transform(reader, budget=0)`` (or
        ``memory.summarize`` directly) before ``clear`` if needed."""
        session = Session(key="test:1")
        session.messages = [{"role": "user", "content": "hi"}]
        memory = _make_mock_memory()
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=memory,
            prompt=_make_mock_prompt(),
        )
        await conv.clear("test:1")
        memory.summarize.assert_not_called()

    def test_list_sessions(self) -> None:
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        sessions = conv.list_sessions()
        assert sessions == [{"key": "test:1"}]

    def test_create_classmethod(self, tmp_path: Path) -> None:
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        conv = DefaultConversation.create(tmp_path, provider, "test-model")
        assert isinstance(conv, DefaultConversation)


# ---------------------------------------------------------------------------
# Additional coverage: skills builtin, memory edge cases, context paths
# ---------------------------------------------------------------------------


class TestSkillsLoaderBuiltin:
    def _make_builtin_skill(self, builtin_dir: Path, name: str, content: str = "# Skill") -> None:
        skill_dir = builtin_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content)

    def test_list_skills_from_builtin(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtins"
        self._make_builtin_skill(builtin, "builtin_skill")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)
        skills = loader.list_skills()
        assert any(s["name"] == "builtin_skill" for s in skills)

    def test_load_skill_from_builtin(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtins"
        self._make_builtin_skill(builtin, "builtin_skill", "# Builtin content")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)
        assert loader.load_skill("builtin_skill") == "# Builtin content"

    def test_workspace_takes_priority_over_builtin(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtins"
        self._make_builtin_skill(builtin, "shared", "# Builtin")
        skill_dir = tmp_path / "skills" / "shared"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("# Workspace")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)
        assert loader.load_skill("shared") == "# Workspace"

    def test_build_skills_summary_unavailable_shows_requires(self, tmp_path: Path) -> None:
        content = '---\nmetadata: {"exoclaw": {"requires": {"bins": ["definitely_not_installed_xyz"]}}}\n---\n# Skill'
        skill_dir = tmp_path / "skills" / "missing_dep"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content)
        loader = SkillsLoader(tmp_path)
        summary = loader.build_skills_summary()
        assert "missing_dep" in summary
        assert 'available="false"' in summary

    def test_get_skill_hook_scripts_finds_hook(self, tmp_path: Path) -> None:
        hook_path = tmp_path / "skills" / "myhook" / "hooks" / "exoclaw" / "pre_turn"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/sh\necho hi")
        hook_path.chmod(0o755)
        loader = SkillsLoader(tmp_path)
        scripts = loader.get_skill_hook_scripts("pre_turn")
        assert any(p.name == "pre_turn" for p in scripts)

    def test_get_skill_hook_scripts_legacy_nanobot_path(self, tmp_path: Path) -> None:
        hook_path = tmp_path / "skills" / "myhook" / "hooks" / "nanobot" / "pre_turn"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/sh\necho hi")
        hook_path.chmod(0o755)
        loader = SkillsLoader(tmp_path)
        scripts = loader.get_skill_hook_scripts("pre_turn")
        assert any(p.name == "pre_turn" for p in scripts)

    def test_get_missing_requirements_bin_and_env(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        missing = loader._get_missing_requirements(
            {
                "requires": {
                    "bins": ["definitely_not_installed_xyz"],
                    "env": ["DEFINITELY_NOT_SET_XYZ"],
                }
            }
        )
        assert "CLI:" in missing
        assert "ENV:" in missing


class TestMemoryStoreEdgeCases:
    async def test_summarize_message_with_tools_used(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = {"history_entry": "entry", "memory_update": "update"}
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")
        msgs = [
            {
                "role": "user",
                "content": "run it",
                "timestamp": "2024-01-01T00:00",
                "tools_used": ["exec"],
            },
        ]
        assert await store.summarize(msgs, archive_all=True) == "entry"

    async def test_summarize_non_string_history_entry(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        # Non-string history_entry — backend coerces via json.dumps before
        # writing, so summarize still succeeds.
        tc.arguments = {"history_entry": {"nested": "dict"}, "memory_update": "update"}
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")
        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        entry = await store.summarize(msgs, archive_all=True)
        assert entry is not None
        assert "nested" in entry

    async def test_summarize_empty_list_args(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = []
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")
        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        assert await store.summarize(msgs, archive_all=True) is None

    async def test_policy_advances_boundary_past_tool_pair(self, tmp_path: Path) -> None:
        """Regression: consolidation boundary must not leave orphan tool_results.

        Reproduces the MiniMax "tool result's tool id(...) not found" 400 seen
        in Luna's long autonomous agent runs. Scenario: one user message at the
        start, then a long sequence of assistant-tool_calls / tool-result pairs
        with no further user messages. When consolidation cuts between an
        assistant message and its corresponding tool result, the kept tail
        starts with a tool_result whose tool_call_id is in the archived region.
        get_history()'s "drop leading non-user" guard relies on a user message
        existing in the tail — it doesn't fire here, so the orphan passes
        through to the provider.
        """
        from exoclaw_conversation.summarizing_policy import SummarizingConsolidationPolicy

        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = {"history_entry": "summary", "memory_update": "facts"}
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")

        # 20 messages: 1 user at start, then 9 asst/tool pairs, then asst "done".
        # With memory_window=14, the next-chunk slice ends mid-pair if the
        # policy doesn't repair the boundary.
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "start", "timestamp": "2024-01-01T00:00"},
        ]
        for n in range(1, 10):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"T{n}",
                            "type": "function",
                            "function": {"name": "x", "arguments": "{}"},
                        }
                    ],
                    "timestamp": f"2024-01-01T00:{n:02d}",
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"T{n}",
                    "content": f"result {n}",
                    "timestamp": f"2024-01-01T00:{n:02d}",
                }
            )
        messages.append({"role": "assistant", "content": "done", "timestamp": "2024-01-01T00:19"})

        # Drive the policy directly with a list-backed reader so the test
        # exercises the boundary-repair logic without a full
        # SessionManager round-trip.
        policy = SummarizingConsolidationPolicy(
            memory=store, state_dir=tmp_path, memory_window=14
        )
        reader = _MockReader("ut:test", messages)
        await policy.on_turn_complete(reader)

        # The naive boundary would land mid-pair, splitting an
        # asst(tool_calls=Tn) from its tool(Tn). The repair pass must
        # advance past the orphan; verify the resulting view contains
        # no tool_results whose tool_call_id appears only in the
        # archived prefix.
        view = [m async for m in policy.transform(reader)]
        seen_tool_ids: set[str] = set()
        orphans: list[str] = []
        for m in view:
            if m.get("role") == "assistant":
                for tc_entry in m.get("tool_calls") or []:
                    if tid := tc_entry.get("id"):
                        seen_tool_ids.add(tid)
            elif m.get("role") == "tool":
                tid = m.get("tool_call_id")
                if tid and tid not in seen_tool_ids:
                    orphans.append(tid)
        assert not orphans, (
            f"transform leaked orphan tool_call_ids {orphans} — "
            f"consolidation boundary split a tool_use/tool_result pair. "
            f"This is what makes MiniMax return 400 invalid_params."
        )


class TestContextBuilderExtra:
    def _make_skill(
        self, workspace: Path, name: str, content: str = "# Skill", always: bool = False
    ) -> None:
        skill_dir = workspace / "skills" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        if always:
            content = f'---\nmetadata: {{"exoclaw": {{"always": true}}}}\n---\n{content}'
        (skill_dir / "SKILL.md").write_text(content)

    def test_build_system_prompt_with_always_skills(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "always_tool", "# Always skill content", always=True)
        builder = ContextBuilder(tmp_path)
        prompt = builder.build_system_prompt()
        assert "Always skill content" in prompt

    def test_build_system_prompt_with_skill_names(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "extra_skill", "# Extra skill content")
        builder = ContextBuilder(tmp_path)
        prompt = builder.build_system_prompt(skill_names=["extra_skill"])
        assert "Extra skill content" in prompt

    def test_build_system_prompt_with_bootstrap_hook(self, tmp_path: Path) -> None:
        hook = tmp_path / "skills" / "hook_skill" / "hooks" / "exoclaw" / "bootstrap.md"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("# Bootstrap injection")
        builder = ContextBuilder(tmp_path)
        prompt = builder.build_system_prompt()
        assert "Bootstrap injection" in prompt

    def test_build_messages_with_skill_names(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "test_skill", "# skill body")
        builder = ContextBuilder(tmp_path)
        msgs = builder.build_messages(history=[], current_message="hi", skill_names=["test_skill"])
        assert msgs[0]["role"] == "system"

    def test_add_assistant_message_with_reasoning(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        msgs: list[dict[str, Any]] = []
        builder.add_assistant_message(msgs, "hello", reasoning_content="<think>...</think>")
        assert msgs[0]["reasoning_content"] == "<think>...</think>"

    def test_add_assistant_message_with_thinking_blocks(self, tmp_path: Path) -> None:
        builder = ContextBuilder(tmp_path)
        msgs: list[dict[str, Any]] = []
        builder.add_assistant_message(
            msgs, "hello", thinking_blocks=[{"type": "thinking", "thinking": "..."}]
        )
        assert msgs[0]["thinking_blocks"] is not None


class TestDefaultConversationRecoverFromOverflow:
    """``DefaultConversation.recover_from_overflow`` is the
    Conversation-side seam consumed by AgentLoop (via
    ``Executor.recover_from_overflow``) on
    ``ContextWindowExceededError``. It delegates to the
    consolidation policy and re-assembles the prompt from the
    post-recovery view."""

    async def test_returns_none_when_policy_lacks_method(self) -> None:
        """A policy that doesn't implement ``recover_from_overflow``
        (e.g. the inline ``_NoOpPolicy``) makes the conversation give
        up — there's no way to make progress."""
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        result = await conv.recover_from_overflow("test:1")
        assert result is None

    async def test_returns_none_when_policy_cant_advance(self) -> None:
        """Policy returns ``False`` (nothing left to summarize) →
        conversation surfaces ``None`` to the loop."""

        class _StuckPolicy:
            def transform(self, reader, *, budget=None):  # type: ignore[no-untyped-def]
                async def _gen():  # type: ignore[no-untyped-def]
                    return
                    yield  # unreachable; satisfies async-generator shape

                return _gen()

            async def on_turn_complete(self, reader) -> None:  # type: ignore[no-untyped-def]
                return None

            async def recover_from_overflow(self, reader) -> bool:  # type: ignore[no-untyped-def]
                return False

        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            consolidation_policy=_StuckPolicy(),  # type: ignore[arg-type]
        )
        result = await conv.recover_from_overflow("test:1")
        assert result is None

    async def test_returns_assembled_prompt_on_success(self) -> None:
        """When the policy advances, the conversation re-materializes
        the view and prepends the system prompt. Returns the new
        message list — caller passes to ``executor.set_messages``."""

        class _RecoveringPolicy:
            def transform(self, reader, *, budget=None):  # type: ignore[no-untyped-def]
                async def _gen():  # type: ignore[no-untyped-def]
                    yield {"role": "system", "content": "## Previous Session Summary\nold"}
                    yield {"role": "user", "content": "in-flight question"}

                return _gen()

            async def on_turn_complete(self, reader) -> None:  # type: ignore[no-untyped-def]
                return None

            async def recover_from_overflow(self, reader) -> bool:  # type: ignore[no-untyped-def]
                return True

        prompt = MagicMock(spec=["build_messages", "build_system_prompt", "get_active_optional_tools"])
        prompt.build_system_prompt = MagicMock(return_value="SYSTEM PROMPT")
        prompt.get_active_optional_tools = MagicMock(return_value=set())

        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=prompt,
            consolidation_policy=_RecoveringPolicy(),  # type: ignore[arg-type]
        )
        result = await conv.recover_from_overflow("test:1")
        assert result is not None
        # First message is the system prompt the conversation prepended.
        assert result[0] == {"role": "system", "content": "SYSTEM PROMPT"}
        # Followed by whatever the policy emitted via transform.
        assert result[1] == {
            "role": "system",
            "content": "## Previous Session Summary\nold",
        }
        assert result[-1] == {"role": "user", "content": "in-flight question"}

    async def test_returns_view_only_when_prompt_lacks_build_system_prompt(self) -> None:
        """Custom PromptBuilder that doesn't implement
        ``build_system_prompt`` falls through to a no-system-prompt
        path. The result is still a valid (leaner) prompt list."""

        class _RecoveringPolicy:
            def transform(self, reader, *, budget=None):  # type: ignore[no-untyped-def]
                async def _gen():  # type: ignore[no-untyped-def]
                    yield {"role": "user", "content": "carryover"}

                return _gen()

            async def on_turn_complete(self, reader) -> None:  # type: ignore[no-untyped-def]
                return None

            async def recover_from_overflow(self, reader) -> bool:  # type: ignore[no-untyped-def]
                return True

        # spec'd MagicMock with build_messages but no build_system_prompt.
        prompt = MagicMock(spec=["build_messages", "get_active_optional_tools"])
        prompt.get_active_optional_tools = MagicMock(return_value=set())

        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=prompt,
            consolidation_policy=_RecoveringPolicy(),  # type: ignore[arg-type]
        )
        result = await conv.recover_from_overflow("test:1")
        assert result == [{"role": "user", "content": "carryover"}]


class TestDefaultConversationExtra:
    async def test_clear_exception_returns_false(self) -> None:
        session = Session(key="test:1")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        history = _make_mock_history(session)
        # Make ``save`` raise so ``clear`` exits via the except branch.
        history.save.side_effect = Exception("boom")
        conv = DefaultConversation(
            history=history,
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        result = await conv.clear("test:1")
        assert result is False

    async def test_clear_deletes_sidecar_when_history_exposes_dir(
        self, tmp_path: Path
    ) -> None:
        """If the ``HistoryStore`` exposes a ``sessions_dir``, ``clear``
        also deletes the policy sidecar that lives next to the session
        JSONL. Without this, a sidecar from a deleted session would
        re-seed itself on the next ``transform`` call."""
        from exoclaw_conversation import _consolidation_state as ss
        from exoclaw_conversation.session.manager import SessionManager

        sessions = SessionManager(tmp_path)
        sess = sessions.get_or_create("ut:bye")
        sess.add_message("user", "hi")
        sessions.save(sess)
        # Seed a sidecar so ``clear`` has something to delete.
        ss.save_state(
            tmp_path / "sessions",
            "ut:bye",
            ss.ConsolidationState(summarized_through=1, summary="x"),
        )
        sidecar = tmp_path / "sessions" / "ut_bye.consolidation.json"
        assert sidecar.exists()

        conv = DefaultConversation(
            history=sessions,
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        assert await conv.clear("ut:bye") is True
        assert not sidecar.exists()

    async def test_schedule_maintenance_is_idempotent(self) -> None:
        """If maintenance is already running for a session, a second
        ``post_turn`` doesn't spawn a duplicate task — the policy
        runs once per session at a time."""

        class _GatedPolicy:
            def __init__(self) -> None:
                self.gate = asyncio.Event()
                self.calls = 0

            def transform(self, reader, *, budget=None):  # type: ignore[no-untyped-def]
                async def _gen():  # type: ignore[no-untyped-def]
                    async for m in reader.stream():
                        yield m

                return _gen()

            async def on_turn_complete(self, reader) -> None:  # type: ignore[no-untyped-def]
                self.calls += 1
                await self.gate.wait()

        policy = _GatedPolicy()
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            consolidation_policy=policy,  # type: ignore[arg-type]
        )
        await conv.post_turn("test:1")
        await conv.post_turn("test:1")  # second post_turn — should be a no-op
        await asyncio.sleep(0.01)
        assert policy.calls == 1
        # Release the gate so the task finishes cleanly.
        policy.gate.set()
        await asyncio.gather(*list(conv._consolidation_tasks), return_exceptions=True)

    async def test_maintenance_swallows_policy_exception(self) -> None:
        """A policy that raises during ``on_turn_complete`` must not
        bubble out and crash the next turn — the task logs and
        unregisters itself so the session lock releases."""

        class _CrashyPolicy:
            def transform(self, reader, *, budget=None):  # type: ignore[no-untyped-def]
                async def _gen():  # type: ignore[no-untyped-def]
                    async for m in reader.stream():
                        yield m

                return _gen()

            async def on_turn_complete(self, reader) -> None:  # type: ignore[no-untyped-def]
                raise RuntimeError("policy blew up")

        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            consolidation_policy=_CrashyPolicy(),  # type: ignore[arg-type]
        )
        await conv.post_turn("test:1")
        await asyncio.gather(*list(conv._consolidation_tasks), return_exceptions=True)
        # Session lock released — a second post_turn re-spawns cleanly.
        assert "test:1" not in conv._consolidating

    async def test_record_fires_hooks_when_bus_present(self) -> None:
        """Legacy batch ``record`` fires end-of-turn hooks itself
        (the post_turn path runs only under the append surface).
        Verify the hook-fire delegation when a bus is configured."""

        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            bus=bus,
        )
        hooks = AsyncMock()
        conv._fire_agent_hooks = hooks  # type: ignore[method-assign]
        await conv.record("test:1", [{"role": "user", "content": "hi"}])
        hooks.assert_awaited_once_with("test:1")

    async def test_fire_agent_hooks_publishes_inbound(self) -> None:
        """``_fire_agent_hooks`` reaches into the prompt's skills loader,
        finds ``agent_end`` hooks, and publishes one ``InboundMessage``
        per hook on the bus. Failures on a single publish are logged
        but don't stop subsequent hooks from firing."""

        bus = MagicMock()
        bus.publish_inbound = AsyncMock()

        # Two hooks: first publishes fine, second raises — both must
        # be attempted (failure logged, loop continues).
        hook_a = MagicMock(skill_name="a", prompt="run-a", tools=["t1"], skills=["s1"])
        hook_b = MagicMock(skill_name="b", prompt="run-b", tools=[], skills=[])

        skills_loader = MagicMock()
        skills_loader.get_agent_hooks = MagicMock(return_value=[hook_a, hook_b])

        prompt = _make_mock_prompt()
        prompt.skills = skills_loader

        # Second publish raises so we exercise the except branch.
        bus.publish_inbound = AsyncMock(side_effect=[None, RuntimeError("downstream")])

        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=prompt,
            bus=bus,
        )
        conv._turn_chat_id = "topic-x"
        await conv._fire_agent_hooks("test:1")

        assert bus.publish_inbound.await_count == 2
        first_msg = bus.publish_inbound.await_args_list[0].args[0]
        assert first_msg.channel == "hook"
        assert first_msg.chat_id == "topic-x"
        assert first_msg.metadata["hook_skill"] == "a"

    async def test_fire_agent_hooks_skips_when_no_skills_loader(self) -> None:
        """A prompt builder without a ``skills`` attribute (e.g. a stub)
        short-circuits early — no bus calls."""

        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        prompt = MagicMock(spec=["build_messages"])
        # spec excludes ``skills`` so getattr returns None.
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=prompt,
            bus=bus,
        )
        await conv._fire_agent_hooks("test:1")
        bus.publish_inbound.assert_not_called()

    async def test_fire_agent_hooks_skips_when_no_hooks(self) -> None:
        """Skills loader present but no ``agent_end`` hooks registered
        — the empty-hooks early-return."""

        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        skills_loader = MagicMock()
        skills_loader.get_agent_hooks = MagicMock(return_value=[])
        prompt = _make_mock_prompt()
        prompt.skills = skills_loader
        conv = DefaultConversation(
            history=_make_mock_history(),
            memory=_make_mock_memory(),
            prompt=prompt,
            bus=bus,
        )
        await conv._fire_agent_hooks("test:1")
        bus.publish_inbound.assert_not_called()

    async def test_record_drops_user_with_only_filtered_list_content(self) -> None:
        """A user message whose only list-content blocks all get
        filtered (e.g. only a runtime-context tag) becomes empty —
        ``_prepare_turn`` drops it rather than persisting an empty
        message that would confuse the LLM."""
        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{CONV_TAG}\nmetadata only"},
            ],
        }
        await conv.record("test:1", [msg])
        assert session.messages == []


# ---------------------------------------------------------------------------
# active_tools() unit tests
# ---------------------------------------------------------------------------


class TestActiveTools:
    def _make_skill(self, workspace: Path, name: str, content: str) -> None:
        skill_dir = workspace / "skills" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content)

    def test_active_tools_empty_without_tools_frontmatter(self, tmp_path: Path) -> None:
        """Always-active skill with no tools: key contributes nothing."""
        self._make_skill(
            tmp_path,
            "plain",
            '---\nmetadata: {"exoclaw": {"always": true}}\n---\n# skill',
        )
        builder = ContextBuilder(tmp_path)
        builder.build_system_prompt()
        assert builder.get_active_optional_tools() == set()

    def test_active_tools_returned_from_always_skill(self, tmp_path: Path) -> None:
        """Always-active skill with tools: frontmatter surfaces those names."""
        self._make_skill(
            tmp_path,
            "sentry",
            '---\ntools: mcp_sentry_list_issues, mcp_sentry_resolve_issue\nmetadata: {"exoclaw": {"always": true}}\n---\n# Sentry skill',
        )
        builder = ContextBuilder(tmp_path)
        builder.build_system_prompt()
        tools = builder.get_active_optional_tools()
        assert "mcp_sentry_list_issues" in tools
        assert "mcp_sentry_resolve_issue" in tools

    def test_default_conversation_active_tools_no_skills(self, tmp_path: Path) -> None:
        """Returns empty set when workspace has no skills."""
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        conv = DefaultConversation.create(tmp_path, provider, "test-model")
        assert conv.active_tools() == set()

    def test_default_conversation_active_tools_with_always_skill(self, tmp_path: Path) -> None:
        """Reflects tools from always-active skill after build_system_prompt fires."""
        self._make_skill(
            tmp_path,
            "sentry",
            '---\ntools: mcp_sentry_list_issues\nmetadata: {"exoclaw": {"always": true}}\n---\n# Sentry skill',
        )
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        conv = DefaultConversation.create(tmp_path, provider, "test-model")
        # Trigger skill resolution the same way AgentLoop does — via build_prompt
        conv.prompt.build_system_prompt()  # type: ignore[attr-defined]
        assert "mcp_sentry_list_issues" in conv.active_tools()

    def test_default_conversation_active_tools_delegates_to_prompt(self) -> None:
        """Delegates to prompt.get_active_optional_tools() when present."""
        p = MagicMock(spec=["build_messages", "get_active_optional_tools"])
        p.build_messages.return_value = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "h"},
        ]
        p.get_active_optional_tools.return_value = {"my_tool", "other_tool"}
        conv = DefaultConversation(
            history=_make_mock_history(), memory=_make_mock_memory(), prompt=p
        )
        assert conv.active_tools() == {"my_tool", "other_tool"}


# ---------------------------------------------------------------------------
# active_tools() integration tests — real AgentLoop + real DefaultConversation
# ---------------------------------------------------------------------------


def _make_provider_response() -> MagicMock:
    r = MagicMock()
    r.has_tool_calls = False
    r.content = "done"
    r.finish_reason = "stop"
    r.tool_calls = []
    r.reasoning_content = None
    r.thinking_blocks = None
    return r


def _make_optional_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = f"Tool {name}"
    t.parameters = {"type": "object", "properties": {}}
    return t


class TestActiveToolsLoopIntegration:
    """Real AgentLoop + real DefaultConversation — no mocking of the loop."""

    def _make_skill(self, workspace: Path, name: str, content: str) -> None:
        skill_dir = workspace / "skills" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content)

    async def test_optional_tool_surfaced_when_skill_active(self, tmp_path: Path) -> None:
        """Optional tool appears in the tool list sent to the LLM when declared by an always-active skill."""
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.agent.tools.registry import ToolRegistry
        from exoclaw.bus.queue import MessageBus

        self._make_skill(
            tmp_path,
            "sentry",
            '---\ntools: spy_tool\nmetadata: {"exoclaw": {"always": true}}\n---\n# Sentry skill',
        )

        memory_provider = MagicMock()
        memory_provider.get_default_model.return_value = "test-model"
        conv = DefaultConversation.create(tmp_path, memory_provider, "test-model")

        registry = ToolRegistry()
        registry.register(_make_optional_tool("spy_tool"), optional=True)

        tools_seen: list[list[dict[str, object]]] = []

        async def _chat(**kwargs: object) -> MagicMock:
            tools_seen.append(list(kwargs.get("tools") or []))  # type: ignore[arg-type]
            return _make_provider_response()

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(side_effect=_chat)

        bus = MessageBus()
        loop = AgentLoop(bus=bus, provider=provider, conversation=conv, registry=registry)
        await loop.process_direct("hello", session_key="test:main", channel="cli", chat_id="main")

        assert tools_seen, "provider.chat was never called"
        tool_names = [t["function"]["name"] for t in tools_seen[0]]  # type: ignore[index]
        assert "spy_tool" in tool_names

    async def test_optional_tool_hidden_when_no_skill_declares_it(self, tmp_path: Path) -> None:
        """Optional tool absent from LLM tool list when no skill declares it."""
        from exoclaw.agent.loop import AgentLoop
        from exoclaw.agent.tools.registry import ToolRegistry
        from exoclaw.bus.queue import MessageBus

        memory_provider = MagicMock()
        memory_provider.get_default_model.return_value = "test-model"
        conv = DefaultConversation.create(tmp_path, memory_provider, "test-model")

        registry = ToolRegistry()
        registry.register(_make_optional_tool("spy_tool"), optional=True)

        tools_seen: list[list[dict[str, object]]] = []

        async def _chat(**kwargs: object) -> MagicMock:
            tools_seen.append(list(kwargs.get("tools") or []))  # type: ignore[arg-type]
            return _make_provider_response()

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(side_effect=_chat)

        bus = MessageBus()
        loop = AgentLoop(bus=bus, provider=provider, conversation=conv, registry=registry)
        await loop.process_direct("hello", session_key="test:main", channel="cli", chat_id="main")

        assert tools_seen, "provider.chat was never called"
        tool_names = [t["function"]["name"] for t in tools_seen[0]]  # type: ignore[index]
        assert "spy_tool" not in tool_names
