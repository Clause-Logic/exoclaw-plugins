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

    def test_get_history_respects_last_consolidated(self) -> None:
        s = Session(key="test")
        for i in range(10):
            s.messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": str(i)})
        s.last_consolidated = 6
        history = s.get_history()
        assert all(m["content"] in [str(i) for i in range(6, 10)] for m in history)

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
        s.last_consolidated = 1
        s.clear()
        assert s.messages == []
        assert s.last_consolidated == 0


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
        assert len(s2.messages) == 1  # unconsolidated
        assert s2.messages[0]["content"] == "hello"
        assert s2.last_consolidated == 0

    def test_save_and_load_consolidated_not_in_ram(self, tmp_path: Path) -> None:
        """Consolidated messages stay on disk, not loaded into RAM."""
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:123")
        s.add_message("user", "old")
        s.add_message("assistant", "old reply")
        s.add_message("user", "new")
        s.last_consolidated = 2
        s.total_messages = 3
        mgr.save(s)

        mgr2 = SessionManager(tmp_path)
        s2 = mgr2.get_or_create("ch:123")
        assert s2.total_messages == 3
        assert s2.last_consolidated == 2
        assert len(s2.messages) == 1  # only unconsolidated tail
        assert s2.messages[0]["content"] == "new"

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

    def test_load_skips_consolidated_lines(self, tmp_path: Path) -> None:
        """_load must not hold consolidated messages in memory."""
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:big")
        for i in range(100):
            s.add_message("user" if i % 2 == 0 else "assistant", f"msg-{i}")
        s.last_consolidated = 90
        mgr.save(s)

        mgr2 = SessionManager(tmp_path)
        s2 = mgr2.get_or_create("ch:big")
        # Only 10 unconsolidated messages in RAM
        assert len(s2.messages) == 10
        assert s2.messages[0]["content"] == "msg-90"
        assert s2.total_messages == 100

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

    async def test_consolidate_no_provider(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        result = await store.consolidate(session, archive_all=True)
        assert result is False

    async def test_consolidate_empty_session(self, tmp_path: Path) -> None:
        provider = MagicMock()
        store = MemoryStore(tmp_path, provider=provider, model="test-model")
        session = Session(key="test")
        result = await store.consolidate(session, archive_all=False)
        assert result is True

    async def test_consolidate_calls_provider(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = {"history_entry": "[2024-01-01] summary", "memory_update": "facts"}
        response.tool_calls = [tc]

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        session = Session(key="test")
        session.messages = [
            {"role": "user", "content": "hello", "timestamp": "2024-01-01T00:00"},
            {"role": "assistant", "content": "hi", "timestamp": "2024-01-01T00:01"},
        ]
        result = await store.consolidate(session, archive_all=True)
        assert result is True
        assert "facts" in store.read_long_term()

    async def test_consolidate_no_tool_call(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = False

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        result = await store.consolidate(session, archive_all=True)
        assert result is False

    async def test_consolidate_provider_exception(self, tmp_path: Path) -> None:
        provider = MagicMock()
        provider.chat = AsyncMock(side_effect=Exception("boom"))
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        result = await store.consolidate(session, archive_all=True)
        assert result is False

    async def test_consolidate_args_as_string(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = json.dumps({"history_entry": "entry", "memory_update": "update"})
        response.tool_calls = [tc]

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        result = await store.consolidate(session, archive_all=True)
        assert result is True

    async def test_consolidate_args_as_list(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = [{"history_entry": "entry", "memory_update": "update"}]
        response.tool_calls = [tc]

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        result = await store.consolidate(session, archive_all=True)
        assert result is True

    async def test_consolidate_args_unexpected_type(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = 42
        response.tool_calls = [tc]

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="test-model")

        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        result = await store.consolidate(session, archive_all=True)
        assert result is False


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


def _make_mock_history(session: Session | None = None) -> MagicMock:
    s = session or Session(key="test:1")
    h = MagicMock(
        spec=[
            "get_or_create",
            "save",
            "save_append",
            "save_metadata",
            "load_range",
            "read_history",
            "invalidate",
            "list_sessions",
        ]
    )
    h.get_or_create.return_value = s
    h.list_sessions.return_value = [{"key": "test:1"}]
    h.load_range.return_value = []
    # Match the protocol's default impl: delegate to session.get_history so
    # tests that seed session.messages keep working without needing to
    # configure read_history separately.
    h.read_history.side_effect = lambda key, max_messages=None: s.get_history(
        max_messages=max_messages or 500
    )
    return h


def _make_mock_memory() -> MagicMock:
    m = MagicMock(spec=["get_memory_context", "consolidate", "consolidate_messages"])
    m.get_memory_context.return_value = ""
    m.consolidate = AsyncMock(return_value=True)
    m.consolidate_messages = AsyncMock(return_value=True)
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

    async def test_build_prompt_triggers_consolidation(self) -> None:
        session = Session(key="test:1")
        session.messages = [{"role": "user", "content": str(i)} for i in range(110)]
        session.last_consolidated = 0

        memory = _make_mock_memory()
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=memory,
            prompt=_make_mock_prompt(),
            memory_window=100,
        )
        await conv.build_prompt("test:1", "hello")
        await asyncio.sleep(0.05)
        # consolidation now goes through consolidate_messages (disk-backed path)
        # or consolidate (legacy path via ConsolidationPolicy)
        assert memory.consolidate_messages.called or memory.consolidate.called

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

    # ------------------------------------------------------------------
    # load_persisted_history — sync reader used by phase 2b PriorSource
    # (added for exoclaw>=0.20.0)
    # ------------------------------------------------------------------

    def test_load_persisted_history_returns_session_messages(self) -> None:
        """Baseline: ``load_persisted_history`` returns whatever
        ``session.get_history(max_messages=memory_window)`` returns —
        the same path ``build_prompt`` uses to assemble its history
        slice, minus the system prompt / runtime context / new user
        message that wrap it.
        """
        session = Session(key="test:1")
        session.messages = [
            {"role": "user", "content": "msg-1"},
            {"role": "assistant", "content": "reply-1"},
        ]
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )

        result = conv.load_persisted_history("test:1")

        assert result == [
            {"role": "user", "content": "msg-1"},
            {"role": "assistant", "content": "reply-1"},
        ]

    def test_load_persisted_history_is_sync(self) -> None:
        """Phase 2b's executor ``PriorSource`` closure invokes this
        method synchronously from inside an already-running event
        loop (the agent loop's iteration). It must not be a
        coroutine — a coroutine return would bypass the event loop
        machinery and leak an un-awaited warning at best, or return
        a coroutine object instead of a list at worst.
        """
        import asyncio

        session = Session(key="test:1")
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )

        # Direct call returns a list, not a coroutine.
        result = conv.load_persisted_history("test:1")
        assert not asyncio.iscoroutine(result)
        assert isinstance(result, list)

    def test_load_persisted_history_does_not_trigger_consolidation(self) -> None:
        """Unlike ``build_prompt``, this method is meant to be called
        on every LLM iteration as part of the PriorSource closure.
        Triggering consolidation per iteration would DoS the memory
        backend and stall the loop — the contract is strictly read."""
        session = Session(key="test:1")
        memory = _make_mock_memory()
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=memory,
            prompt=_make_mock_prompt(),
        )

        for _ in range(5):
            conv.load_persisted_history("test:1")

        # None of the consolidation entry points fired.
        assert memory.consolidate.call_count == 0
        assert memory.consolidate_messages.call_count == 0

    def test_load_persisted_history_respects_memory_window(self) -> None:
        """``session.get_history`` trims to ``memory_window`` (and
        aligns the start to a user turn so tool-result messages don't
        get orphaned at the prefix). ``load_persisted_history``
        inherits that — otherwise callers would pass a different-size
        list to the LLM than ``build_prompt`` did on the initial turn.
        """
        session = Session(key="test:1")
        # 10 messages; memory_window will be set to 4. Tail-4 starts
        # at msg-6 (user), which already satisfies the
        # "first msg is user" alignment — so get_history returns all 4.
        session.messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"}
            for i in range(10)
        ]
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
            memory_window=4,
        )

        result = conv.load_persisted_history("test:1")
        assert len(result) == 4
        assert result[0]["content"] == "msg-6"  # user
        assert result[-1]["content"] == "msg-9"  # assistant — most recent

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
        memory = _make_mock_memory()
        memory.consolidate = AsyncMock(return_value=True)
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=memory,
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

    async def test_clear_consolidation_failure(self) -> None:
        session = Session(key="test:1")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        memory = _make_mock_memory()
        memory.consolidate = AsyncMock(return_value=False)
        memory.consolidate_messages = AsyncMock(return_value=False)
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=memory,
            prompt=_make_mock_prompt(),
        )
        result = await conv.clear("test:1")
        assert result is False

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
    async def test_consolidate_already_consolidated(self, tmp_path: Path) -> None:
        """all messages already consolidated (last_consolidated == len)"""
        provider = MagicMock()
        store = MemoryStore(tmp_path, provider=provider, model="m")
        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi"}]
        session.last_consolidated = 1
        result = await store.consolidate(session, archive_all=False, memory_window=10)
        assert result is True
        provider.chat.assert_not_called()

    async def test_consolidate_old_messages_empty_after_slice(self, tmp_path: Path) -> None:
        """slice produces empty old_messages"""
        provider = MagicMock()
        store = MemoryStore(tmp_path, provider=provider, model="m")
        session = Session(key="test")
        # 4 messages, keep_count = 5 (window//2=5), so slice is empty
        session.messages = [{"role": "user", "content": str(i)} for i in range(4)]
        session.last_consolidated = 0
        result = await store.consolidate(session, archive_all=False, memory_window=10)
        assert result is True

    async def test_consolidate_non_archive_updates_last_consolidated(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = {"history_entry": "entry", "memory_update": "update"}
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")
        session = Session(key="test")
        session.messages = [
            {"role": "user", "content": str(i), "timestamp": "2024-01-01T00:00"} for i in range(20)
        ]
        session.last_consolidated = 0
        result = await store.consolidate(session, archive_all=False, memory_window=10)
        assert result is True
        assert session.last_consolidated > 0

    async def test_consolidate_message_with_tools_used(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = {"history_entry": "entry", "memory_update": "update"}
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")
        session = Session(key="test")
        session.messages = [
            {
                "role": "user",
                "content": "run it",
                "timestamp": "2024-01-01T00:00",
                "tools_used": ["exec"],
            },
        ]
        result = await store.consolidate(session, archive_all=True)
        assert result is True

    async def test_consolidate_non_string_history_entry(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = {"history_entry": {"nested": "dict"}, "memory_update": "update"}
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")
        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        result = await store.consolidate(session, archive_all=True)
        assert result is True

    async def test_consolidate_empty_list_args(self, tmp_path: Path) -> None:
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = []
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")
        session = Session(key="test")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        result = await store.consolidate(session, archive_all=True)
        assert result is False

    async def test_consolidate_does_not_split_tool_pair(self, tmp_path: Path) -> None:
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
        response = MagicMock()
        response.has_tool_calls = True
        tc = MagicMock()
        tc.arguments = {"history_entry": "summary", "memory_update": "facts"}
        response.tool_calls = [tc]
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        store = MemoryStore(tmp_path, provider=provider, model="m")

        # 20 messages: 1 user at start, then 9 asst/tool pairs, then asst "done".
        # With memory_window=12, keep_count=6, last_consolidated lands at 14 —
        # between the asst at idx 13 and its tool result at idx 14.
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "start", "timestamp": "2024-01-01T00:00"},
        ]
        for n in range(1, 10):  # pairs 1..9 → indices 1..18
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

        session = Session(key="test")
        session.messages = messages
        session.last_consolidated = 0

        result = await store.consolidate(session, archive_all=False, memory_window=12)
        assert result is True
        # Naive boundary would be total-keep_count = 20-6 = 14, splitting the
        # asst(tool_calls=T7) at idx 13 from its tool(T7) at idx 14. The fix
        # advances the boundary past the orphan so the kept tail starts at 15.
        assert session.last_consolidated >= 15

        # Every tool message returned must have its tool_call_id introduced
        # earlier in the returned history by an assistant message's tool_calls.
        history = session.get_history(max_messages=500)
        seen_tool_ids: set[str] = set()
        orphans: list[str] = []
        for m in history:
            if m.get("role") == "assistant":
                for tc_entry in m.get("tool_calls") or []:
                    if tid := tc_entry.get("id"):
                        seen_tool_ids.add(tid)
            elif m.get("role") == "tool":
                tid = m.get("tool_call_id")
                if tid and tid not in seen_tool_ids:
                    orphans.append(tid)
        assert not orphans, (
            f"get_history leaked orphan tool_call_ids {orphans} — "
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


class TestDefaultConversationExtra:
    async def test_clear_exception_returns_false(self) -> None:
        session = Session(key="test:1")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        memory = _make_mock_memory()
        memory.consolidate = AsyncMock(side_effect=Exception("boom"))
        memory.consolidate_messages = AsyncMock(side_effect=Exception("boom"))
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=memory,
            prompt=_make_mock_prompt(),
        )
        result = await conv.clear("test:1")
        assert result is False


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
