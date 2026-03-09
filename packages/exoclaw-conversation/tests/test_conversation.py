"""Tests for exoclaw-conversation package."""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw_conversation.helpers import detect_image_mime, ensure_dir, safe_filename
from exoclaw_conversation.memory import MemoryStore
from exoclaw_conversation.session.manager import Session, SessionManager
from exoclaw_conversation.skills import SkillsLoader
from exoclaw_conversation.context import ContextBuilder, _RUNTIME_CONTEXT_TAG
from exoclaw_conversation.conversation import DefaultConversation, _RUNTIME_CONTEXT_TAG as CONV_TAG
from exoclaw_conversation.protocols import HistoryStore, MemoryBackend, PromptBuilder


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
        s.last_consolidated = 1
        mgr.save(s)

        mgr2 = SessionManager(tmp_path)
        s2 = mgr2.get_or_create("ch:123")
        assert len(s2.messages) == 1
        assert s2.messages[0]["content"] == "hello"
        assert s2.last_consolidated == 1

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
        assert loader._check_requirements({"requires": {"bins": ["definitely_not_installed_xyz"]}}) is False

    def test_check_requirements_missing_env(self) -> None:
        loader = SkillsLoader(Path("/tmp"))
        assert loader._check_requirements({"requires": {"env": ["DEFINITELY_NOT_SET_XYZ_ABC"]}}) is False

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
        assert "hello" in msgs[-1]["content"]

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
        msgs = builder.build_messages(history=[], current_message="look", media=["/nonexistent.png"])
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
    h = MagicMock(spec=["get_or_create", "save", "invalidate", "list_sessions"])
    h.get_or_create.return_value = s
    h.list_sessions.return_value = [{"key": "test:1"}]
    return h


def _make_mock_memory() -> MagicMock:
    m = MagicMock(spec=["get_memory_context", "consolidate"])
    m.get_memory_context.return_value = ""
    m.consolidate = AsyncMock(return_value=True)
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
        call_kwargs = conv.prompt.build_messages.call_args[1]
        assert "extra" in call_kwargs["extra_context"]

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
        memory.consolidate.assert_called_once()

    async def test_record_saves_turn(self) -> None:
        session = Session(key="test:1")
        history = _make_mock_history(session)
        conv = DefaultConversation(
            history=history,
            memory=_make_mock_memory(),
            prompt=_make_mock_prompt(),
        )
        await conv.record("test:1", [{"role": "user", "content": "hi"}])
        history.save.assert_called_once()

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
        missing = loader._get_missing_requirements({
            "requires": {
                "bins": ["definitely_not_installed_xyz"],
                "env": ["DEFINITELY_NOT_SET_XYZ"],
            }
        })
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
        session.messages = [{"role": "user", "content": str(i), "timestamp": "2024-01-01T00:00"} for i in range(20)]
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
            {"role": "user", "content": "run it", "timestamp": "2024-01-01T00:00", "tools_used": ["exec"]},
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


class TestContextBuilderExtra:
    def _make_skill(self, workspace: Path, name: str, content: str = "# Skill", always: bool = False) -> None:
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
        builder.add_assistant_message(msgs, "hello", thinking_blocks=[{"type": "thinking", "thinking": "..."}])
        assert msgs[0]["thinking_blocks"] is not None


class TestDefaultConversationExtra:
    async def test_clear_exception_returns_false(self) -> None:
        session = Session(key="test:1")
        session.messages = [{"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00"}]
        memory = _make_mock_memory()
        memory.consolidate = AsyncMock(side_effect=Exception("boom"))
        conv = DefaultConversation(
            history=_make_mock_history(session),
            memory=memory,
            prompt=_make_mock_prompt(),
        )
        result = await conv.clear("test:1")
        assert result is False
