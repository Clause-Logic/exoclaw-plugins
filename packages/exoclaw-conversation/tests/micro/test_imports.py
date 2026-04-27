"""MicroPython import smoke test for ``exoclaw-conversation``.

Pure-Python — no pytest. Runs under
``tests/_micropython_runner/run.py`` (driven via the workspace's
``mise run test-micro`` task) on a coverage-variant MicroPython
binary.

The bar here is "does the module graph import cleanly under MP".
Once import works, smaller per-class behavioural tests can land
in this file or sibling files.
"""


def test_helpers_imports():
    from exoclaw_conversation import helpers

    # ``ensure_dir`` and ``safe_filename`` are the public surface
    # callers reach for; verify they exist as callables.
    assert callable(helpers.ensure_dir)
    assert callable(helpers.safe_filename)


def test_session_manager_imports():
    from exoclaw_conversation.session.manager import Session, SessionManager

    # The dual-class pattern (CPython @dataclass / MP plain class)
    # lives behind the public ``Session`` name; both branches must
    # produce a constructable type.
    s = Session(key="ut:test")
    assert s.key == "ut:test"
    assert s.last_consolidated == 0
    assert s.total_messages == 0

    # SessionManager is the file-backed history store.
    assert callable(SessionManager)


def test_skills_imports():
    from exoclaw_conversation.skills import (
        AgentHook,
        LoadSkillResult,
        SkillsLoader,
    )

    # Both dataclass-or-plain dual classes must be constructable
    # with the documented kwargs.
    r = LoadSkillResult(content="hi")
    assert r.content == "hi"
    assert r.tool_names == []

    h = AgentHook(skill_name="s", hook_name="h", prompt="p")
    assert h.skill_name == "s"
    assert h.tools == []
    assert h.skills == []

    assert callable(SkillsLoader)


def test_memory_imports():
    from exoclaw_conversation.memory import MemoryStore

    assert callable(MemoryStore)


def test_context_imports():
    from exoclaw_conversation.context import ContextBuilder

    assert callable(ContextBuilder)


def test_conversation_imports():
    from exoclaw_conversation.conversation import DefaultConversation

    assert callable(DefaultConversation)
