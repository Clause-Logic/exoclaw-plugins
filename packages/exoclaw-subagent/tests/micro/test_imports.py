"""MicroPython import smoke test for ``exoclaw-subagent``.

Pure-Python — no pytest. Driven by the workspace's
``mise run test-micro`` task on a coverage-variant MicroPython
binary.

The real subagent lifecycle behaviour (spawn / cancel / batch
announcement) is exercised by the CPython test suite. This MP
gate verifies the import graph clears cleanly + the bits that
diverge between runtimes (``os.urandom`` task ids, ``Path`` from
``exoclaw._compat``, log contextvars via ``exoclaw._compat``) work
without ``uuid`` / ``pathlib`` / ``structlog`` / ``contextvars``
on MicroPython.
"""

import asyncio


def test_top_level_imports():
    from exoclaw_subagent import (
        AnnounceCallback,
        AsyncioSpawner,
        BatchSnapshot,
        BatchStore,
        InMemoryBatchStore,
        Runner,
        SpawnerFactory,
        SpawnManager,
        SpawnTool,
        SubagentHandle,
        SubagentManager,
        SubagentSpawner,
    )

    # Concrete classes are callable; the protocol bases shimmed by
    # ``typing.Protocol`` on MP are also callable (no-op base) so the
    # ``callable`` check works for both.
    assert callable(AsyncioSpawner)
    assert callable(SubagentManager)
    assert callable(SpawnTool)
    assert callable(InMemoryBatchStore)
    assert callable(BatchSnapshot)
    assert callable(BatchStore)
    assert callable(SpawnManager)
    assert callable(SubagentHandle)
    assert callable(SubagentSpawner)
    # Type aliases — ``Runner``, ``SpawnerFactory``, ``AnnounceCallback``
    # are callable on CPython (``Callable[...]`` evaluates to a generic
    # alias) and on MP they're whatever ``typing`` returns; we just
    # assert they imported.
    assert Runner is not None
    assert SpawnerFactory is not None
    assert AnnounceCallback is not None


def test_skill_entry_point_returns_dict():
    """``exoclaw_subagent.skills.spawn`` is the entry point that
    ``SkillsLoader`` (deployment-bundled path) consumes. It reads
    ``SKILL.md`` adjacent to the package, which on MP must use the
    ``Path`` shim from ``exoclaw._compat``."""
    from exoclaw_subagent.skills import spawn

    skill = spawn()
    assert isinstance(skill, dict)
    assert skill["name"] == "spawn"
    assert "content" in skill
    assert "path" in skill


def test_task_id_is_hex():
    """``SubagentManager.spawn`` uses ``os.urandom(4).hex()`` for
    task ids — ``uuid`` isn't on MP. Verify the shape directly: 8
    hex chars, lowercase."""
    import os

    out = os.urandom(4).hex()
    assert isinstance(out, str)
    assert len(out) == 8
    for c in out:
        assert c in "0123456789abcdef"


def test_in_memory_batch_store_register_and_announce():
    """Basic ``BatchStore`` round-trip — register a batch, mark
    members done, and verify the announcement callback fires
    exactly once when the batch completes. Pure in-memory protocol
    exercise, no I/O.

    This is the MP coverage path for the dataclass-style
    ``BatchSnapshot`` (``@dataclass`` with
    ``field(default_factory=list)``), the dict-of-sets bookkeeping
    in ``InMemoryBatchStore``, and ``asyncio.Lock`` — all of which
    need to clear on the MP side.
    """
    from exoclaw_subagent import InMemoryBatchStore

    announced = []

    async def _announce(snap):
        announced.append(snap)

    async def _exercise():
        store = InMemoryBatchStore()
        await store.register(
            "b1",
            "t1",
            session_key="s",
            origin_channel="serial",
            origin_chat_id="default",
        )
        await store.register(
            "b1",
            "t2",
            session_key="s",
            origin_channel="serial",
            origin_chat_id="default",
        )
        # First completion — batch not yet done, nothing announced.
        snap1 = await store.record_completion_and_maybe_announce(
            "b1",
            "t1",
            status="ok",
            label="task-1",
            result_path=None,
            announce=_announce,
        )
        assert snap1.completed == 1
        assert snap1.total == 2
        assert announced == []
        # Second completion — batch is done, callback fires once.
        snap2 = await store.record_completion_and_maybe_announce(
            "b1",
            "t2",
            status="ok",
            label="task-2",
            result_path=None,
            announce=_announce,
        )
        assert snap2.completed == 2
        assert len(announced) == 1
        finalsnap = announced[0]
        assert finalsnap.batch_id == "b1"
        assert len(finalsnap.results) == 2

    asyncio.run(_exercise())


def test_asyncio_spawner_runs_inline():
    """``AsyncioSpawner`` is the chip-side spawner that runs the
    child via plain ``asyncio.create_task`` — no DBOS, no journal.
    ``start()`` returns a handle that completes when the runner
    coroutine completes.
    """
    from exoclaw_subagent import AsyncioSpawner

    ran = []

    async def _runner(**kwargs):
        ran.append(kwargs["task_id"])

    async def _exercise():
        spawner = AsyncioSpawner(_runner)
        handle = await spawner.start(
            task_id="abc",
            task="hello",
            label="t",
            origin_channel="serial",
            origin_chat_id="default",
            session_key="serial:default",
            batch=None,
            skills=None,
            model=None,
        )
        await handle.wait()
        assert ran == ["abc"]
        assert handle.done()

    asyncio.run(_exercise())
