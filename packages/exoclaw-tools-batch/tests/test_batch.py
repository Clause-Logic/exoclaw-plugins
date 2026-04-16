"""Tests for BatchTool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_tools_batch import BatchTool


class FakeTool:
    name = "echo"
    description = "Returns the input as-is"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str = "", **kwargs: object) -> str:
        return f"echo:{text}"


class SlowTool:
    name = "slow"
    description = "Simulates slow work"
    parameters = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

    async def execute(self, n: int = 0, **kwargs: object) -> str:
        import asyncio

        await asyncio.sleep(0.01)
        return f"done:{n}"


class FailTool:
    name = "fail"
    description = "Always fails"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: object) -> str:
        raise RuntimeError("boom")


def _make_registry(*tools: object) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)  # type: ignore[arg-type]
    return reg


@pytest.fixture
def batch(tmp_path: object) -> BatchTool:
    tool = BatchTool(output_dir=str(tmp_path))
    reg = _make_registry(FakeTool(), SlowTool(), FailTool())
    reg.register(tool)  # type: ignore[arg-type]
    tool.set_registry(reg)
    return tool


def _read_output(result_json: str) -> dict:
    """Parse batch result, read the output file, return its contents."""
    meta = json.loads(result_json)
    path = meta["output_path"]
    with open(path) as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_basic_batch(batch: BatchTool) -> None:
    result = await batch.execute(
        tool="echo",
        items=[{"text": "a"}, {"text": "b"}, {"text": "c"}],
    )
    meta = json.loads(result)
    assert meta["count"] == 3
    assert meta["output_path"].endswith(".json")

    data = _read_output(result)
    assert data["tool"] == "echo"
    assert data["count"] == 3
    assert data["results"][0]["result"] == "echo:a"
    assert data["results"][1]["result"] == "echo:b"
    assert data["results"][2]["result"] == "echo:c"


@pytest.mark.asyncio
async def test_empty_items(batch: BatchTool) -> None:
    result = await batch.execute(tool="echo", items=[])
    meta = json.loads(result)
    assert meta["count"] == 0
    assert meta["output_path"] == ""


@pytest.mark.asyncio
async def test_unknown_tool(batch: BatchTool) -> None:
    result = await batch.execute(tool="nonexistent", items=[{"x": 1}])
    assert "Error" in result
    assert "nonexistent" in result


@pytest.mark.asyncio
async def test_no_registry() -> None:
    tool = BatchTool()
    result = await tool.execute(tool="echo", items=[{"text": "a"}])
    assert "Error" in result
    assert "registry" in result.lower()


@pytest.mark.asyncio
async def test_error_handling(batch: BatchTool) -> None:
    # Per-item exceptions land in an ``error`` slot, not the ``result``
    # slot. exoclaw 0.14 stopped catching tool exceptions inside
    # ``ToolRegistry.execute`` (they now propagate to the agent loop
    # for proper logging), so BatchTool's per-item ``try/except`` is
    # the only thing that catches them — and it writes
    # ``{"error": str(e)}``. Before exoclaw 0.14 this test passed
    # because registry.execute stringified exceptions into the
    # ``result`` slot directly.
    result = await batch.execute(tool="fail", items=[{}, {}])
    data = _read_output(result)
    assert data["count"] == 2
    for r in data["results"]:
        assert "error" in r
        assert "boom" in r["error"]


@pytest.mark.asyncio
async def test_preserves_order(batch: BatchTool) -> None:
    result = await batch.execute(
        tool="slow",
        items=[{"n": i} for i in range(20)],
        concurrency=5,
    )
    data = _read_output(result)
    assert data["count"] == 20
    for i, r in enumerate(data["results"]):
        assert r["result"] == f"done:{i}"


@pytest.mark.asyncio
async def test_concurrency_limit(batch: BatchTool) -> None:
    result = await batch.execute(
        tool="echo",
        items=[{"text": str(i)} for i in range(10)],
        concurrency=2,
    )
    data = _read_output(result)
    assert data["count"] == 10


@pytest.mark.asyncio
async def test_output_file_cleanup(batch: BatchTool) -> None:
    """Output files exist and are valid JSON."""
    result = await batch.execute(tool="echo", items=[{"text": "x"}])
    meta = json.loads(result)
    path = meta["output_path"]
    assert Path(path).exists()
    with open(path) as f:
        data = json.load(f)
    assert data["results"][0]["result"] == "echo:x"


@pytest.mark.asyncio
async def test_batch_threads_toolcontext_to_per_item_dispatch(tmp_path: Path) -> None:
    """When invoked via ``ToolRegistry.execute(..., ctx=...)``, BatchTool
    must forward the *same* ToolContext into every per-item dispatch.

    Fixes the production bug from 2026-04-16 where
    ``batch(tool="spawn", items=[13])`` dispatched ``SpawnTool`` without
    a context, so spawn fell through to ``execute`` (not
    ``execute_with_context``) and read the stored origin channel/chat_id
    — which was a stale ``cli:direct`` left over from a prior cron fire.
    Every one of the 13 resulting subagent completion announcements
    went to ``cli:direct`` instead of the zulip session the user was
    actually in, so the user never saw them.

    The fix routes ctx → ``BatchTool.execute_with_context`` →
    ``registry.execute(tool, params, ctx)`` for every item, so
    context-aware tools (``SpawnTool``, anything with
    ``execute_with_context``) see the caller's real turn context.
    """
    from exoclaw.agent.tools.protocol import ToolContext

    seen_ctx: list[ToolContext | None] = []

    class _RecordingTool:
        name = "record_ctx"
        description = "records the ctx it was given"
        parameters = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs: object) -> str:
            seen_ctx.append(None)  # fell through to context-less path
            return "no-ctx"

        async def execute_with_context(self, ctx: ToolContext, **kwargs: object) -> str:
            seen_ctx.append(ctx)
            return f"ctx:{ctx.channel}:{ctx.chat_id}"

    registry = ToolRegistry()
    registry.register(_RecordingTool())
    batch_tool = BatchTool(output_dir=str(tmp_path))
    registry.register(batch_tool)  # type: ignore[arg-type]

    ctx = ToolContext(channel="zulip", chat_id="583983:feeds", session_key="zulip:583983:feeds")
    items = [{}, {}, {}]
    result = await registry.execute("batch", {"tool": "record_ctx", "items": items}, ctx)

    meta = json.loads(result)
    assert meta["count"] == 3
    assert len(seen_ctx) == 3, f"expected 3 per-item dispatches, got {len(seen_ctx)}"
    for observed in seen_ctx:
        assert observed is not None, (
            "per-item dispatch landed in execute() instead of "
            "execute_with_context() — ctx was not threaded"
        )
        assert observed.channel == "zulip"
        assert observed.chat_id == "583983:feeds"
        assert observed.session_key == "zulip:583983:feeds"


@pytest.mark.asyncio
async def test_batch_execute_without_ctx_still_works(tmp_path: Path) -> None:
    """``BatchTool.execute`` (no ctx) keeps working for tests and
    callers that drive the tool directly without going through
    ``ToolRegistry.execute``. Per-item dispatches use ``ctx=None``.
    """
    batch_tool = BatchTool(output_dir=str(tmp_path))
    registry = ToolRegistry()

    class _EchoTool:
        name = "echo"
        description = "echo"
        parameters = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs: object) -> str:
            return "ok"

    registry.register(_EchoTool())
    registry.register(batch_tool)  # type: ignore[arg-type]
    batch_tool.set_registry(registry)

    result = await batch_tool.execute(tool="echo", items=[{}, {}])
    meta = json.loads(result)
    assert meta["count"] == 2


@pytest.mark.asyncio
async def test_batch_uses_dispatching_registry_not_stored_ref(tmp_path: Path) -> None:
    """When invoked via ``ToolRegistry.execute``, BatchTool dispatches
    against the registry doing the dispatching — not whichever registry
    happened to be stored last via ``set_registry``.

    Pins the production bug from 2026-04-15 where a single ``BatchTool``
    instance was shared between the main agent loop and every subagent
    loop. Each subagent's ``AgentLoop.__init__`` called
    ``batch_tool.set_registry(subagent_registry)``, clobbering the main
    agent's reference. Minutes later the main agent's next batch call
    dispatched against a stale subagent registry that didn't include
    ``spawn``. The ContextVar-based lookup added in exoclaw 0.15.2
    fixes the race: ``ToolRegistry.execute`` binds the current registry
    for the duration of the tool body and BatchTool reads from
    ``ToolRegistry.current()``.
    """
    batch_tool = BatchTool(output_dir=str(tmp_path))

    main_reg = ToolRegistry()
    sub_reg = ToolRegistry()

    # Only register echo in main. The subagent registry has a different
    # tool set — call it 'echo' too but with a distinct return value so
    # we can tell which registry handled the dispatch.
    class _MainEcho:
        name = "echo"
        description = "main"
        parameters = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs: object) -> str:
            return "from-main"

    class _SubEcho:
        name = "echo"
        description = "sub"
        parameters = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs: object) -> str:
            return "from-sub"

    main_reg.register(_MainEcho())
    sub_reg.register(_SubEcho())
    main_reg.register(batch_tool)  # type: ignore[arg-type]
    sub_reg.register(batch_tool)  # type: ignore[arg-type]

    # Simulate the production race: the subagent's AgentLoop.__init__
    # calls set_registry LAST, which used to clobber the stored ref.
    batch_tool.set_registry(main_reg)
    batch_tool.set_registry(sub_reg)  # stale — points at sub

    # Dispatch via the MAIN registry. With the old stored-ref
    # behavior this would dispatch against sub_reg (because
    # set_registry was last called with sub_reg) and return
    # 'from-sub'. With the ContextVar fix, it must see main_reg via
    # ``ToolRegistry.current()`` and return 'from-main'.
    result = await main_reg.execute("batch", {"tool": "echo", "items": [{}]})
    meta = json.loads(result)
    with open(meta["output_path"]) as f:
        payload = json.load(f)
    assert payload["results"][0]["result"] == "from-main", (
        f"batch dispatched against the stale stored registry instead of "
        f"the dispatching one — payload={payload}"
    )

    # Now dispatch via the SUB registry. Same batch_tool instance, but
    # the dispatch context swaps. Must see sub_reg and return 'from-sub'.
    result = await sub_reg.execute("batch", {"tool": "echo", "items": [{}]})
    meta = json.loads(result)
    with open(meta["output_path"]) as f:
        payload = json.load(f)
    assert payload["results"][0]["result"] == "from-sub", (
        f"batch failed to switch registries per dispatch — payload={payload}"
    )

    # And one more from main to prove it's not sticky after sub.
    result = await main_reg.execute("batch", {"tool": "echo", "items": [{}]})
    meta = json.loads(result)
    with open(meta["output_path"]) as f:
        payload = json.load(f)
    assert payload["results"][0]["result"] == "from-main"
