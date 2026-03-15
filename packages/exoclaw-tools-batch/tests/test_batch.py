"""Tests for BatchTool."""

from __future__ import annotations

import json

import pytest

from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_tools_batch import BatchTool


class FakeTool:
    name = "echo"
    description = "Returns the input as-is"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

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
def batch() -> BatchTool:
    tool = BatchTool()
    reg = _make_registry(FakeTool(), SlowTool(), FailTool())
    reg.register(tool)  # type: ignore[arg-type]
    tool.set_registry(reg)
    return tool


@pytest.mark.asyncio
async def test_basic_batch(batch: BatchTool) -> None:
    result = await batch.execute(
        tool="echo",
        items=[{"text": "a"}, {"text": "b"}, {"text": "c"}],
    )
    data = json.loads(result)
    assert data["count"] == 3
    assert data["results"][0]["result"] == "echo:a"
    assert data["results"][1]["result"] == "echo:b"
    assert data["results"][2]["result"] == "echo:c"


@pytest.mark.asyncio
async def test_empty_items(batch: BatchTool) -> None:
    result = await batch.execute(tool="echo", items=[])
    data = json.loads(result)
    assert data["count"] == 0
    assert data["results"] == []


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
    result = await batch.execute(tool="fail", items=[{}, {}])
    data = json.loads(result)
    assert data["count"] == 2
    # Registry wraps errors, so result contains "Error"
    for r in data["results"]:
        assert "result" in r  # registry catches and returns error string
        assert "boom" in r["result"]


@pytest.mark.asyncio
async def test_preserves_order(batch: BatchTool) -> None:
    result = await batch.execute(
        tool="slow",
        items=[{"n": i} for i in range(20)],
        concurrency=5,
    )
    data = json.loads(result)
    assert data["count"] == 20
    for i, r in enumerate(data["results"]):
        assert r["result"] == f"done:{i}"


@pytest.mark.asyncio
async def test_concurrency_limit(batch: BatchTool) -> None:
    """Verify concurrency parameter is respected (doesn't crash)."""
    result = await batch.execute(
        tool="echo",
        items=[{"text": str(i)} for i in range(10)],
        concurrency=2,
    )
    data = json.loads(result)
    assert data["count"] == 10
