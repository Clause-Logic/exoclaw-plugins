"""Tests for ReduceTool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_tools_batch import ReduceTool


@pytest.fixture
def reduce(tmp_path: Path) -> ReduceTool:
    return ReduceTool(output_dir=str(tmp_path))


def _write_json(path: Path, data: object) -> str:
    path.write_text(json.dumps(data))
    return str(path)


def _read_output(result_json: str) -> dict:
    meta = json.loads(result_json)
    with open(meta["output_path"]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Basic merge tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_files(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"results": [{"url": "a"}, {"url": "b"}]})
    f2 = _write_json(tmp_path / "b.json", {"results": [{"url": "c"}]})

    result = await reduce.execute(files=[f1, f2])
    meta = json.loads(result)
    assert meta["count"] == 3

    data = _read_output(result)
    urls = [r["url"] for r in data["results"]]
    assert urls == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_merge_from_dir(reduce: ReduceTool, tmp_path: Path) -> None:
    d = tmp_path / "inputs"
    d.mkdir()
    _write_json(d / "01.json", {"results": [1, 2]})
    _write_json(d / "02.json", {"results": [3, 4]})

    result = await reduce.execute(dir=str(d))
    data = _read_output(result)
    assert data["results"] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_custom_key(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"entries": ["x", "y"]})
    f2 = _write_json(tmp_path / "b.json", {"entries": ["z"]})

    result = await reduce.execute(files=[f1, f2], key="entries")
    data = _read_output(result)
    assert data["results"] == ["x", "y", "z"]


@pytest.mark.asyncio
async def test_empty_key_takes_root(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", [1, 2])
    f2 = _write_json(tmp_path / "b.json", [3])

    result = await reduce.execute(files=[f1, f2], key="")
    data = _read_output(result)
    assert data["results"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_dedup(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"results": [{"url": "a"}, {"url": "b"}]})
    f2 = _write_json(tmp_path / "b.json", {"results": [{"url": "b"}, {"url": "c"}]})

    result = await reduce.execute(files=[f1, f2], dedup="url")
    data = _read_output(result)
    urls = [r["url"] for r in data["results"]]
    assert urls == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_explicit_output(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"results": [1]})
    out = str(tmp_path / "merged.json")

    result = await reduce.execute(files=[f1], output=out)
    meta = json.loads(result)
    assert meta["output_path"] == out
    assert Path(out).exists()


@pytest.mark.asyncio
async def test_missing_file(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"results": [1]})

    result = await reduce.execute(files=[f1, "/nonexistent/file.json"])
    data = _read_output(result)
    assert data["count"] == 1
    assert len(data["errors"]) == 1


@pytest.mark.asyncio
async def test_no_inputs(reduce: ReduceTool) -> None:
    result = await reduce.execute()
    assert "Error" in result


@pytest.mark.asyncio
async def test_empty_dir(reduce: ReduceTool, tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    result = await reduce.execute(dir=str(d))
    meta = json.loads(result)
    assert meta["count"] == 0


@pytest.mark.asyncio
async def test_non_list_values_appended(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"results": "scalar_value"})
    f2 = _write_json(tmp_path / "b.json", {"results": {"nested": True}})

    result = await reduce.execute(files=[f1, f2])
    data = _read_output(result)
    assert data["results"] == ["scalar_value", {"nested": True}]


# ---------------------------------------------------------------------------
# chunk_size tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_size(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"results": list(range(10))})

    result = await reduce.execute(files=[f1], chunk_size=3)
    meta = json.loads(result)
    assert meta["count"] == 10
    assert meta["chunks"] == 4  # ceil(10/3)
    assert len(meta["files"]) == 4

    # Verify chunk contents
    for path in meta["files"]:
        data = json.loads(Path(path).read_text())
        assert len(data["results"]) <= 3


@pytest.mark.asyncio
async def test_chunk_size_not_triggered_when_small(reduce: ReduceTool, tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"results": [1, 2]})

    result = await reduce.execute(files=[f1], chunk_size=10)
    meta = json.loads(result)
    # Small enough — single output, no chunking
    assert "output_path" in meta
    assert "chunks" not in meta


# ---------------------------------------------------------------------------
# tree-reduce tests
# ---------------------------------------------------------------------------


class FakeCondenser:
    """Fake tool that reads input_path and returns a condensed JSON list."""

    name = "condenser"
    description = "fake"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, input_path: str = "", **kwargs: object) -> str:
        if input_path and Path(input_path).exists():
            data = json.loads(Path(input_path).read_text())
            if isinstance(data, list) and len(data) > 0:
                return json.dumps([f"summary_of_{len(data)}_items"])
            return json.dumps(["empty"])
        return json.dumps(["no_input"])


def _make_tree_reduce(tmp_path: Path) -> ReduceTool:
    tool = ReduceTool(output_dir=str(tmp_path))
    reg = ToolRegistry()
    reg.register(FakeCondenser())  # type: ignore[arg-type]
    reg.register(tool)  # type: ignore[arg-type]
    tool.set_registry(reg)
    return tool


@pytest.mark.asyncio
async def test_tree_reduce_converges(tmp_path: Path) -> None:
    reduce = _make_tree_reduce(tmp_path)

    # 100 items, chunk_size=20 → 5 chunks → 5 summaries → 1 round to get <=5
    f1 = _write_json(tmp_path / "big.json", {"results": list(range(100))})

    result = await reduce.execute(
        files=[f1],
        until=5,
        chunk_size=20,
        then={"tool": "condenser"},
    )
    meta = json.loads(result)
    assert meta["count"] <= 5
    assert meta["rounds"] >= 1


@pytest.mark.asyncio
async def test_tree_reduce_single_round(tmp_path: Path) -> None:
    reduce = _make_tree_reduce(tmp_path)

    # 10 items, chunk_size=5 → 2 chunks → 2 summaries → done (<=5)
    f1 = _write_json(tmp_path / "small.json", {"results": list(range(10))})

    result = await reduce.execute(
        files=[f1],
        until=5,
        chunk_size=5,
        then={"tool": "condenser"},
    )
    meta = json.loads(result)
    assert meta["count"] <= 5
    assert meta["rounds"] == 1


@pytest.mark.asyncio
async def test_tree_reduce_already_below_until(tmp_path: Path) -> None:
    reduce = _make_tree_reduce(tmp_path)

    f1 = _write_json(tmp_path / "tiny.json", {"results": [1, 2]})

    result = await reduce.execute(
        files=[f1],
        until=5,
        then={"tool": "condenser"},
    )
    meta = json.loads(result)
    assert meta["count"] == 2
    assert meta["rounds"] == 0  # No reduction needed


@pytest.mark.asyncio
async def test_tree_reduce_with_extra_params(tmp_path: Path) -> None:
    """Extra params in then.params are passed through to the tool."""
    reduce = _make_tree_reduce(tmp_path)

    f1 = _write_json(tmp_path / "a.json", {"results": list(range(10))})

    result = await reduce.execute(
        files=[f1],
        until=5,
        chunk_size=5,
        then={"tool": "condenser", "params": {"extra_key": "extra_value"}},
    )
    meta = json.loads(result)
    assert meta["count"] <= 5


@pytest.mark.asyncio
async def test_tree_reduce_no_registry(tmp_path: Path) -> None:
    reduce = ReduceTool(output_dir=str(tmp_path))
    f1 = _write_json(tmp_path / "a.json", {"results": list(range(50))})

    result = await reduce.execute(files=[f1], until=1, then={"tool": "condenser"})
    assert "Error" in result
    assert "registry" in result.lower()


@pytest.mark.asyncio
async def test_tree_reduce_tool_not_found(tmp_path: Path) -> None:
    reduce = ReduceTool(output_dir=str(tmp_path))
    reg = ToolRegistry()
    reg.register(reduce)  # type: ignore[arg-type]
    reduce.set_registry(reg)

    f1 = _write_json(tmp_path / "a.json", {"results": list(range(50))})

    result = await reduce.execute(files=[f1], until=1, then={"tool": "nonexistent"})
    assert "Error" in result
    assert "nonexistent" in result


@pytest.mark.asyncio
async def test_tree_reduce_missing_tool_name(tmp_path: Path) -> None:
    reduce = _make_tree_reduce(tmp_path)
    f1 = _write_json(tmp_path / "a.json", {"results": list(range(10))})

    result = await reduce.execute(files=[f1], until=1, then={"params": {}})
    assert "Error" in result
    assert "then.tool" in result
