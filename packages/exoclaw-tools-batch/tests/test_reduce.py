"""Tests for ReduceTool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    """When extracted value is not a list, it gets appended as a single item."""
    f1 = _write_json(tmp_path / "a.json", {"results": "scalar_value"})
    f2 = _write_json(tmp_path / "b.json", {"results": {"nested": True}})

    result = await reduce.execute(files=[f1, f2])
    data = _read_output(result)
    assert data["results"] == ["scalar_value", {"nested": True}]
