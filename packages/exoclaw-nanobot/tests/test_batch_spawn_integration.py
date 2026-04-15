"""Integration test for ``batch(tool="spawn", items=[…])``.

In production, the feed-curator skill fans out enrichment by asking the
``BatchTool`` to dispatch the ``SpawnTool`` across a list of items —
identical in shape to ``batch(tool="web_fetch", items=[…])``. Neither
package (``exoclaw-tools-batch`` nor ``exoclaw-tools-spawn``) tests the
combination in isolation, so the wiring that makes batch see spawn via
``set_registry`` has only ever been exercised by hand in production.

This test wires the real composition — one ``ToolRegistry`` shared by
both tools — and drives ``batch.execute(tool="spawn", items=[N])``.
It must actually dispatch N spawns. Regressions that either:

* fail to share the registry (``batch._registry.has("spawn")`` False), or
* short-circuit the items loop so only one spawn fires,

should both surface here instead of on a 172-item production run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_subagent import SpawnTool
from exoclaw_tools_batch import BatchTool


class _RecordingSpawner:
    """Minimal ``SpawnManager`` that records every spawn call.

    Avoids pulling ``SubagentManager`` + DBOS into this unit test; the
    point here is the registry wiring, not the subagent lifecycle.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        batch: str | None = None,
        skills: list[str] | None = None,
        model: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "task": task,
                "label": label,
                "batch": batch,
                "model": model,
            }
        )
        return f"Subagent [{label or task[:20]}] started (id: t{len(self.calls)})."

    def get_status(self) -> dict[str, Any]:
        return {"running": [], "completed": len(self.calls)}

    def list_results(self, limit: int = 20) -> list[dict[str, str]]:
        return []


def _build_registry(manager: _RecordingSpawner, tmp_path: Any) -> tuple[ToolRegistry, BatchTool]:
    """Wire BatchTool + SpawnTool into a single shared ToolRegistry.

    Mirrors what ``exoclaw/agent/loop.py`` does during ``AgentLoop.__init__``:
    register every tool, then call ``set_registry`` on any tool that wants
    one. A shared reference — not a snapshot — is required.
    """
    registry = ToolRegistry()

    spawn_tool = SpawnTool(manager=manager)
    registry.register(spawn_tool)

    batch_tool = BatchTool(output_dir=str(tmp_path))
    registry.register(batch_tool)  # type: ignore[arg-type]
    batch_tool.set_registry(registry)

    return registry, batch_tool


@pytest.mark.asyncio
async def test_batch_can_resolve_spawn_in_shared_registry(tmp_path: Any) -> None:
    """Sanity: batch's registry reference must contain spawn.

    This is the minimal invariant; the failing production error message
    was ``Tool 'spawn' not found`` even though spawn was registered — a
    symptom of batch holding a stale/different registry. If this
    assertion fires, the rest of the test is meaningless.
    """
    manager = _RecordingSpawner()
    registry, batch_tool = _build_registry(manager, tmp_path)

    assert registry.has("spawn"), "spawn missing from shared registry"
    assert batch_tool._registry is not None, "batch never received set_registry()"
    assert batch_tool._registry is registry, (
        "batch holds a different registry than the agent's — "
        "set_registry must share a reference, not snapshot"
    )
    assert batch_tool._registry.has("spawn"), (
        "batch's registry view does not contain spawn — dispatch will fail with 'Tool not found'"
    )


@pytest.mark.asyncio
async def test_batch_dispatches_spawn_for_every_item(tmp_path: Any) -> None:
    """``batch(tool="spawn", items=[N])`` must spawn N subagents.

    This is the feed-curator use case: one batch call, N parallel
    enrichment spawns. Regression fingerprint from production was
    "only 1 item processed" — the exact failure this asserts against.
    """
    manager = _RecordingSpawner()
    _, batch_tool = _build_registry(manager, tmp_path)

    items = [{"task": f"enrich item {i}", "label": f"enrich-{i}"} for i in range(5)]

    result = await batch_tool.execute(tool="spawn", items=items)

    meta = json.loads(result)
    assert meta["count"] == 5, f"batch reported count={meta.get('count')}: {result}"
    assert len(manager.calls) == 5, (
        f"spawn manager received {len(manager.calls)} calls, expected 5 — "
        "batch is short-circuiting the items loop"
    )

    # Items must be dispatched in order and each item's params must be
    # forwarded to spawn verbatim.
    for i, call in enumerate(manager.calls):
        assert call["task"] == f"enrich item {i}", call
        assert call["label"] == f"enrich-{i}", call


@pytest.mark.asyncio
async def test_batch_forwards_spawn_specific_params(tmp_path: Any) -> None:
    """Per-item ``batch`` grouping id and ``model`` overrides must survive
    the dispatch. Feed-curator relies on ``model=openrouter/openai/gpt-oss-120b``
    to route enrichment to the cheap model — if batch drops the param,
    every spawn falls back to the expensive default model.
    """
    manager = _RecordingSpawner()
    _, batch_tool = _build_registry(manager, tmp_path)

    items = [
        {
            "task": "t1",
            "label": "enrich-a",
            "batch": "feed-enrich",
            "model": "openrouter/openai/gpt-oss-120b",
        },
        {
            "task": "t2",
            "label": "enrich-b",
            "batch": "feed-enrich",
            "model": "openrouter/openai/gpt-oss-120b",
        },
    ]

    await batch_tool.execute(tool="spawn", items=items)

    assert len(manager.calls) == 2
    for call in manager.calls:
        assert call["batch"] == "feed-enrich", call
        assert call["model"] == "openrouter/openai/gpt-oss-120b", call


@pytest.mark.asyncio
async def test_batch_surfaces_spawn_failure_without_aborting(
    tmp_path: Any,
) -> None:
    """One spawn failing must not cancel the rest of the fan-out.

    Since exoclaw 0.14, ``ToolRegistry.execute`` propagates tool
    exceptions to the caller (it used to swallow them into an
    ``"Error executing X"`` string). ``BatchTool``'s per-item
    ``try/except`` catches them and writes ``{"error": str(e)}``
    into the per-item dict — so the failed item shows up in the
    ``error`` slot, not the ``result`` slot. The remaining items
    must still dispatch.
    """
    manager = _RecordingSpawner()
    call_counter = {"n": 0}

    async def flaky_spawn(**kwargs: Any) -> str:
        call_counter["n"] += 1
        if call_counter["n"] == 2:
            raise RuntimeError("simulated spawn failure")
        return f"Subagent started (id: t{call_counter['n']})."

    manager.spawn = flaky_spawn  # type: ignore[method-assign]

    _, batch_tool = _build_registry(manager, tmp_path)

    items = [{"task": f"t{i}", "label": f"l{i}"} for i in range(5)]
    result = await batch_tool.execute(tool="spawn", items=items)

    meta = json.loads(result)
    assert meta["count"] == 5, "batch must report a result entry for every item"

    with open(meta["output_path"]) as f:
        payload = json.load(f)
    results = payload["results"]
    assert len(results) == 5, f"payload={payload}"

    # Per-item exceptions land in an ``error`` slot, not the ``result``
    # slot. exoclaw 0.14 stopped having ``ToolRegistry.execute`` swallow
    # tool exceptions (they propagate to the agent loop for proper
    # logging), so BatchTool's per-item ``try/except`` is the only thing
    # that catches them — and it writes ``{"error": str(e)}``.
    failure_lines = [r for r in results if "error" in r and "simulated spawn failure" in r["error"]]
    success_lines = [r for r in results if "result" in r]
    assert len(failure_lines) == 1, (
        f"expected exactly one failed spawn surfaced as an error entry, got payload={payload}"
    )
    assert len(success_lines) == 4, f"remaining items must still dispatch — got {success_lines}"
