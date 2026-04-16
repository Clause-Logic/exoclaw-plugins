"""Batch tool — fan-out a tool across multiple inputs with controlled concurrency."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from exoclaw.agent.tools.protocol import ToolBase
from exoclaw.agent.tools.registry import ToolRegistry


class BatchTool(ToolBase):
    """Run a registered tool against a list of inputs concurrently.

    The LLM calls batch() with a tool name + list of param dicts.
    Each item is executed via the registry (no LLM involved).
    Results are written to a temp file and the path is returned,
    keeping the agent's context window clean.

    **Registry lookup is ContextVar-based as of exoclaw 0.15.2.**
    On each ``execute`` call we read ``ToolRegistry.current()`` — the
    *dispatching* registry that the outer ``registry.execute("batch",
    …)`` bound for the duration of the tool body. ``set_registry``
    stores a fallback reference for callers that drive the tool
    directly (tests, custom wiring) without going through
    ``ToolRegistry.execute``.

    The stored-reference pattern alone was last-write-wins across
    multiple ``AgentLoop`` instances sharing a ``BatchTool`` — each
    loop's ctor would clobber the pointer, so the main agent's
    ``BatchTool`` would end up pointing at the last-constructed
    subagent's (spawn-less) registry. The ContextVar is
    per-asyncio-task so concurrent dispatches don't interfere.
    """

    DEFAULT_CONCURRENCY = 10

    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        output_dir: str | None = None,
    ) -> None:
        self._registry: ToolRegistry | None = None
        self._concurrency = concurrency
        self._output_dir = output_dir

    def set_registry(self, registry: ToolRegistry) -> None:
        """Stored as a fallback for non-``execute``-driven callers.

        The preferred path is ``ToolRegistry.current()`` from inside
        ``execute`` — it matches the dispatching registry on a
        per-asyncio-task basis and avoids the shared-instance race
        that broke production on 2026-04-15.
        """
        self._registry = registry

    def _resolve_registry(self) -> ToolRegistry | None:
        """Return the registry batch should dispatch against.

        Prefer ``ToolRegistry.current()`` (set by the outer
        ``registry.execute("batch", …)`` frame) over the stored
        fallback. Direct tool-body invocations that bypass the
        registry (e.g. unit tests that call ``batch_tool.execute``)
        still see the stored reference.
        """
        return ToolRegistry.current() or self._registry

    @property
    def name(self) -> str:
        return "batch"

    @property
    def description(self) -> str:
        return (
            "Run a tool against multiple inputs concurrently. "
            "Each item in the list is a dict of parameters for the tool. "
            "Results are written to a temp file and the path is returned. "
            "Use read_file to inspect results. "
            "Use this when you need to call the same tool many times "
            "(e.g., fetch 20 URLs, read 10 files). "
            "No LLM calls are made — tools run directly."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Name of the tool to call for each item",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of parameter dicts, one per invocation",
                },
                "concurrency": {
                    "type": "integer",
                    "description": f"Max parallel executions (default {self.DEFAULT_CONCURRENCY})",
                },
            },
            "required": ["tool", "items"],
        }

    async def execute(
        self,
        tool: str,
        items: list[dict[str, Any]],
        concurrency: int | None = None,
        **kwargs: Any,
    ) -> str:
        registry = self._resolve_registry()
        if registry is None:
            return "Error: BatchTool has no registry — cannot execute tools."

        if not registry.has(tool):
            available = ", ".join(registry.tool_names)
            return f"Error: Tool '{tool}' not found. Available: {available}"

        if not items:
            return json.dumps({"output_path": "", "count": 0})

        max_concurrent = concurrency or self._concurrency
        if max_concurrent < 1:
            return "Error: concurrency must be >= 1"
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_one(index: int, params: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                try:
                    result = await registry.execute(tool, params)
                    return {"index": index, "result": result}
                except Exception as e:
                    return {"index": index, "error": str(e)}

        tasks = [_run_one(i, item) for i, item in enumerate(items)]
        completed = await asyncio.gather(*tasks)

        # Restore original order
        ordered = sorted(completed, key=lambda r: r["index"])
        results = [
            {"result": r["result"]} if "result" in r else {"error": r["error"]} for r in ordered
        ]

        # Aggregate usage from results that contain it (e.g. llm_call)
        total_usage: dict[str, int] = {}
        for r in results:
            if "result" not in r:
                continue
            try:
                parsed = json.loads(r["result"])
                if isinstance(parsed, dict) and "usage" in parsed:
                    for k, v in parsed["usage"].items():
                        if isinstance(v, int):
                            total_usage[k] = total_usage.get(k, 0) + v
            except (json.JSONDecodeError, TypeError):
                pass

        # Write to temp file
        output: dict[str, Any] = {"tool": tool, "count": len(results), "results": results}
        if total_usage:
            output["usage"] = total_usage
        output_dir = self._output_dir
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=".json", prefix=f"batch_{tool}_", dir=output_dir)
        with open(fd, "w") as f:
            json.dump(output, f, indent=2)

        meta: dict[str, Any] = {"output_path": path, "count": len(results)}
        if total_usage:
            meta["usage"] = total_usage
        return json.dumps(meta)
