"""Batch tool — fan-out a tool across multiple inputs with controlled concurrency."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from exoclaw.agent.tools.protocol import ToolBase, ToolContext
from exoclaw.agent.tools.registry import ToolRegistry


class BatchTool(ToolBase):
    """Run a registered tool against a list of inputs concurrently.

    The LLM calls batch() with a tool name + list of param dicts.
    Each item is executed via the registry (no LLM involved).
    Results are coalesced into a single response.

    Receives the registry via duck-typed set_registry() called at
    registration time (same pattern as set_bus).
    """

    DEFAULT_CONCURRENCY = 10

    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY) -> None:
        self._registry: ToolRegistry | None = None
        self._concurrency = concurrency

    def set_registry(self, registry: ToolRegistry) -> None:
        """Called at registration time by AgentLoop."""
        self._registry = registry

    @property
    def name(self) -> str:
        return "batch"

    @property
    def description(self) -> str:
        return (
            "Run a tool against multiple inputs concurrently. "
            "Each item in the list is a dict of parameters for the tool. "
            "Results are returned as a JSON array in the same order. "
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
                    "description": f"Max parallel executions (default {DEFAULT_CONCURRENCY})",
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
        if not self._registry:
            return "Error: BatchTool has no registry — cannot execute tools."

        if not self._registry.has(tool):
            available = ", ".join(self._registry.tool_names)
            return f"Error: Tool '{tool}' not found. Available: {available}"

        if not items:
            return json.dumps({"results": [], "count": 0})

        max_concurrent = concurrency or self._concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_one(index: int, params: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                try:
                    result = await self._registry.execute(tool, params)
                    return {"index": index, "result": result}
                except Exception as e:
                    return {"index": index, "error": str(e)}

        tasks = [_run_one(i, item) for i, item in enumerate(items)]
        completed = await asyncio.gather(*tasks)

        # Restore original order
        ordered = sorted(completed, key=lambda r: r["index"])
        results = [
            {"result": r["result"]} if "result" in r else {"error": r["error"]}
            for r in ordered
        ]

        return json.dumps({"results": results, "count": len(results)})
