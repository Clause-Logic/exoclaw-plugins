"""Reduce tool — merge multiple batch output files into one, with optional tree-reduce."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from exoclaw.agent.tools.protocol import ToolBase
from exoclaw.agent.tools.registry import ToolRegistry


class ReduceTool(ToolBase):
    """Merge multiple batch output files or JSON files into one.

    Reads JSON files from a list of paths or a directory, extracts values
    at a given key (default: "results"), and concatenates them into a
    single output file.

    Supports:
        - Explicit file list or directory glob
        - Configurable extract key
        - Dedup by field
        - chunk_size: split output into N-item chunks (for feeding next batch round)
        - until + then: tree-reduce loop — keeps chunking, calling a tool on each
          chunk, and merging until item count <= until

    Receives the registry via duck-typed set_registry() for tree-reduce
    (needs to call tools internally).
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = output_dir
        self._registry: ToolRegistry | None = None

    def set_registry(self, registry: ToolRegistry) -> None:
        """Called at registration time by AgentLoop."""
        self._registry = registry

    @property
    def name(self) -> str:
        return "reduce"

    @property
    def description(self) -> str:
        return (
            "Merge multiple JSON files into one. "
            "Provide a list of file paths or a directory. "
            "Extracts values at a key (default: 'results') from each file "
            "and concatenates them into a single output file. "
            "Use after batch() to combine results from multiple runs. "
            "Set chunk_size to split into multiple output files. "
            "Set until + then to tree-reduce: repeatedly chunk → call tool → merge "
            "until count <= until. The tool receives input_path in its params."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of JSON file paths to merge",
                },
                "dir": {
                    "type": "string",
                    "description": "Directory containing .json files to merge (alternative to files)",
                },
                "key": {
                    "type": "string",
                    "description": "JSON key to extract from each file (default: 'results'). "
                    "Use empty string to take root value.",
                },
                "dedup": {
                    "type": "string",
                    "description": "Field name to deduplicate by (e.g., 'url'). Optional.",
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Max items per output file. Creates multiple output files in a directory.",
                },
                "until": {
                    "type": "integer",
                    "description": "Tree-reduce: keep reducing until this many items remain. Requires then.",
                },
                "then": {
                    "type": "object",
                    "description": "Tool to call on each chunk during tree-reduce. "
                    "Properties: tool (tool name), params (dict — input_path is injected automatically).",
                    "properties": {
                        "tool": {"type": "string"},
                        "params": {"type": "object"},
                    },
                    "required": ["tool"],
                },
                "output": {
                    "type": "string",
                    "description": "Output file path. If omitted, a temp file is created.",
                },
            },
        }

    def _extract_items(self, paths: list[Path], key: str) -> tuple[list[Any], list[str]]:
        """Read files and extract items at key."""
        merged: list[Any] = []
        errors: list[str] = []
        for p in paths:
            if not p.exists():
                errors.append(f"not found: {p}")
                continue
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError) as e:
                errors.append(f"{p.name}: {e}")
                continue

            if key:
                value = data.get(key) if isinstance(data, dict) else data
            else:
                value = data

            if isinstance(value, list):
                merged.extend(value)
            else:
                merged.append(value)
        return merged, errors

    def _dedup(self, items: list[Any], field: str) -> list[Any]:
        """Deduplicate items by field."""
        seen: set[Any] = set()
        unique: list[Any] = []
        for item in items:
            k = item.get(field) if isinstance(item, dict) else None
            if k is not None and k in seen:
                continue
            if k is not None:
                seen.add(k)
            unique.append(item)
        return unique

    def _write_output(self, data: dict[str, Any], output: str | None) -> str:
        """Write result to file, return path."""
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(data, indent=2))
            return output

        out_dir = self._output_dir
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=".json", prefix="reduce_", dir=out_dir)
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _write_chunks(self, items: list[Any], chunk_size: int) -> tuple[str, list[str]]:
        """Split items into chunk_size files in a temp directory. Returns (dir_path, file_paths)."""
        chunk_dir = tempfile.mkdtemp(prefix="reduce_chunks_", dir=self._output_dir)
        paths: list[str] = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            path = Path(chunk_dir) / f"{i // chunk_size:04d}.json"
            path.write_text(json.dumps({"count": len(chunk), "results": chunk}, indent=2))
            paths.append(str(path))
        return chunk_dir, paths

    async def execute(
        self,
        files: list[str] | None = None,
        dir: str | None = None,
        key: str = "results",
        dedup: str | None = None,
        chunk_size: int | None = None,
        until: int | None = None,
        then: dict[str, Any] | None = None,
        output: str | None = None,
        **kwargs: Any,
    ) -> str:
        # Resolve file list
        paths: list[Path] = []
        if files:
            paths = [Path(f) for f in files]
        elif dir:
            d = Path(dir)
            if not d.is_dir():
                return f"Error: '{dir}' is not a directory"
            paths = sorted(d.glob("*.json"))
        else:
            return "Error: provide either 'files' (list of paths) or 'dir' (directory path)"

        if not paths:
            return json.dumps({"output_path": "", "count": 0})

        # Extract and merge
        merged, errors = self._extract_items(paths, key)

        # Dedup
        if dedup and merged:
            merged = self._dedup(merged, dedup)

        # Tree-reduce mode
        if until is not None and then is not None:
            return await self._tree_reduce(merged, until, then, chunk_size or 20, errors, output)

        # Chunk mode — split into multiple files
        if chunk_size and len(merged) > chunk_size:
            chunk_dir, chunk_paths = self._write_chunks(merged, chunk_size)
            return json.dumps(
                {
                    "output_dir": chunk_dir,
                    "files": chunk_paths,
                    "count": len(merged),
                    "chunks": len(chunk_paths),
                }
            )

        # Single output
        result: dict[str, Any] = {"count": len(merged), "results": merged}
        if errors:
            result["errors"] = errors
        out_path = self._write_output(result, output)
        return json.dumps({"output_path": out_path, "count": len(merged)})

    async def _tree_reduce(
        self,
        items: list[Any],
        until: int,
        then: dict[str, Any],
        chunk_size: int,
        errors: list[str],
        output: str | None,
    ) -> str:
        """Repeatedly chunk → call tool → merge until count <= until."""
        if not self._registry:
            return "Error: tree-reduce requires registry (set_registry not wired)."

        tool_name = then.get("tool", "")
        if not tool_name:
            return "Error: then.tool is required for tree-reduce."
        if not self._registry.has(tool_name):
            available = ", ".join(self._registry.tool_names)
            return f"Error: tool '{tool_name}' not found. Available: {available}"

        base_params = then.get("params", {})
        max_rounds = 20  # safety limit
        round_num = 0

        while len(items) > until and round_num < max_rounds:
            round_num += 1

            # Chunk
            chunks: list[list[Any]] = []
            for i in range(0, len(items), chunk_size):
                chunks.append(items[i : i + chunk_size])

            # Call tool on each chunk concurrently
            semaphore = asyncio.Semaphore(10)

            async def _process_chunk(chunk: list[Any]) -> str:
                async with semaphore:
                    # Write chunk to temp file
                    fd, chunk_path = tempfile.mkstemp(
                        suffix=".json", prefix=f"tree_r{round_num}_", dir=self._output_dir
                    )
                    with open(fd, "w") as f:
                        json.dump(chunk, f, indent=2)

                    # Build params with input_path injected
                    params = {**base_params, "input_path": chunk_path}
                    return await self._registry.execute(tool_name, params)

            tasks = [_process_chunk(chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)

            # Collect results as the new items for next round
            items = []
            for r in results:
                # Try to parse as JSON (structured output)
                try:
                    parsed = json.loads(r)
                    if isinstance(parsed, list):
                        items.extend(parsed)
                    else:
                        items.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    # Plain text result — wrap it
                    items.append(r)

        # Write final output
        result_data: dict[str, Any] = {
            "count": len(items),
            "results": items,
            "rounds": round_num,
        }
        if errors:
            result_data["errors"] = errors
        out_path = self._write_output(result_data, output)
        return json.dumps(
            {
                "output_path": out_path,
                "count": len(items),
                "rounds": round_num,
            }
        )
