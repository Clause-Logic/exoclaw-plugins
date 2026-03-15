"""Reduce tool — merge multiple batch output files into one."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from exoclaw.agent.tools.protocol import ToolBase


class ReduceTool(ToolBase):
    """Merge multiple batch output files or JSON files into one.

    Reads JSON files from a list of paths or a directory, extracts values
    at a given key (default: "results"), and concatenates them into a
    single output file.

    Supports:
        - Explicit file list
        - Directory glob (all .json files)
        - Configurable extract key (e.g., "results", "entries", or "" for root)
        - Optional dedup by a field
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = output_dir

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
            "Use after batch() to combine results from multiple runs."
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
                "output": {
                    "type": "string",
                    "description": "Output file path. If omitted, a temp file is created.",
                },
            },
        }

    async def execute(
        self,
        files: list[str] | None = None,
        dir: str | None = None,
        key: str = "results",
        dedup: str | None = None,
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

        # Extract and concatenate
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

            # Extract at key
            if key:
                value = data.get(key) if isinstance(data, dict) else data
            else:
                value = data

            if isinstance(value, list):
                merged.extend(value)
            else:
                merged.append(value)

        # Dedup
        if dedup and merged:
            seen: set[Any] = set()
            unique: list[Any] = []
            for item in merged:
                k = item.get(dedup) if isinstance(item, dict) else None
                if k is not None and k in seen:
                    continue
                if k is not None:
                    seen.add(k)
                unique.append(item)
            merged = unique

        # Write output
        result = {"count": len(merged), "results": merged}
        if errors:
            result["errors"] = errors

        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(result, indent=2))
            out_path = output
        else:
            out_dir = self._output_dir
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            fd, out_path = tempfile.mkstemp(suffix=".json", prefix="reduce_", dir=out_dir)
            with open(fd, "w") as f:
                json.dump(result, f, indent=2)

        return json.dumps({"output_path": out_path, "count": len(merged)})
