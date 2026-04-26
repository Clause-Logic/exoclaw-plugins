"""Memory vs tool-result size — Step D headline chart.

Companion to ``memory_loadtest.py`` (single-chat session length) and
``memory_loadtest_concurrent.py`` (N concurrent chats). This one
measures **peak Python heap during the wire serialization of one
tool result** as the result size grows.

Hypothesis to test:

* **inline tool** (``execute`` returns a ``str``): the full result
  is held in the message buffer, then serialized into the JSON
  request body. Peak heap during ``_stream_body`` grows linearly
  with result size.
* **streaming tool** (``execute_streaming`` yields chunks): the
  executor drains chunks into a per-turn scratch file and the tool
  message carries ``_content_file=<path>``. The provider streams
  the file from disk into the JSON ``content`` field 8192 chars at
  a time. Peak heap during ``_stream_body`` is bounded by the
  read-chunk size — flat regardless of tool-result size.

Procedure:

1. Register two tools side-by-side: ``inline_fat`` and
   ``streaming_fat``. Both accept a ``size_bytes`` argument.
2. For each ``size`` in ``--sizes`` and each tool variant:
     * ``executor.execute_tool_with_handle(...)`` → ``ToolResult``
     * Build the tool message (with ``_content_file`` for streaming).
     * Reset tracemalloc peak.
     * Drain ``_stream_body(head, [user, tool, asst])`` to bytes.
     * Sample tracemalloc peak — that's the wire-side peak.
     * Unlink scratch file (if any).
3. Plot ``size`` vs peak heap with two lines (inline vs streaming).

Run from the workspace root::

    uv run --with matplotlib --with psutil python \\
        packages/exoclaw-nanobot/scripts/memory_loadtest_tool_result.py \\
        --sizes 10000,100000,1000000,5000000,25000000 \\
        --out /tmp/memload_v3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import logging
import tempfile
import tracemalloc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import structlog
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.executor import DirectExecutor
from exoclaw_provider_openai.provider import _stream_body


def _silence_logs() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )


_CHUNK = 8192


class InlineFatTool:
    """Tool that returns a big string inline. The full result lives
    as one Python string from ``execute`` return through tool message
    through ``json.dumps`` in ``_stream_body``."""

    name = "inline_fat"
    description = "Returns size_bytes worth of x's, inline."
    parameters: dict[str, object] = {
        "type": "object",
        "properties": {"size_bytes": {"type": "integer"}},
        "required": ["size_bytes"],
    }

    async def execute(self, size_bytes: int, **kwargs: object) -> str:
        return "x" * size_bytes


class StreamingFatTool:
    """Tool that yields the same payload chunk-by-chunk via the
    Step D opt-in capability. The executor drains chunks into a
    per-turn scratch file."""

    name = "streaming_fat"
    description = "Returns size_bytes worth of x's, streamed."
    parameters: dict[str, object] = {
        "type": "object",
        "properties": {"size_bytes": {"type": "integer"}},
        "required": ["size_bytes"],
    }

    async def execute(self, size_bytes: int, **kwargs: object) -> str:
        # Should not be called when execute_streaming is present.
        raise AssertionError("inline execute should not run for streaming tool")

    async def execute_streaming(self, size_bytes: int, **kwargs: object) -> AsyncIterator[str]:
        chunk = "x" * _CHUNK
        full_chunks, remainder = divmod(size_bytes, _CHUNK)
        for _ in range(full_chunks):
            yield chunk
        if remainder:
            yield "x" * remainder


@dataclass
class Sample:
    mode: str  # "inline" or "streaming"
    size_bytes: int
    body_bytes: int
    tm_current: int
    tm_peak: int


async def _drain(gen: AsyncIterator[bytes]) -> int:
    """Consume a streaming body iterator and return total bytes — same
    shape as the httpx socket consumer would do, without keeping any
    chunk alive past its yield."""
    total = 0
    async for chunk in gen:
        total += len(chunk)
    return total


async def _measure(
    executor: DirectExecutor,
    registry: ToolRegistry,
    tool_name: str,
    size_bytes: int,
) -> Sample:
    outcome = await executor.execute_tool_with_handle(
        registry, tool_name, {"size_bytes": size_bytes}, tool_call_id="tc1"
    )

    tool_msg: dict[str, object] = {
        "role": "tool",
        "tool_call_id": "tc1",
        "name": tool_name,
        "content": outcome.content,
    }
    if outcome.content_file is not None:
        tool_msg["_content_file"] = str(outcome.content_file)

    head = {"model": "fake", "stream": True}
    messages = [
        {"role": "user", "content": "go"},
        tool_msg,
        {"role": "assistant", "content": "summary"},
    ]

    # Reset peak right before the wire serialization so the
    # measurement captures only this stage. The tool execution above
    # already wrote the scratch file (for streaming) — its allocations
    # are gone by now.
    gc.collect()
    tracemalloc.reset_peak()

    body_bytes = await _drain(_stream_body(head, messages))

    tm_current, tm_peak = tracemalloc.get_traced_memory()

    # Cleanup scratch immediately so it doesn't pollute the next
    # iteration's measurement. (Normally ``post_turn`` does this.)
    if outcome.content_file is not None:
        try:
            outcome.content_file.unlink()
        except OSError:
            pass

    mode = "streaming" if outcome.content_file is not None else "inline"
    return Sample(
        mode=mode,
        size_bytes=size_bytes,
        body_bytes=body_bytes,
        tm_current=tm_current,
        tm_peak=tm_peak,
    )


async def run(*, sizes: list[int]) -> list[Sample]:
    executor = DirectExecutor()
    registry = ToolRegistry()
    registry.register(InlineFatTool())
    registry.register(StreamingFatTool())

    samples: list[Sample] = []
    tracemalloc.start()

    for size in sizes:
        for tool_name in ("inline_fat", "streaming_fat"):
            sample = await _measure(executor, registry, tool_name, size)
            samples.append(sample)
            print(
                f"  {sample.mode:>9} size={size:>10}B  "
                f"tm_peak={sample.tm_peak / 1024:>10.1f}KiB  "
                f"body_bytes={sample.body_bytes:>10}"
            )

    tracemalloc.stop()
    return samples


def write_csv(samples: list[Sample], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "size_bytes", "body_bytes", "tm_current", "tm_peak"])
        for s in samples:
            w.writerow([s.mode, s.size_bytes, s.body_bytes, s.tm_current, s.tm_peak])


def write_png(samples: list[Sample], out: Path) -> None:
    import matplotlib.pyplot as plt

    def pick(mode: str) -> list[Sample]:
        return sorted([s for s in samples if s.mode == mode], key=lambda s: s.size_bytes)

    inline = pick("inline")
    streaming = pick("streaming")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        [s.size_bytes / 1024 for s in inline],
        [s.tm_peak / 1024 for s in inline],
        label="inline (today's path)",
        marker="o",
        linewidth=2,
    )
    ax.plot(
        [s.size_bytes / 1024 for s in streaming],
        [s.tm_peak / 1024 for s in streaming],
        label="streaming (Step D)",
        marker="s",
        linewidth=2,
    )

    ax.set_xlabel("tool result size (KiB)")
    ax.set_ylabel("peak Python heap during _stream_body (KiB)")
    ax.set_title(
        "Step D win: peak during wire serialization\n"
        "vs tool result size — inline vs streaming"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)


def _parse_sizes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=[10_000, 100_000, 1_000_000, 5_000_000, 25_000_000],
        help="Comma-separated tool result sizes in bytes.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.mkdtemp(prefix="memload_v3_")),
    )
    args = p.parse_args()

    _silence_logs()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"out={args.out}")
    print(f"sizes: {args.sizes}")

    samples = asyncio.run(run(sizes=args.sizes))

    csv_path = args.out / "memory.csv"
    png_path = args.out / "memory.png"
    write_csv(samples, csv_path)
    print(f"\nwrote {csv_path}")
    try:
        write_png(samples, png_path)
        print(f"wrote {png_path}")
    except ImportError:
        print("matplotlib not installed — skipping PNG. Re-run with `uv run --with matplotlib …`")


if __name__ == "__main__":
    main()
