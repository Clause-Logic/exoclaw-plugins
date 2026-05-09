"""Drive nanobot's chat path with a fake LLM and chart memory vs session length.

Single session, N sequential turns. The fake provider returns a canned
LLMResponse (no tool calls), so each turn appends two messages
(user + assistant) to the session JSONL. With phase 2b's lazy
PriorSource active (exoclaw 0.20.1+), Python-heap usage between turns
should stay roughly flat as the JSONL grows; without it, it climbs
linearly with session length.

Run from the workspace root:

    uv run --with matplotlib --with psutil \
        python packages/exoclaw-nanobot/scripts/memory_loadtest.py \
        --turns 200 --out /tmp/memload

Outputs ``memory.csv`` and ``memory.png`` in --out.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import logging
import tempfile
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import structlog
from exoclaw.agent.loop import AgentLoop
from exoclaw.bus.queue import MessageBus
from exoclaw.executor import DirectExecutor
from exoclaw.providers.types import LLMResponse
from exoclaw_conversation.context import ContextBuilder
from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_conversation.memory import MemoryStore
from exoclaw_conversation.session.manager import SessionManager


def _silence_logs() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )


class FakeProvider:
    """LLMProvider stub. Returns a canned text response, no tool calls."""

    def __init__(self, default_model: str, reply_bytes: int) -> None:
        self._model = default_model
        self._reply = "x" * reply_bytes

    def get_default_model(self) -> str:
        return self._model

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        return LLMResponse(content=self._reply, finish_reason="stop")


@dataclass
class Sample:
    turn: int
    session_messages: int
    jsonl_bytes: int
    tm_current: int
    tm_peak: int
    rss: int


def _rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return 0


async def run(
    *,
    turns: int,
    user_bytes: int,
    reply_bytes: int,
    workspace: Path,
    sample_every: int,
    streaming: bool,
) -> list[Sample]:
    bus = MessageBus()
    provider = FakeProvider(default_model="fake-model", reply_bytes=reply_bytes)

    # Mirror nanobot.app's wiring so the loadtest exercises the same
    # SessionManager + MemoryStore path that ships in production. The
    # ``streaming`` flag flips on memory-model.md Step C — under it the
    # unconsolidated tail lives only on disk and ``session.messages``
    # stays empty across turns, capping per-session RAM.
    history_store = SessionManager(workspace, streaming_history=streaming)
    memory_store = MemoryStore(workspace, provider, "fake-model")
    # memory_window very large so SummarizingConsolidationPolicy never fires —
    # otherwise the curve gets stairstepped by background consolidation tasks
    # mutating the session out from under the measurement loop.
    conversation = DefaultConversation(
        history=history_store,
        memory=memory_store,
        prompt=ContextBuilder(workspace, memory=memory_store),
        memory_window=10**9,
    )
    executor = DirectExecutor()
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        conversation=conversation,
        executor=executor,
        model="fake-model",
        max_iterations=2,
    )

    user_msg = "u" * user_bytes
    session_key = "loadtest:single"
    samples: list[Sample] = []

    tracemalloc.start()
    session_path = workspace / "sessions" / "loadtest_single.jsonl"

    for turn in range(1, turns + 1):
        tracemalloc.reset_peak()
        await loop.process_direct(
            user_msg,
            session_key=session_key,
            channel="cli",
            chat_id="single",
        )
        if streaming:
            # DirectExecutor.build_prompt today snapshots the full prompt
            # list into a closure (set_messages) — that closure stays
            # bound to _prior_var between turns, so the per-turn history
            # is heap-resident even though session.messages is empty.
            # DBOSExecutor (production) auto-detects load_persisted_history
            # and installs a lazy source instead. Mirror that here so the
            # loadtest measures the same behaviour the deployed bot gets.
            executor.set_prior_source(
                lambda key=session_key: conversation.load_persisted_history(key)
            )
        if turn % sample_every == 0 or turn == turns:
            gc.collect()
            tm_current, tm_peak = tracemalloc.get_traced_memory()
            jsonl_bytes = session_path.stat().st_size if session_path.exists() else 0
            session = conversation.history.get_or_create(session_key)
            samples.append(
                Sample(
                    turn=turn,
                    session_messages=session.total_messages,
                    jsonl_bytes=jsonl_bytes,
                    tm_current=tm_current,
                    tm_peak=tm_peak,
                    rss=_rss_bytes(),
                )
            )
            print(
                f"turn={turn:>4}  msgs={session.total_messages:>5}  "
                f"jsonl={jsonl_bytes / 1024:>7.1f}KiB  "
                f"tm_current={tm_current / 1024:>7.1f}KiB  "
                f"tm_peak={tm_peak / 1024:>7.1f}KiB  "
                f"rss={samples[-1].rss / 1024 / 1024:>6.1f}MiB"
            )
    tracemalloc.stop()
    return samples


def write_csv(samples: list[Sample], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn", "session_messages", "jsonl_bytes", "tm_current", "tm_peak", "rss"])
        for s in samples:
            w.writerow([s.turn, s.session_messages, s.jsonl_bytes, s.tm_current, s.tm_peak, s.rss])


def write_png(samples: list[Sample], out: Path) -> None:
    import matplotlib.pyplot as plt

    turns = [s.turn for s in samples]
    tm_cur_kib = [s.tm_current / 1024 for s in samples]
    tm_peak_kib = [s.tm_peak / 1024 for s in samples]
    rss_mib = [s.rss / 1024 / 1024 for s in samples]
    jsonl_kib = [s.jsonl_bytes / 1024 for s in samples]
    has_rss = any(r > 0 for r in rss_mib)

    n_rows = 3 if has_rss else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True)
    axes = list(axes) if n_rows > 1 else [axes]

    ax = axes[0]
    ax.plot(turns, tm_cur_kib, label="current (post-turn)", marker="o", markersize=3)
    ax.plot(turns, tm_peak_kib, label="per-turn peak", marker="s", markersize=3)
    ax.set_ylabel("Python heap (KiB)")
    ax.set_title("nanobot memory vs session length (single chat, fake LLM)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    if has_rss:
        ax = axes[1]
        ax.plot(turns, rss_mib, color="tab:green", marker="^", markersize=3)
        ax.set_ylabel("process RSS (MiB)")
        ax.grid(True, alpha=0.3)

    ax = axes[-1]
    ax.plot(turns, jsonl_kib, color="tab:gray")
    ax.set_ylabel("JSONL on disk (KiB)")
    ax.set_xlabel("turn")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--turns", type=int, default=200)
    p.add_argument("--user-bytes", type=int, default=200)
    p.add_argument("--reply-bytes", type=int, default=800)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path(tempfile.mkdtemp(prefix="memload_")))
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Construct SessionManager with streaming_history=True (memory-model Step C).",
    )
    args = p.parse_args()

    _silence_logs()
    args.out.mkdir(parents=True, exist_ok=True)
    workspace = args.out / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"workspace={workspace}")
    print(f"out={args.out}")

    print(f"streaming={args.streaming}")

    samples = asyncio.run(
        run(
            turns=args.turns,
            user_bytes=args.user_bytes,
            reply_bytes=args.reply_bytes,
            workspace=workspace,
            sample_every=args.sample_every,
            streaming=args.streaming,
        )
    )

    csv_path = args.out / "memory.csv"
    png_path = args.out / "memory.png"
    write_csv(samples, csv_path)
    print(f"wrote {csv_path}")
    try:
        write_png(samples, png_path)
        print(f"wrote {png_path}")
    except ImportError:
        print("matplotlib not installed — skipping PNG. Re-run with `uv run --with matplotlib …`")


if __name__ == "__main__":
    main()
