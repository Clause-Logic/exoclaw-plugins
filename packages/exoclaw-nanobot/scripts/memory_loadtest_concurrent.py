"""Memory vs concurrent in-flight chats — sweep N, compare streaming vs cached.

Companion to ``memory_loadtest.py`` (which sweeps session length on a
single chat). Multi-tenant openclaw cares about the *concurrent*
dimension: how does total Python heap grow when N independent sessions
are all mid-turn at once?

Hypothesis to test:

* ``streaming_history=False`` (today's default): per-session RAM grows
  with session history, so total = N × tail_size + per-turn working set.
* ``streaming_history=True`` (this PR): per-session RAM is a small
  constant (just the metadata shell), so total ≈ N × per-turn working
  set, independent of how long each chat is.

Procedure:

1. Pre-seed K sessions with L turns of synthetic JSONL history, so
   each session has a meaningful unconsolidated tail.
2. For each ``N`` in the sweep:
     * Spawn N concurrent ``process_direct`` calls, one per session.
     * A barrier inside ``FakeProvider.chat`` blocks each task until
       all N have entered. The main loop samples memory at that
       moment — when all N tasks are simultaneously holding their
       per-turn working set.
     * Release the barrier; await completion.
3. Run the whole sweep twice — once with ``streaming_history=False``,
   once with ``True``. Plot both as lines on the same chart.

Run:

    uv run --with matplotlib --with psutil python \
        packages/exoclaw-nanobot/scripts/memory_loadtest_concurrent.py \
        --concurrency 1,2,4,8,16,32,64 --seed-turns 100 --out /tmp/memload_v2
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import logging
import tempfile
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
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


class BarrierProvider:
    """LLMProvider that blocks every ``chat`` call until N have entered.

    Lets the loadtest sample memory at the exact moment all N concurrent
    tasks are simultaneously inside ``provider.chat`` — the natural peak
    in-flight point for measuring per-task working set.
    """

    def __init__(self, default_model: str, reply_bytes: int) -> None:
        self._model = default_model
        self._reply = "x" * reply_bytes
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.in_flight = 0
        self.target = 0

    def reset(self, target: int) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.in_flight = 0
        self.target = target

    def get_default_model(self) -> str:
        return self._model

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        self.in_flight += 1
        if self.in_flight >= self.target:
            self.entered.set()
        await self.release.wait()
        return LLMResponse(content=self._reply, finish_reason="stop")


@dataclass
class Sample:
    mode: str  # "cached" or "streaming"
    phase: str  # "in_flight" (mid-chat) or "post_turn" (after completion)
    concurrency: int
    tm_current: int
    tm_peak: int
    rss: int


def _rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return 0


def _seed_sessions(workspace: Path, num_sessions: int, seed_turns: int) -> list[str]:
    """Write pre-populated JSONL files so each session has a meaningful
    unconsolidated tail. Returns the list of session keys.
    """
    sessions_dir = workspace / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for s in range(num_sessions):
        key = f"loadtest:s-{s}"
        keys.append(key)
        path = sessions_dir / f"loadtest_s-{s}.jsonl"
        with open(path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "_type": "metadata",
                        "key": key,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "metadata": {},
                        "last_consolidated": 0,
                    }
                )
                + "\n"
            )
            for t in range(seed_turns):
                f.write(
                    json.dumps(
                        {
                            "role": "user",
                            "content": "u" * 200,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": "a" * 800,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    + "\n"
                )
    return keys


async def _run_mode(
    *,
    streaming: bool,
    workspace: Path,
    session_keys: list[str],
    concurrency_sweep: list[int],
    reply_bytes: int,
) -> list[Sample]:
    bus = MessageBus()
    provider = BarrierProvider(default_model="fake-model", reply_bytes=reply_bytes)

    history_store = SessionManager(workspace, streaming_history=streaming)
    memory_store = MemoryStore(workspace, provider, "fake-model", history=history_store)
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

    samples: list[Sample] = []
    mode = "streaming" if streaming else "cached"

    # Warm the WeakValueDictionary by touching every session once,
    # then drop refs so they aren't strong-held into the sweep.
    for key in session_keys:
        history_store.get_or_create(key)
    gc.collect()

    for n in concurrency_sweep:
        if n > len(session_keys):
            print(f"  skipping concurrency={n} (only {len(session_keys)} sessions seeded)")
            continue
        provider.reset(target=n)
        tracemalloc.reset_peak()

        async def _one(i: int) -> None:
            await loop.process_direct(
                "hi",
                session_key=session_keys[i],
                channel="cli",
                chat_id=f"s-{i}",
            )
            if streaming:
                executor.set_prior_source(
                    lambda key=session_keys[i]: conversation.load_persisted_history(key)
                )

        tasks = [asyncio.create_task(_one(i)) for i in range(n)]

        await provider.entered.wait()
        # ── In-flight sample ─────────────────────────────────────────
        # All N tasks are suspended inside ``provider.chat``. Each
        # task's frame holds the prompt list it passed in (system +
        # history + new user message). Step B (streaming request body)
        # is what attacks this peak; streaming_history doesn't help
        # here.
        gc.collect()
        tm_current, tm_peak = tracemalloc.get_traced_memory()
        samples.append(
            Sample(
                mode=mode,
                phase="in_flight",
                concurrency=n,
                tm_current=tm_current,
                tm_peak=tm_peak,
                rss=_rss_bytes(),
            )
        )
        print(
            f"  {mode:>9} n={n:>3} in_flight  "
            f"tm_current={tm_current / 1024:>8.1f}KiB  "
            f"tm_peak={tm_peak / 1024:>8.1f}KiB"
        )

        provider.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

        # ── Post-turn sample ─────────────────────────────────────────
        # Tasks complete; under streaming the manual ``set_prior_source``
        # in ``_one`` has just installed a lazy source. Reset peak so
        # we measure the true post-completion baseline rather than
        # the in-flight peak.
        gc.collect()
        tracemalloc.reset_peak()
        tm_current, tm_peak = tracemalloc.get_traced_memory()
        samples.append(
            Sample(
                mode=mode,
                phase="post_turn",
                concurrency=n,
                tm_current=tm_current,
                tm_peak=tm_peak,
                rss=_rss_bytes(),
            )
        )
        print(f"  {mode:>9} n={n:>3} post_turn  tm_current={tm_current / 1024:>8.1f}KiB")

    return samples


async def run(
    *,
    concurrency_sweep: list[int],
    seed_turns: int,
    reply_bytes: int,
    out: Path,
) -> list[Sample]:
    max_concurrency = max(concurrency_sweep)
    print(f"seeding {max_concurrency} sessions × {seed_turns} turns each…")

    cached_workspace = out / "workspace_cached"
    streaming_workspace = out / "workspace_streaming"
    cached_workspace.mkdir(parents=True, exist_ok=True)
    streaming_workspace.mkdir(parents=True, exist_ok=True)
    cached_keys = _seed_sessions(cached_workspace, max_concurrency, seed_turns)
    streaming_keys = _seed_sessions(streaming_workspace, max_concurrency, seed_turns)

    tracemalloc.start()
    print("\ncached run:")
    cached_samples = await _run_mode(
        streaming=False,
        workspace=cached_workspace,
        session_keys=cached_keys,
        concurrency_sweep=concurrency_sweep,
        reply_bytes=reply_bytes,
    )
    print("\nstreaming run:")
    streaming_samples = await _run_mode(
        streaming=True,
        workspace=streaming_workspace,
        session_keys=streaming_keys,
        concurrency_sweep=concurrency_sweep,
        reply_bytes=reply_bytes,
    )
    tracemalloc.stop()

    return cached_samples + streaming_samples


def write_csv(samples: list[Sample], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "phase", "concurrency", "tm_current", "tm_peak", "rss"])
        for s in samples:
            w.writerow([s.mode, s.phase, s.concurrency, s.tm_current, s.tm_peak, s.rss])


def write_png(samples: list[Sample], out: Path) -> None:
    import matplotlib.pyplot as plt

    def pick(mode: str, phase: str) -> list[Sample]:
        return sorted(
            [s for s in samples if s.mode == mode and s.phase == phase],
            key=lambda s: s.concurrency,
        )

    def xs(group: list[Sample]) -> list[int]:
        return [s.concurrency for s in group]

    def ys_kib(group: list[Sample], attr: str = "tm_current") -> list[float]:
        return [getattr(s, attr) / 1024 for s in group]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: between-turn baseline. The Step C win lives here — cached
    # holds session.messages × N, streaming doesn't.
    ax = axes[0]
    ax.plot(
        xs(pick("cached", "post_turn")),
        ys_kib(pick("cached", "post_turn")),
        label="cached",
        marker="o",
    )
    ax.plot(
        xs(pick("streaming", "post_turn")),
        ys_kib(pick("streaming", "post_turn")),
        label="streaming",
        marker="s",
    )
    ax.set_ylabel("tracemalloc current (KiB)")
    ax.set_title("Post-turn baseline — RAM held between LLM calls (Step C win)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Bottom: peak during chat() — both modes hold the prompt list to
    # pass to provider.chat. This is Step B territory (streaming
    # request body); streaming_history alone doesn't shrink it.
    ax = axes[1]
    ax.plot(
        xs(pick("cached", "in_flight")),
        ys_kib(pick("cached", "in_flight"), "tm_peak"),
        label="cached",
        marker="o",
    )
    ax.plot(
        xs(pick("streaming", "in_flight")),
        ys_kib(pick("streaming", "in_flight"), "tm_peak"),
        label="streaming",
        marker="s",
    )
    ax.set_ylabel("tracemalloc peak (KiB)")
    ax.set_title("In-flight peak — all N tasks suspended inside chat() (Step B territory)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("concurrent in-flight chats (N)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def _parse_sweep(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--concurrency",
        type=_parse_sweep,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Comma-separated list of N values to sweep (default: 1,2,4,8,16,32,64).",
    )
    p.add_argument(
        "--seed-turns",
        type=int,
        default=100,
        help="Pre-seed each session with this many turns of synthetic history (default 100).",
    )
    p.add_argument("--reply-bytes", type=int, default=800)
    p.add_argument("--out", type=Path, default=Path(tempfile.mkdtemp(prefix="memload_v2_")))
    args = p.parse_args()

    _silence_logs()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"out={args.out}")
    print(f"concurrency sweep: {args.concurrency}")

    samples = asyncio.run(
        run(
            concurrency_sweep=args.concurrency,
            seed_turns=args.seed_turns,
            reply_bytes=args.reply_bytes,
            out=args.out,
        )
    )

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
