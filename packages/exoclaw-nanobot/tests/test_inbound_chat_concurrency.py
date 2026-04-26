"""End-to-end test that two different chats can process inbound messages
concurrently.

The ``exoclaw-inbound`` DBOS queue exists to replace the in-memory
``AgentLoop._processing_lock``. Its job is to serialize turns *within
the same chat/session* — two messages from the same chat must not
race the per-session message buffer. It must NOT serialize across
unrelated chats: a long-running turn in chat A blocking every other
chat is the whole-bot stall mode that motivated this test.

Symptom in the wild: a multi-minute scheduled turn in one chat held
the queue's only concurrency slot, and an unrelated message in a
different chat sat ``ENQUEUED`` for many minutes behind it. Subagent
result deliveries (system-channel inbound messages) piled up too. The
queue was declared with global ``concurrency=1`` and no partitioning,
so every chat shared one slot.

Fix shape this test enforces: the inbound queue is partitioned by chat,
so ``concurrency=1`` is a per-chat guarantee, not a global one. Two
chats enqueued at the same time observe peak in-flight count == 2 — the
within-chat serialization is left to per-chat partition semantics
(verified separately in ``test_inbound_enqueue.py``).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import pytest
from dbos import DBOS, DBOSConfig
from exoclaw.bus.events import InboundMessage
from exoclaw_executor_dbos import DBOSExecutor

_DB_PATH = f"/tmp/dbos_inbound_chat_concurrency_test_{os.getpid()}.sqlite"


@pytest.fixture(scope="module")
def dbos_instance() -> Any:
    """Module-scoped DBOS — singleton; isolate lifetime to this module."""
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "inbound-chat-concurrency-test",
        "system_database_url": f"sqlite:///{_DB_PATH}",
        "enable_otlp": False,
    }
    import exoclaw_executor_dbos.turn  # noqa: F401  registers run_inbound_turn

    dbos = DBOS(config=config)
    DBOS.launch()
    yield dbos
    DBOS.destroy()
    if os.path.exists(_DB_PATH):
        os.unlink(_DB_PATH)


def _reset_loop_default_executor() -> None:
    """Earlier DBOS-using test modules destroy DBOS in teardown, which
    shuts down the running event loop's default ``ThreadPoolExecutor``.
    ``queue.enqueue_async`` hands work off via ``loop.run_in_executor
    (None, …)`` and dies with "cannot schedule new futures after
    shutdown" against the dead executor. Clearing ``_default_executor``
    on the *currently running* loop makes asyncio lazily mint a fresh
    one. Must be called from inside the async test body — fixture
    setup runs on a different loop than the one the test ultimately
    uses, so resetting at fixture-entry time has no effect.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    loop._default_executor = None  # type: ignore[attr-defined]


class _ConcurrencyProbe:
    """Stub AgentLoop that records peak in-flight ``_dispatch`` calls.

    ``run_inbound_turn`` resolves its loop reference from the module
    global ``_loop`` and forwards the message via ``loop._dispatch``.
    Pinning a probe there lets the test observe how many dispatches
    overlap without booting a real loop.
    """

    def __init__(self, hold_seconds: float) -> None:
        self.hold_seconds = hold_seconds
        self.in_flight = 0
        self.peak_in_flight = 0
        self.dispatched_chats: list[str] = []
        self._lock = asyncio.Lock()
        self._done = asyncio.Event()
        self._target_dispatches = 0

    def expect(self, n: int) -> None:
        self._target_dispatches = n

    async def _dispatch(self, msg: InboundMessage) -> None:
        async with self._lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.dispatched_chats.append(f"{msg.channel}:{msg.chat_id}")
        try:
            await asyncio.sleep(self.hold_seconds)
        finally:
            async with self._lock:
                self.in_flight -= 1
                if len(self.dispatched_chats) >= self._target_dispatches:
                    self._done.set()

    async def wait_for_completion(self, timeout: float) -> None:
        await asyncio.wait_for(self._done.wait(), timeout=timeout)


@pytest.mark.asyncio(loop_scope="session")
async def test_two_chats_process_concurrently(dbos_instance: Any) -> None:
    """Two inbound messages on different chats must run in parallel.

    Pre-fix: ``_INBOUND_QUEUE`` is declared with ``concurrency=1`` and
    no partition — DBOS's queue worker dequeues one workflow at a time
    across the whole queue, so the second chat sits ``ENQUEUED`` until
    the first finishes. Peak in-flight is 1.

    Post-fix: queue is partitioned by chat; concurrency=1 is enforced
    per partition. Peak in-flight is 2.
    """
    import exoclaw_executor_dbos.turn as turn_mod

    _reset_loop_default_executor()

    probe = _ConcurrencyProbe(hold_seconds=0.4)
    probe.expect(2)

    saved_loop = turn_mod._loop
    turn_mod._loop = probe
    try:
        executor = DBOSExecutor()

        msg_a = InboundMessage(
            channel="zulip",
            sender_id="u-a",
            chat_id="111:chat-a",
            content="hello from a",
            metadata={"message_id": "m-a-1"},
        )
        msg_b = InboundMessage(
            channel="zulip",
            sender_id="u-b",
            chat_id="222:chat-b",
            content="hello from b",
            metadata={"message_id": "m-b-1"},
        )

        start = time.monotonic()
        await asyncio.gather(
            executor.enqueue_inbound(msg_a),
            executor.enqueue_inbound(msg_b),
        )
        await probe.wait_for_completion(timeout=5.0)
        elapsed = time.monotonic() - start
    finally:
        turn_mod._loop = saved_loop

    assert sorted(probe.dispatched_chats) == ["zulip:111:chat-a", "zulip:222:chat-b"]

    # Two 0.4s holds run concurrently in ~0.4s. If the queue serialized
    # them, wall time would approach 0.8s. The slack accounts for queue
    # poll interval (default 1.0s) — concurrent dispatch should still
    # land well under the serial ceiling.
    assert probe.peak_in_flight == 2, (
        f"inbound queue serialized two unrelated chats: peak in-flight "
        f"was {probe.peak_in_flight}, wall time {elapsed:.2f}s. The queue "
        f"is partitioned per-chat, so concurrency=1 must be a per-chat "
        f"guarantee, not a global one."
    )
