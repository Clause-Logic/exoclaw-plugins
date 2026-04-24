"""Durable inbound enqueue — ``DBOSExecutor.enqueue_inbound``.

Verifies the wiring that closes the crash window between "channel
received a message" and "agent started processing it". The full
DBOS-level durability (workflow args journaled to SQLite before
enqueue returns) is exercised by ``test_durability.py``; these tests
focus on what lives in this repo: correct workflow-id construction
for dedup, correct kwargs forwarded to the queue, and that the
capability flag is advertised so ``AgentLoop`` wires the bus hook.

Unit-style on purpose: exercising ``queue.enqueue_async`` end-to-end
would require a second session-scoped DBOS fixture in this same
pytest run, which conflicts with the DBOS/asyncio-loop teardown
already performed by the other test files (DBOS shuts down the
event loop's default ThreadPoolExecutor, so a fresh DBOS in the
same process can still see dead futures).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from exoclaw.bus.events import InboundMessage


class TestDBOSInboundEnqueueCapability:
    def test_capability_flag_advertised(self) -> None:
        """``handles_inbound_enqueue`` is the flag ``AgentLoop`` checks
        when deciding whether to wire the bus's inbound hook. Must be
        ``True`` on ``DBOSExecutor`` or the crash window reopens."""
        from exoclaw_executor_dbos import DBOSExecutor

        assert DBOSExecutor.handles_inbound_enqueue is True

    def test_queue_declared_with_concurrency_one(self) -> None:
        """``concurrency=1`` is the serialization guarantee that
        replaces ``AgentLoop._processing_lock``. A regression to an
        unbounded queue would let two turns run concurrently in the
        same session and race the per-session message buffer."""
        from exoclaw_executor_dbos.turn import _INBOUND_QUEUE, INBOUND_QUEUE_NAME

        assert _INBOUND_QUEUE.name == INBOUND_QUEUE_NAME
        assert _INBOUND_QUEUE.concurrency == 1


@pytest.mark.asyncio
class TestDBOSInboundEnqueueWiring:
    async def test_enqueue_uses_message_id_for_workflow_id(self) -> None:
        """Channels that provide a stable ``message_id`` get a
        deterministic workflow id. DBOS dedupes on workflow id, so a
        channel replay (Zulip event-queue re-registration, Slack retry)
        collapses into the first enqueue instead of double-processing."""
        from exoclaw_executor_dbos import DBOSExecutor

        executor = DBOSExecutor()
        msg = InboundMessage(
            channel="zulip",
            sender_id="123",
            chat_id="589226:email check",
            content="hi",
            metadata={"message_id": "msg-abc-123"},
        )

        enqueue_mock = AsyncMock()
        captured_wfid: dict[str, str] = {}

        class _FakeSetWorkflowID:
            def __init__(self, wfid: str) -> None:
                captured_wfid["wfid"] = wfid

            def __enter__(self) -> None:
                return None

            def __exit__(self, *args: object) -> None:
                return None

        with (
            patch("exoclaw_executor_dbos.turn._get_inbound_queue") as get_queue,
            patch("exoclaw_executor_dbos.executor.SetWorkflowID", _FakeSetWorkflowID),
        ):
            get_queue.return_value.enqueue_async = enqueue_mock
            await executor.enqueue_inbound(msg)

        assert captured_wfid["wfid"] == "inbound:zulip:589226:email check:msg-abc-123"
        enqueue_mock.assert_awaited_once()
        _, kwargs = enqueue_mock.call_args
        assert kwargs["channel"] == "zulip"
        assert kwargs["sender_id"] == "123"
        assert kwargs["chat_id"] == "589226:email check"
        assert kwargs["content"] == "hi"
        assert kwargs["metadata"]["message_id"] == "msg-abc-123"

    async def test_enqueue_uses_uuid_when_message_id_missing(self) -> None:
        """Channels without a stable ``message_id`` still get a
        durable workflow — the id just isn't dedup-stable. Two
        enqueues of the same payload produce different uuids and run
        as independent workflows."""
        from exoclaw_executor_dbos import DBOSExecutor

        executor = DBOSExecutor()
        msg = InboundMessage(
            channel="cli",
            sender_id="u",
            chat_id="c",
            content="hi",
            metadata={},
        )

        wfids: list[str] = []

        class _FakeSetWorkflowID:
            def __init__(self, wfid: str) -> None:
                wfids.append(wfid)

            def __enter__(self) -> None:
                return None

            def __exit__(self, *args: object) -> None:
                return None

        with (
            patch("exoclaw_executor_dbos.turn._get_inbound_queue") as get_queue,
            patch("exoclaw_executor_dbos.executor.SetWorkflowID", _FakeSetWorkflowID),
        ):
            get_queue.return_value.enqueue_async = AsyncMock()
            await executor.enqueue_inbound(msg)
            await executor.enqueue_inbound(msg)

        assert len(wfids) == 2
        assert wfids[0].startswith("inbound:cli:c:")
        assert wfids[1].startswith("inbound:cli:c:")
        assert wfids[0] != wfids[1]

    async def test_enqueue_forwards_all_inbound_message_fields(self) -> None:
        """Every ``InboundMessage`` field that matters at dispatch
        time must round-trip to ``run_inbound_turn``. A dropped field
        on this boundary silently degrades behavior (e.g. losing
        ``model_override`` would reset per-turn model selection)."""
        from exoclaw_executor_dbos import DBOSExecutor

        executor = DBOSExecutor()
        msg = InboundMessage(
            channel="zulip",
            sender_id="u|u@example.com",
            chat_id="1:t",
            content="text",
            media=["attach-1", "attach-2"],
            metadata={"message_id": "m1", "extra": {"k": "v"}},
            session_key_override="zulip:override",
            model_override="zai/glm-5.1",
        )

        enqueue_mock = AsyncMock()
        with (
            patch("exoclaw_executor_dbos.turn._get_inbound_queue") as get_queue,
            patch("exoclaw_executor_dbos.executor.SetWorkflowID"),
        ):
            get_queue.return_value.enqueue_async = enqueue_mock
            await executor.enqueue_inbound(msg)

        _, kwargs = enqueue_mock.call_args
        assert kwargs["media"] == ["attach-1", "attach-2"]
        assert kwargs["metadata"]["extra"] == {"k": "v"}
        assert kwargs["session_key_override"] == "zulip:override"
        assert kwargs["model_override"] == "zai/glm-5.1"
