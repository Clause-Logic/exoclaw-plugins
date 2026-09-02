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

import os
from pathlib import Path
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


class TestSteeringInbox:
    def test_interrupted_drain_keeps_older_and_newer_messages(self, tmp_path: Path) -> None:
        """A crash after the rename must not discard messages received later.

        The next drain consumes the interrupted batch first, then the new
        live mailbox.  This is the filesystem half of the durable hand-off;
        the DBOS step replay behavior is covered in ``test_durability.py``.
        """
        from exoclaw_executor_dbos.steering import (
            _append_message,
            _drain_messages,
            _inbox_path,
        )

        root = tmp_path / "steering"
        session_id = "zulip:42:topic"
        _append_message(root, session_id, "older")
        inbox_path = _inbox_path(root, session_id)
        os.replace(inbox_path, inbox_path.with_suffix(".draining"))
        _append_message(root, session_id, "newer")

        assert _drain_messages(root, session_id) == ["older", "newer"]

    async def test_active_turn_follow_up_bypasses_inbound_queue(self, tmp_path: Path) -> None:
        """A same-session follow-up is persisted for steering, not queued.

        This is the behavior that lets an active tool/model turn observe the
        new user message at its next safe boundary instead of only after the
        current inbound queue partition is released.
        """
        from exoclaw_executor_dbos import DBOSExecutor

        executor = DBOSExecutor(steering_workspace=tmp_path)
        msg = InboundMessage(
            channel="zulip",
            sender_id="u",
            chat_id="42:topic",
            content="actually, stop after this",
            metadata={"message_id": "follow-up-1"},
        )
        handle = AsyncMock()
        handle.get_result = AsyncMock()
        await executor.activate_steering(msg.session_key)
        try:
            with patch(
                "exoclaw_executor_dbos.executor.DBOS.start_workflow_async",
                new=AsyncMock(return_value=handle),
            ) as start_workflow:
                await executor.enqueue_inbound(msg)
        finally:
            await executor.deactivate_steering(msg.session_key)

        start_workflow.assert_awaited_once()
        assert start_workflow.call_args is not None
        args = start_workflow.call_args.args
        assert args[0].__name__ == "run_inbound_turn"
        assert start_workflow.call_args.kwargs["content"] == msg.content
        assert start_workflow.call_args.kwargs["session_key_override"] is None
        handle.get_result.assert_awaited_once()

    async def test_steering_and_later_queue_retry_share_workflow_id(self, tmp_path: Path) -> None:
        """A channel retry is deduped even if the active turn has finished.

        DBOS treats a repeated workflow id as a duplicate only when it names
        the same workflow function.  The active path starts
        ``run_inbound_turn`` directly and the later normal path queues that
        same function, so both must retain the exact inbound workflow id.
        """
        from exoclaw_executor_dbos import DBOSExecutor

        executor = DBOSExecutor(steering_workspace=tmp_path)
        msg = InboundMessage(
            channel="zulip",
            sender_id="u",
            chat_id="42:topic",
            content="follow up",
            metadata={"message_id": "event-42"},
        )
        workflow_ids: list[str] = []

        class _CaptureWorkflowID:
            def __init__(self, workflow_id: str) -> None:
                workflow_ids.append(workflow_id)

            def __enter__(self) -> None:
                return None

            def __exit__(self, *args: object) -> None:
                return None

        handle = AsyncMock()
        handle.get_result = AsyncMock()
        queue_enqueue = AsyncMock()
        await executor.activate_steering(msg.session_key)
        try:
            with (
                patch(
                    "exoclaw_executor_dbos.executor.SetWorkflowID",
                    _CaptureWorkflowID,
                ),
                patch(
                    "exoclaw_executor_dbos.executor.DBOS.start_workflow_async",
                    new=AsyncMock(return_value=handle),
                ),
            ):
                await executor.enqueue_inbound(msg)
        finally:
            await executor.deactivate_steering(msg.session_key)

        with (
            patch("exoclaw_executor_dbos.turn._get_inbound_queue") as get_queue,
            patch(
                "exoclaw_executor_dbos.executor.SetWorkflowID",
                _CaptureWorkflowID,
            ),
        ):
            get_queue.return_value.enqueue_async = queue_enqueue
            await executor.enqueue_inbound(msg)

        assert workflow_ids == ["inbound:zulip:42:topic:event-42"] * 2
        queue_enqueue.assert_awaited_once()
        assert msg.session_key not in executor._active_sessions
