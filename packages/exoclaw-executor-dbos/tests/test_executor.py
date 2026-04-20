"""Basic tests for exoclaw-executor-dbos."""

import re

from exoclaw.providers.types import LLMResponse, ToolCallRequest
from exoclaw_executor_dbos.executor import (
    DBOSExecutor,
    _dict_to_response,
    _response_to_dict,
)


class TestSerialization:
    def test_response_roundtrip(self) -> None:
        resp = LLMResponse(
            content="hello",
            tool_calls=[ToolCallRequest(id="1", name="exec", arguments={"cmd": "ls"})],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        d = _response_to_dict(resp)
        restored = _dict_to_response(d)
        assert restored.content == "hello"
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].name == "exec"
        assert restored.tool_calls[0].arguments == {"cmd": "ls"}
        assert restored.finish_reason == "tool_calls"

    def test_response_roundtrip_no_tools(self) -> None:
        resp = LLMResponse(content="done", finish_reason="stop")
        d = _response_to_dict(resp)
        restored = _dict_to_response(d)
        assert restored.content == "done"
        assert restored.tool_calls == []
        assert restored.finish_reason == "stop"

    def test_response_roundtrip_with_reasoning(self) -> None:
        resp = LLMResponse(
            content="answer",
            reasoning_content="I thought about it",
            thinking_blocks=[{"type": "thinking", "text": "hmm"}],
        )
        d = _response_to_dict(resp)
        restored = _dict_to_response(d)
        assert restored.reasoning_content == "I thought about it"
        assert restored.thinking_blocks == [{"type": "thinking", "text": "hmm"}]

    def test_dict_to_response_does_not_mutate_input(self) -> None:
        d = {
            "content": "hi",
            "tool_calls": [{"id": "1", "name": "exec", "arguments": {}}],
            "finish_reason": "stop",
            "usage": {},
            "reasoning_content": None,
            "thinking_blocks": None,
        }
        original_keys = set(d.keys())
        _dict_to_response(d)
        assert set(d.keys()) == original_keys  # not mutated


class TestDBOSExecutorProtocol:
    def test_has_required_methods(self) -> None:
        executor = DBOSExecutor()
        assert hasattr(executor, "chat")
        assert hasattr(executor, "execute_tool")
        assert hasattr(executor, "build_prompt")
        assert hasattr(executor, "record")
        assert hasattr(executor, "clear")
        assert hasattr(executor, "run_hook")
        # The message-buffer methods were added to the Executor protocol
        # in exoclaw 0.13; subagent spawn paths call them via
        # AgentLoop.process_direct, so they must be implemented here too.
        assert hasattr(executor, "set_messages")
        assert hasattr(executor, "append_messages")
        assert hasattr(executor, "load_messages")

    def test_message_buffer_roundtrip(self) -> None:
        executor = DBOSExecutor()
        msgs: list[dict[str, object]] = [{"role": "user", "content": "hi"}]
        executor.set_messages(msgs)
        assert executor.load_messages() == msgs
        executor.append_messages([{"role": "assistant", "content": "hello"}])
        loaded = executor.load_messages()
        assert len(loaded) == 2
        assert loaded[1]["role"] == "assistant"
        # load_messages must return a copy, not the internal buffer
        loaded.clear()
        assert len(executor.load_messages()) == 2

    def test_two_instances_isolate_messages(self) -> None:
        """Two DBOSExecutors in the same task must not share a buffer.

        Isolation comes from each executor instance owning its own
        ContextVar object. A module-level ContextVar would make a
        second executor reset the first when tests (or any caller)
        construct both in one task.
        """
        a = DBOSExecutor()
        b = DBOSExecutor()
        a.set_messages([{"role": "user", "content": "a"}])
        b.set_messages([{"role": "user", "content": "b"}])
        assert [m["content"] for m in a.load_messages()] == ["a"]
        assert [m["content"] for m in b.load_messages()] == ["b"]

    async def test_concurrent_turns_isolate_messages(self) -> None:
        """Concurrent turns on the same executor must not leak messages.

        Regression for the cross-session contamination where a periodic
        background turn running concurrently with a user-initiated turn
        trampled the shared ``_messages`` list, the peer's LLM inherited
        the wrong context, and each turn's final ``record()`` wrote the
        merged transcript into the wrong session JSONL.

        Each ``asyncio.Task`` inherits a snapshot of the current context
        at creation, so per-task ContextVar bindings stay isolated.
        """
        import asyncio

        executor = DBOSExecutor()
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def turn(label: str, out: dict[str, list[dict[str, object]]]) -> None:
            executor.set_messages([{"role": "user", "content": f"{label}:user"}])
            entered.set()
            await proceed.wait()
            executor.append_messages([{"role": "assistant", "content": f"{label}:asst"}])
            out[label] = executor.load_messages()

        results: dict[str, list[dict[str, object]]] = {}
        t1 = asyncio.create_task(turn("a", results))
        await entered.wait()
        entered.clear()
        t2 = asyncio.create_task(turn("b", results))
        await entered.wait()
        proceed.set()
        await asyncio.gather(t1, t2)

        assert [m["content"] for m in results["a"]] == ["a:user", "a:asst"]
        assert [m["content"] for m in results["b"]] == ["b:user", "b:asst"]


class TestWorkflowIDUniqueness:
    def test_workflow_id_format(self) -> None:
        """run_turn sets a workflow ID matching turn:{session_id}:{uuid7_hex}."""
        from unittest.mock import AsyncMock, patch

        executor = DBOSExecutor()
        captured_ids: list[str] = []

        original_set_wf = __import__("dbos").SetWorkflowID

        class CapturingSetWorkflowID(original_set_wf):
            def __init__(self, wfid: str) -> None:
                captured_ids.append(wfid)
                super().__init__(wfid)

        with (
            patch(
                "exoclaw_executor_dbos.executor.SetWorkflowID",
                CapturingSetWorkflowID,
            ),
            patch(
                "exoclaw_executor_dbos.turn.run_durable_turn",
                new=AsyncMock(return_value=("ok", [])),
            ),
        ):
            import asyncio

            loop = AsyncMock()
            asyncio.run(executor.run_turn(loop, "sess-123", "hello"))

        assert len(captured_ids) == 1
        assert re.match(r"^turn:sess-123:[0-9a-f]{32}$", captured_ids[0])

    def test_two_calls_produce_distinct_ids(self) -> None:
        """Two run_turn calls for the same session_id get different workflow IDs."""
        from unittest.mock import AsyncMock, patch

        executor = DBOSExecutor()
        captured_ids: list[str] = []

        original_set_wf = __import__("dbos").SetWorkflowID

        class CapturingSetWorkflowID(original_set_wf):
            def __init__(self, wfid: str) -> None:
                captured_ids.append(wfid)
                super().__init__(wfid)

        with (
            patch(
                "exoclaw_executor_dbos.executor.SetWorkflowID",
                CapturingSetWorkflowID,
            ),
            patch(
                "exoclaw_executor_dbos.turn.run_durable_turn",
                new=AsyncMock(return_value=("ok", [])),
            ),
        ):
            import asyncio

            loop = AsyncMock()

            async def run_both() -> None:
                await asyncio.gather(
                    executor.run_turn(loop, "sess-abc", "msg1"),
                    executor.run_turn(loop, "sess-abc", "msg2"),
                )

            asyncio.run(run_both())

        assert len(captured_ids) == 2
        assert captured_ids[0] != captured_ids[1]
