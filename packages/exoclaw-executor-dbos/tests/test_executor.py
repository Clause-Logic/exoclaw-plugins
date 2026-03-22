"""Basic tests for exoclaw-executor-dbos."""

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


class TestDBOSExecutorProtocol:
    def test_has_required_methods(self) -> None:
        executor = DBOSExecutor()
        assert hasattr(executor, "chat")
        assert hasattr(executor, "execute_tool")
        assert hasattr(executor, "build_prompt")
        assert hasattr(executor, "record")
        assert hasattr(executor, "clear")
        assert hasattr(executor, "run_hook")
