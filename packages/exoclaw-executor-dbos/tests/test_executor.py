"""Basic tests for exoclaw-executor-dbos."""

import dataclasses
import re
from unittest.mock import AsyncMock, MagicMock

from exoclaw.agent.tools.protocol import ToolContext
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

    def test_deepcopy_through_toolcontext_asdict(self) -> None:
        """``execute_tool`` calls ``dataclasses.asdict(ctx)`` to serialize
        step arguments. ``ToolContext.executor`` references the executor
        singleton, and asdict's internal deep-copy chokes on the
        ``ContextVar`` instance attribute unless ``__deepcopy__`` is
        overridden.

        Regression for: every tool call from an ``/agent/call``-initiated
        turn raised ``TypeError: cannot pickle '_contextvars.ContextVar'``
        after the per-instance-ContextVar refactor landed.
        """
        executor = DBOSExecutor()
        ctx = ToolContext(
            session_key="test:deepcopy",
            channel="ipc",
            chat_id="x",
            executor=executor,
        )
        data = dataclasses.asdict(ctx)  # must not raise
        assert data["session_key"] == "test:deepcopy"

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


class TestDBOSExecutorPriorDeltaSplit:
    """Phase 2a/2b invariants on ``DBOSExecutor``. The single
    ``_messages_var`` was replaced with a ``_prior_var`` source +
    ``_delta_var`` list split — mirrors what ``DirectExecutor`` got
    in exoclaw 0.19.1/0.20.0. Without this, the phase 2b disk-backed
    prior-source path can't land on the openclaw-deployed executor.
    """

    def test_set_seeds_prior_not_delta(self) -> None:
        executor = DBOSExecutor()
        executor.set_messages([{"role": "system", "content": "sys"}])

        assert executor._get_prior() == [{"role": "system", "content": "sys"}]
        assert executor._get_delta() == []

    def test_append_grows_delta_not_prior(self) -> None:
        executor = DBOSExecutor()
        executor.set_messages([{"role": "system", "content": "sys"}])
        executor.append_messages([{"role": "assistant", "content": "ok"}])

        assert executor._get_prior() == [{"role": "system", "content": "sys"}]
        assert executor._get_delta() == [{"role": "assistant", "content": "ok"}]

    def test_set_clears_delta(self) -> None:
        """Mid-turn ``set_messages`` (compaction) must wipe any delta
        that grew on the pre-compaction iterations — otherwise the
        next ``append_messages`` double-counts."""
        executor = DBOSExecutor()
        executor.set_messages([{"role": "user", "content": "u"}])
        executor.append_messages([{"role": "assistant", "content": "a1"}])

        executor.set_messages([{"role": "user", "content": "compacted"}])

        assert executor._get_prior() == [{"role": "user", "content": "compacted"}]
        assert executor._get_delta() == []

    def test_set_messages_snapshot_isolated_from_caller_mutation(self) -> None:
        """Pre-refactor fresh-list-per-call guarantee preserved — a
        caller that mutates its list after ``set_messages`` must not
        see the mutation leak into the executor."""
        executor = DBOSExecutor()
        msgs: list[dict[str, object]] = [{"role": "user", "content": "original"}]
        executor.set_messages(msgs)

        msgs.append({"role": "assistant", "content": "injected"})
        assert executor.load_messages() == [{"role": "user", "content": "original"}]

    def test_set_prior_source_invokes_source_on_each_load(self) -> None:
        """Phase 2b surface: the source is called on every
        ``load_messages`` so a disk-backed implementation can re-read
        the history slice instead of holding a Python list."""
        executor = DBOSExecutor()
        counter = {"n": 0}

        def source() -> list[dict[str, object]]:
            counter["n"] += 1
            return [{"role": "user", "content": f"call-{counter['n']}"}]

        executor.set_prior_source(source)
        a = executor.load_messages()
        b = executor.load_messages()

        assert counter["n"] == 2
        assert a == [{"role": "user", "content": "call-1"}]
        assert b == [{"role": "user", "content": "call-2"}]

    def test_set_prior_source_clears_delta(self) -> None:
        executor = DBOSExecutor()
        executor.set_messages([{"role": "user", "content": "t1"}])
        executor.append_messages([{"role": "assistant", "content": "t1-asst"}])

        executor.set_prior_source(lambda: [{"role": "user", "content": "t2"}])

        assert executor._get_delta() == []
        assert executor.load_messages() == [{"role": "user", "content": "t2"}]


class TestDBOSExecutorBuildPromptAutoWire:
    """Phase 2b auto-wire: when the Conversation exposes
    ``load_persisted_history``, ``DBOSExecutor.build_prompt`` installs
    a disk-backed ``PriorSource`` instead of holding the full list.
    Successive LLM iterations re-read the history slice per call,
    reducing the between-iteration heap footprint that caused the
    openclaw OOM incident.
    """

    def _make_conversation(
        self,
        prefix: list[dict[str, str]],
        history: list[dict[str, str]],
        suffix: list[dict[str, str]],
    ) -> MagicMock:
        """Conversation whose build_prompt returns
        ``[*prefix, *history, *suffix]`` with history dicts shared
        with load_persisted_history's return. Mirrors how
        ``DefaultConversation`` + ``session.get_history`` share
        refs into ``session.messages``."""
        conv = MagicMock()
        conv.build_prompt = AsyncMock(return_value=[*prefix, *history, *suffix])
        conv.load_persisted_history = lambda _sid: list(history)
        conv.record = AsyncMock()
        conv.clear = AsyncMock(return_value=True)
        return conv

    async def test_installs_prior_source_when_history_present(self) -> None:
        prefix = [{"role": "system", "content": "sys"}]
        history = [
            {"role": "user", "content": "h1"},
            {"role": "assistant", "content": "h2"},
        ]
        suffix = [{"role": "user", "content": "new"}]
        conv = self._make_conversation(prefix, history, suffix)
        executor = DBOSExecutor()

        await executor.build_prompt(conv, "s:1", "new")

        stored = executor._prior_var.get()
        assert callable(stored)
        assert stored() == [*prefix, *history, *suffix]

    async def test_prior_source_reflects_history_mutations(self) -> None:
        """Disk-backing's payoff: a new message appended to session
        history between LLM iterations shows up on the next
        ``load_messages`` without a full prompt rebuild."""
        prefix = [{"role": "system", "content": "sys"}]
        suffix = [{"role": "user", "content": "new"}]
        history_ref = [{"role": "user", "content": "h1"}]

        conv = MagicMock()
        conv.build_prompt = AsyncMock(return_value=[*prefix, *history_ref, *suffix])
        conv.load_persisted_history = lambda _sid: list(history_ref)
        conv.record = AsyncMock()
        conv.clear = AsyncMock(return_value=True)

        executor = DBOSExecutor()
        await executor.build_prompt(conv, "s:1", "new")

        first = executor.load_messages()
        assert [m["content"] for m in first] == ["sys", "h1", "new"]

        history_ref.append({"role": "assistant", "content": "h2"})

        second = executor.load_messages()
        assert [m["content"] for m in second] == ["sys", "h1", "h2", "new"]

    async def test_falls_back_when_history_empty(self) -> None:
        prefix = [{"role": "system", "content": "sys"}]
        suffix = [{"role": "user", "content": "new"}]
        conv = self._make_conversation(prefix, [], suffix)
        executor = DBOSExecutor()

        await executor.build_prompt(conv, "s:1", "new")

        # Snapshot fallback — full list captured in a closure.
        assert executor.load_messages() == [*prefix, *suffix]

    async def test_falls_back_when_history_refs_dont_match(self) -> None:
        """If a PromptBuilder deep-copies history (breaks shared
        refs), id() matching fails. Snapshot fallback keeps
        behaviour correct."""
        history = [{"role": "user", "content": "h1"}]
        prefix = [{"role": "system", "content": "sys"}]
        suffix = [{"role": "user", "content": "new"}]

        conv = MagicMock()
        conv.build_prompt = AsyncMock(return_value=[*prefix, *[dict(m) for m in history], *suffix])
        conv.load_persisted_history = lambda _sid: list(history)
        conv.record = AsyncMock()
        conv.clear = AsyncMock(return_value=True)

        executor = DBOSExecutor()
        await executor.build_prompt(conv, "s:1", "new")

        first = executor.load_messages()
        history.append({"role": "assistant", "content": "injected"})
        second = executor.load_messages()
        assert first == second  # snapshot doesn't see post-build mutations

    async def test_falls_back_when_history_refs_partially_overlap(self) -> None:
        """If a PromptBuilder preserves SOME history dicts by ref but
        replaces others (e.g. tool-result compaction rewrites a
        subset), id matching would still hit on the preserved ones.
        Using that match to build a lazy source would re-inject the
        UN-transformed full history on later iterations, diverging
        from what the initial LLM call actually saw. The slice-level
        id check catches this and bails to the snapshot path.

        Regression for PR #57 review — without the full-slice id
        verification, this test would install a broken lazy source
        and the second ``load_messages`` would reflect mutations to
        ``history`` (the un-transformed version).
        """
        history = [
            {"role": "user", "content": "h1"},
            {"role": "assistant", "content": "h2"},
        ]
        prefix = [{"role": "system", "content": "sys"}]
        suffix = [{"role": "user", "content": "new"}]

        conv = MagicMock()
        # Shared ref for h1, fresh dict for h2 — partial overlap.
        conv.build_prompt = AsyncMock(
            return_value=[*prefix, history[0], dict(history[1]), *suffix]
        )
        conv.load_persisted_history = lambda _sid: list(history)
        conv.record = AsyncMock()
        conv.clear = AsyncMock(return_value=True)

        executor = DBOSExecutor()
        await executor.build_prompt(conv, "s:1", "new")

        first = executor.load_messages()
        history.append({"role": "assistant", "content": "injected"})
        second = executor.load_messages()
        # Snapshot path — mutations to ``history`` must NOT leak into
        # the executor's prior view. A broken lazy source would let
        # them through.
        assert first == second

    async def test_falls_back_when_no_load_persisted_history(self) -> None:
        prefix = [{"role": "system", "content": "sys"}]
        suffix = [{"role": "user", "content": "new"}]

        class _LegacyConversation:
            async def build_prompt(self, *a: object, **kw: object) -> list[dict[str, str]]:
                return [*prefix, *suffix]

            async def record(self, *a: object, **kw: object) -> None:
                pass

            async def clear(self, *a: object, **kw: object) -> bool:
                return True

            def list_sessions(self) -> list[dict[str, object]]:
                return []

        executor = DBOSExecutor()
        await executor.build_prompt(_LegacyConversation(), "s:1", "new")  # type: ignore[arg-type]

        assert executor.load_messages() == [*prefix, *suffix]


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
