"""Tests for exoclaw-channel-heartbeat package."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw.providers.types import LLMResponse, ToolCallRequest
from exoclaw_channel_heartbeat.service import HeartbeatService, _HEARTBEAT_TOOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(action: str = "skip", tasks: str = "") -> MagicMock:
    """Return a mock LLMProvider that responds to heartbeat tool calls."""
    tool_call = ToolCallRequest(
        id="tc1",
        name="heartbeat",
        arguments={"action": action, "tasks": tasks},
    )
    response = LLMResponse(
        content=None,
        tool_calls=[tool_call],
    )
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=response)
    return provider


def _make_no_tool_provider() -> MagicMock:
    """Return a mock provider that returns a text response (no tool call)."""
    response = LLMResponse(content="ok", tool_calls=[])
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=response)
    return provider


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def heartbeat_file(workspace: Path) -> Path:
    f = workspace / "HEARTBEAT.md"
    f.write_text("# Tasks\n- Do something", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# _HEARTBEAT_TOOL schema
# ---------------------------------------------------------------------------


class TestHeartbeatToolSchema:
    def test_schema_structure(self) -> None:
        assert len(_HEARTBEAT_TOOL) == 1
        fn = _HEARTBEAT_TOOL[0]["function"]
        assert fn["name"] == "heartbeat"
        assert "action" in fn["parameters"]["properties"]


# ---------------------------------------------------------------------------
# HeartbeatService construction
# ---------------------------------------------------------------------------


class TestHeartbeatServiceInit:
    def test_defaults(self, workspace: Path) -> None:
        provider = _make_provider()
        svc = HeartbeatService(workspace=workspace, provider=provider, model="gpt-4")
        assert svc.interval_s == 30 * 60
        assert svc.enabled is True
        assert svc._running is False
        assert svc._task is None

    def test_heartbeat_file_path(self, workspace: Path) -> None:
        svc = HeartbeatService(workspace=workspace, provider=_make_provider(), model="m")
        assert svc.heartbeat_file == workspace / "HEARTBEAT.md"


# ---------------------------------------------------------------------------
# _read_heartbeat_file
# ---------------------------------------------------------------------------


class TestReadHeartbeatFile:
    def test_missing_file_returns_none(self, workspace: Path) -> None:
        svc = HeartbeatService(workspace=workspace, provider=_make_provider(), model="m")
        assert svc._read_heartbeat_file() is None

    def test_reads_existing_file(self, workspace: Path, heartbeat_file: Path) -> None:
        svc = HeartbeatService(workspace=workspace, provider=_make_provider(), model="m")
        content = svc._read_heartbeat_file()
        assert content is not None
        assert "Tasks" in content

    def test_read_exception_returns_none(self, workspace: Path, heartbeat_file: Path) -> None:
        svc = HeartbeatService(workspace=workspace, provider=_make_provider(), model="m")
        with patch.object(Path, "read_text", side_effect=OSError("perm denied")):
            assert svc._read_heartbeat_file() is None


# ---------------------------------------------------------------------------
# _decide
# ---------------------------------------------------------------------------


class TestDecide:
    async def test_returns_run_with_tasks(self, workspace: Path) -> None:
        provider = _make_provider(action="run", tasks="send email")
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        action, tasks = await svc._decide("some content")
        assert action == "run"
        assert tasks == "send email"

    async def test_returns_skip(self, workspace: Path) -> None:
        provider = _make_provider(action="skip")
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        action, tasks = await svc._decide("some content")
        assert action == "skip"
        assert tasks == ""

    async def test_no_tool_call_returns_skip(self, workspace: Path) -> None:
        provider = _make_no_tool_provider()
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        action, tasks = await svc._decide("some content")
        assert action == "skip"
        assert tasks == ""

    async def test_missing_action_defaults_to_skip(self, workspace: Path) -> None:
        tool_call = ToolCallRequest(id="tc1", name="heartbeat", arguments={})
        response = LLMResponse(content=None, tool_calls=[tool_call])
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=response)
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        action, _ = await svc._decide("content")
        assert action == "skip"


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    async def test_start_disabled(self, workspace: Path) -> None:
        svc = HeartbeatService(
            workspace=workspace, provider=_make_provider(), model="m", enabled=False
        )
        await svc.start()
        assert svc._running is False
        assert svc._task is None

    async def test_start_creates_task(self, workspace: Path) -> None:
        svc = HeartbeatService(
            workspace=workspace, provider=_make_provider(), model="m", interval_s=9999
        )
        await svc.start()
        assert svc._running is True
        assert svc._task is not None
        svc.stop()

    async def test_start_already_running_noop(self, workspace: Path) -> None:
        svc = HeartbeatService(
            workspace=workspace, provider=_make_provider(), model="m", interval_s=9999
        )
        await svc.start()
        task1 = svc._task
        await svc.start()  # second start → warning, no new task
        assert svc._task is task1
        svc.stop()

    def test_stop_cancels_task(self, workspace: Path) -> None:
        svc = HeartbeatService(workspace=workspace, provider=_make_provider(), model="m")
        mock_task = MagicMock()
        svc._task = mock_task
        svc._running = True
        svc.stop()
        mock_task.cancel.assert_called_once()
        assert svc._running is False
        assert svc._task is None

    def test_stop_when_not_running(self, workspace: Path) -> None:
        svc = HeartbeatService(workspace=workspace, provider=_make_provider(), model="m")
        svc.stop()  # should not raise


# ---------------------------------------------------------------------------
# _tick
# ---------------------------------------------------------------------------


class TestTick:
    async def test_tick_no_heartbeat_file(self, workspace: Path) -> None:
        provider = _make_provider()
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        await svc._tick()
        provider.chat.assert_not_called()

    async def test_tick_skip_action(self, workspace: Path, heartbeat_file: Path) -> None:
        provider = _make_provider(action="skip")
        on_execute = AsyncMock(return_value="result")
        svc = HeartbeatService(
            workspace=workspace, provider=provider, model="m", on_execute=on_execute
        )
        await svc._tick()
        on_execute.assert_not_called()

    async def test_tick_run_calls_execute(self, workspace: Path, heartbeat_file: Path) -> None:
        provider = _make_provider(action="run", tasks="send email")
        on_execute = AsyncMock(return_value="Email sent")
        on_notify = AsyncMock()
        svc = HeartbeatService(
            workspace=workspace,
            provider=provider,
            model="m",
            on_execute=on_execute,
            on_notify=on_notify,
        )
        await svc._tick()
        on_execute.assert_called_once_with("send email")
        on_notify.assert_called_once_with("Email sent")

    async def test_tick_run_no_on_execute(self, workspace: Path, heartbeat_file: Path) -> None:
        provider = _make_provider(action="run", tasks="task")
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        await svc._tick()  # should not raise

    async def test_tick_run_empty_response_no_notify(
        self, workspace: Path, heartbeat_file: Path
    ) -> None:
        provider = _make_provider(action="run", tasks="task")
        on_execute = AsyncMock(return_value="")
        on_notify = AsyncMock()
        svc = HeartbeatService(
            workspace=workspace,
            provider=provider,
            model="m",
            on_execute=on_execute,
            on_notify=on_notify,
        )
        await svc._tick()
        on_notify.assert_not_called()

    async def test_tick_exception_logged(self, workspace: Path, heartbeat_file: Path) -> None:
        provider = MagicMock()
        provider.chat = AsyncMock(side_effect=RuntimeError("boom"))
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        await svc._tick()  # should not raise


# ---------------------------------------------------------------------------
# trigger_now
# ---------------------------------------------------------------------------


class TestTriggerNow:
    async def test_no_heartbeat_file_returns_none(self, workspace: Path) -> None:
        svc = HeartbeatService(workspace=workspace, provider=_make_provider(), model="m")
        result = await svc.trigger_now()
        assert result is None

    async def test_skip_returns_none(self, workspace: Path, heartbeat_file: Path) -> None:
        provider = _make_provider(action="skip")
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        result = await svc.trigger_now()
        assert result is None

    async def test_run_returns_execute_result(
        self, workspace: Path, heartbeat_file: Path
    ) -> None:
        provider = _make_provider(action="run", tasks="task")
        on_execute = AsyncMock(return_value="done")
        svc = HeartbeatService(
            workspace=workspace, provider=provider, model="m", on_execute=on_execute
        )
        result = await svc.trigger_now()
        assert result == "done"

    async def test_run_no_on_execute_returns_none(
        self, workspace: Path, heartbeat_file: Path
    ) -> None:
        provider = _make_provider(action="run", tasks="task")
        svc = HeartbeatService(workspace=workspace, provider=provider, model="m")
        result = await svc.trigger_now()
        assert result is None


# ---------------------------------------------------------------------------
# _run_loop
# ---------------------------------------------------------------------------


class TestRunLoop:
    async def test_cancelled_error_exits_cleanly(self, workspace: Path) -> None:
        provider = _make_provider()
        svc = HeartbeatService(
            workspace=workspace, provider=provider, model="m", interval_s=9999
        )
        await svc.start()
        svc.stop()  # cancels the task
        # Give event loop a chance to process cancellation
        import asyncio
        await asyncio.sleep(0)

    async def test_run_loop_executes_tick(self, workspace: Path, heartbeat_file: Path) -> None:
        provider = _make_provider(action="skip")
        svc = HeartbeatService(
            workspace=workspace, provider=provider, model="m", interval_s=9999
        )
        svc._running = True
        tick_calls: list[int] = []

        async def fake_tick() -> None:
            tick_calls.append(1)
            svc._running = False  # stop after one iteration

        svc._tick = fake_tick  # type: ignore[method-assign]

        with patch("asyncio.sleep", new=AsyncMock()):
            await svc._run_loop()

        assert len(tick_calls) == 1

    async def test_run_loop_cancelled_error_exits(self, workspace: Path) -> None:
        provider = _make_provider(action="skip")
        svc = HeartbeatService(
            workspace=workspace, provider=provider, model="m", interval_s=9999
        )
        svc._running = True

        with patch("asyncio.sleep", new=AsyncMock(side_effect=asyncio.CancelledError)):
            await svc._run_loop()  # should not raise

        assert svc._running is True  # _run_loop itself doesn't set _running=False

    async def test_run_loop_generic_exception_continues(self, workspace: Path) -> None:
        provider = _make_provider(action="skip")
        svc = HeartbeatService(
            workspace=workspace, provider=provider, model="m", interval_s=9999
        )
        svc._running = True
        call_count = 0

        original_sleep = asyncio.sleep

        async def fake_sleep(delay: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                svc._running = False  # stop after second iteration
            await original_sleep(0)

        async def bad_tick() -> None:
            raise RuntimeError("loop error")

        svc._tick = bad_tick  # type: ignore[method-assign]

        with patch("asyncio.sleep", new=fake_sleep):
            await svc._run_loop()

        assert call_count >= 2  # loop continued past the exception

    async def test_exception_in_loop_continues(self, workspace: Path, heartbeat_file: Path) -> None:
        provider = _make_provider(action="run", tasks="task")
        call_count = 0

        async def bad_execute(tasks: str) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("oops")

        svc = HeartbeatService(
            workspace=workspace,
            provider=provider,
            model="m",
            on_execute=bad_execute,
            interval_s=9999,
        )
        # Directly call _tick to test exception handling without running the loop
        await svc._tick()
        assert call_count == 1
