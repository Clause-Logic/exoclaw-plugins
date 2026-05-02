"""Tests for exoclaw-tools-cron package."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from exoclaw_tools_cron.protocol import CronBackend
from exoclaw_tools_cron.service import (
    CronService,
    LocalCronBackend,
    _compute_next_run,
    _now_ms,
    _validate_schedule_for_add,
)
from exoclaw_tools_cron.tool import CronTool
from exoclaw_tools_cron.types import (
    CronJob,
    CronJobState,
    CronPayload,
    CronSchedule,
    CronStore,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestCronSchedule:
    def test_defaults(self) -> None:
        s = CronSchedule(kind="every")
        assert s.at_ms is None
        assert s.every_ms is None
        assert s.expr is None
        assert s.tz is None

    def test_at_schedule(self) -> None:
        s = CronSchedule(kind="at", at_ms=12345)
        assert s.kind == "at"
        assert s.at_ms == 12345

    def test_cron_schedule(self) -> None:
        s = CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC")
        assert s.expr == "0 9 * * *"
        assert s.tz == "UTC"


class TestCronPayload:
    def test_defaults(self) -> None:
        p = CronPayload()
        assert p.kind == "agent_turn"
        assert p.message == ""
        assert p.deliver is False
        assert p.channel is None
        assert p.to is None
        assert p.skills == []

    def test_custom(self) -> None:
        p = CronPayload(message="hello", deliver=True, channel="whatsapp", to="+1234")
        assert p.message == "hello"
        assert p.deliver is True


class TestCronJobState:
    def test_defaults(self) -> None:
        s = CronJobState()
        assert s.next_run_at_ms is None
        assert s.last_run_at_ms is None
        assert s.last_status is None
        assert s.last_error is None


class TestCronJob:
    def test_defaults(self) -> None:
        job = CronJob(id="abc", name="test")
        assert job.enabled is True
        assert job.delete_after_run is False
        assert job.created_at_ms == 0


class TestCronStore:
    def test_defaults(self) -> None:
        store = CronStore()
        assert store.version == 1
        assert store.jobs == []


# ---------------------------------------------------------------------------
# _now_ms
# ---------------------------------------------------------------------------


class TestNowMs:
    def test_returns_milliseconds(self) -> None:
        before = int(time.time() * 1000)
        result = _now_ms()
        after = int(time.time() * 1000)
        assert before <= result <= after


# ---------------------------------------------------------------------------
# _compute_next_run
# ---------------------------------------------------------------------------


class TestComputeNextRun:
    def test_at_future(self) -> None:
        now = _now_ms()
        result = _compute_next_run(CronSchedule(kind="at", at_ms=now + 5000), now)
        assert result == now + 5000

    def test_at_past_returns_none(self) -> None:
        now = _now_ms()
        result = _compute_next_run(CronSchedule(kind="at", at_ms=now - 1000), now)
        assert result is None

    def test_at_none_at_ms_returns_none(self) -> None:
        result = _compute_next_run(CronSchedule(kind="at"), _now_ms())
        assert result is None

    def test_every_interval(self) -> None:
        now = _now_ms()
        result = _compute_next_run(CronSchedule(kind="every", every_ms=60000), now)
        assert result == now + 60000

    def test_every_zero_returns_none(self) -> None:
        result = _compute_next_run(CronSchedule(kind="every", every_ms=0), _now_ms())
        assert result is None

    def test_every_none_returns_none(self) -> None:
        result = _compute_next_run(CronSchedule(kind="every"), _now_ms())
        assert result is None

    def test_cron_expr(self) -> None:
        now = _now_ms()
        result = _compute_next_run(CronSchedule(kind="cron", expr="* * * * *"), now)
        assert result is not None
        assert result > now

    def test_cron_with_tz(self) -> None:
        now = _now_ms()
        result = _compute_next_run(
            CronSchedule(kind="cron", expr="0 9 * * *", tz="America/New_York"), now
        )
        assert result is not None

    def test_cron_no_expr_returns_none(self) -> None:
        result = _compute_next_run(CronSchedule(kind="cron"), _now_ms())
        assert result is None

    def test_cron_invalid_expr_returns_none(self) -> None:
        result = _compute_next_run(CronSchedule(kind="cron", expr="not-a-cron"), _now_ms())
        assert result is None

    def test_unknown_kind_returns_none(self) -> None:
        s = CronSchedule(kind="at")
        s.kind = "unknown"
        result = _compute_next_run(s, _now_ms())
        assert result is None


# ---------------------------------------------------------------------------
# _validate_schedule_for_add
# ---------------------------------------------------------------------------


class TestValidateScheduleForAdd:
    def test_valid_every(self) -> None:
        _validate_schedule_for_add(CronSchedule(kind="every", every_ms=60000))

    def test_valid_cron_with_tz(self) -> None:
        _validate_schedule_for_add(CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC"))

    def test_tz_on_non_cron_raises(self) -> None:
        with pytest.raises(ValueError, match="tz can only be used with cron"):
            _validate_schedule_for_add(CronSchedule(kind="every", tz="UTC"))

    def test_invalid_tz_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown timezone"):
            _validate_schedule_for_add(CronSchedule(kind="cron", expr="0 9 * * *", tz="Not/Real"))


# ---------------------------------------------------------------------------
# CronService
# ---------------------------------------------------------------------------


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "jobs.json"


@pytest.fixture
def service(store_path: Path) -> CronService:
    return CronService(store_path=store_path)


class TestCronServiceLoadStore:
    def test_empty_store_when_no_file(self, service: CronService) -> None:
        store = service._load_store()
        assert store.jobs == []

    def test_loads_existing_file(self, store_path: Path) -> None:
        data = {
            "version": 1,
            "jobs": [
                {
                    "id": "abc123",
                    "name": "test",
                    "enabled": True,
                    "schedule": {"kind": "every", "everyMs": 60000},
                    "payload": {"kind": "agent_turn", "message": "hi", "deliver": False},
                    "state": {
                        "nextRunAtMs": None,
                        "lastRunAtMs": None,
                        "lastStatus": None,
                        "lastError": None,
                    },
                    "createdAtMs": 0,
                    "updatedAtMs": 0,
                    "deleteAfterRun": False,
                }
            ],
        }
        store_path.write_text(json.dumps(data), encoding="utf-8")
        svc = CronService(store_path=store_path)
        store = svc._load_store()
        assert len(store.jobs) == 1
        assert store.jobs[0].id == "abc123"

    def test_reloads_on_mtime_change(self, store_path: Path) -> None:
        data = {"version": 1, "jobs": []}
        store_path.write_text(json.dumps(data), encoding="utf-8")
        svc = CronService(store_path=store_path)
        svc._load_store()  # loads and caches

        # Simulate external modification by clearing cached mtime
        svc._last_mtime = 0.0
        store = svc._load_store()
        assert store is not None

    def test_corrupt_file_returns_empty(self, store_path: Path) -> None:
        store_path.write_text("not json", encoding="utf-8")
        svc = CronService(store_path=store_path)
        store = svc._load_store()
        assert store.jobs == []


class TestCronServiceSaveStore:
    def test_save_and_reload(self, service: CronService, store_path: Path) -> None:
        store = service._load_store()
        store.jobs.append(
            CronJob(
                id="x1",
                name="test",
                schedule=CronSchedule(kind="every", every_ms=5000),
                payload=CronPayload(message="hi"),
                state=CronJobState(),
            )
        )
        service._save_store()
        assert store_path.exists()
        data = json.loads(store_path.read_text())
        assert data["jobs"][0]["id"] == "x1"

    def test_save_noop_when_no_store(self, store_path: Path) -> None:
        svc = CronService(store_path=store_path)
        svc._save_store()  # should not raise


class TestCronServicePublicApi:
    def test_add_job_every(self, service: CronService, store_path: Path) -> None:
        job = service.add_job(
            name="ping",
            schedule=CronSchedule(kind="every", every_ms=30000),
            message="ping",
        )
        assert job.id
        assert job.name == "ping"
        assert store_path.exists()

    def test_get_job(self, service: CronService) -> None:
        job = service.add_job(
            name="find-me",
            schedule=CronSchedule(kind="every", every_ms=1000),
            message="find",
        )
        found = service.get_job(job.id)
        assert found is not None
        assert found.id == job.id
        assert service.get_job("nonexistent") is None

    def test_add_job_at(self, service: CronService) -> None:
        future_ms = _now_ms() + 60000
        job = service.add_job(
            name="once",
            schedule=CronSchedule(kind="at", at_ms=future_ms),
            message="once",
            delete_after_run=True,
        )
        assert job.delete_after_run is True

    def test_add_job_cron(self, service: CronService) -> None:
        job = service.add_job(
            name="daily",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *"),
            message="daily",
        )
        assert job.schedule.kind == "cron"

    def test_add_job_invalid_tz_raises(self, service: CronService) -> None:
        with pytest.raises(ValueError):
            service.add_job(
                name="bad",
                schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="Bad/Zone"),
                message="bad",
            )

    def test_list_jobs_enabled_only(self, service: CronService) -> None:
        service.add_job(name="a", schedule=CronSchedule(kind="every", every_ms=1000), message="a")
        jobs = service.list_jobs()
        assert len(jobs) == 1

    def test_list_jobs_include_disabled(self, service: CronService) -> None:
        job = service.add_job(
            name="a", schedule=CronSchedule(kind="every", every_ms=1000), message="a"
        )
        service.enable_job(job.id, enabled=False)
        assert service.list_jobs() == []
        assert len(service.list_jobs(include_disabled=True)) == 1

    def test_remove_job(self, service: CronService) -> None:
        job = service.add_job(
            name="rm", schedule=CronSchedule(kind="every", every_ms=1000), message="rm"
        )
        assert service.remove_job(job.id) is True
        assert service.list_jobs() == []

    def test_remove_nonexistent_returns_false(self, service: CronService) -> None:
        assert service.remove_job("nope") is False

    def test_update_job_message(self, service: CronService) -> None:
        job = service.add_job(
            name="upd", schedule=CronSchedule(kind="every", every_ms=1000), message="old"
        )
        updated = service.update_job(job.id, message="new message")
        assert updated is not None
        assert updated.payload.message == "new message"
        assert updated.name == "new message"

    def test_update_job_schedule(self, service: CronService) -> None:
        job = service.add_job(
            name="upd", schedule=CronSchedule(kind="every", every_ms=1000), message="x"
        )
        updated = service.update_job(job.id, schedule=CronSchedule(kind="every", every_ms=5000))
        assert updated is not None
        assert updated.schedule.every_ms == 5000

    def test_update_job_deliver_and_skills(self, service: CronService) -> None:
        job = service.add_job(
            name="u", schedule=CronSchedule(kind="every", every_ms=1000), message="u"
        )
        updated = service.update_job(job.id, deliver=True, skills=["foo"])
        assert updated is not None
        assert updated.payload.deliver is True
        assert updated.payload.skills == ["foo"]

    def test_update_nonexistent_returns_none(self, service: CronService) -> None:
        assert service.update_job("nope") is None

    def test_enable_job(self, service: CronService) -> None:
        job = service.add_job(
            name="e", schedule=CronSchedule(kind="every", every_ms=1000), message="e"
        )
        service.enable_job(job.id, enabled=False)
        assert service.list_jobs() == []
        service.enable_job(job.id, enabled=True)
        assert len(service.list_jobs()) == 1

    def test_enable_nonexistent_returns_none(self, service: CronService) -> None:
        assert service.enable_job("nope") is None

    def test_status(self, service: CronService) -> None:
        s = service.status()
        assert "enabled" in s
        assert "jobs" in s

    async def test_run_job(self, service: CronService) -> None:
        called: list[str] = []

        async def on_job(job: CronJob) -> None:
            called.append(job.id)

        svc = CronService(store_path=service.store_path, on_job=on_job)
        job = svc.add_job(name="r", schedule=CronSchedule(kind="every", every_ms=1000), message="r")
        result = await svc.run_job(job.id)
        assert result is True
        assert job.id in called

    async def test_run_job_disabled_returns_false(self, service: CronService) -> None:
        job = service.add_job(
            name="d", schedule=CronSchedule(kind="every", every_ms=1000), message="d"
        )
        service.enable_job(job.id, enabled=False)
        result = await service.run_job(job.id)
        assert result is False

    async def test_run_job_force_runs_disabled(self, service: CronService) -> None:
        called: list[str] = []

        async def on_job(job: CronJob) -> None:
            called.append(job.id)

        svc = CronService(store_path=service.store_path, on_job=on_job)
        job = svc.add_job(name="f", schedule=CronSchedule(kind="every", every_ms=1000), message="f")
        svc.enable_job(job.id, enabled=False)
        result = await svc.run_job(job.id, force=True)
        assert result is True

    async def test_run_nonexistent_returns_false(self, service: CronService) -> None:
        assert await service.run_job("nope") is False


class TestCronServiceTimer:
    async def test_start_stop(self, service: CronService) -> None:
        await service.start()
        assert service._running is True
        service.stop()
        assert service._running is False

    async def test_start_with_jobs_arms_timer(self, service: CronService) -> None:
        service.add_job(name="t", schedule=CronSchedule(kind="every", every_ms=60000), message="t")
        await service.start()
        assert service._timer_task is not None
        service.stop()

    async def test_on_timer_runs_due_jobs(self, store_path: Path) -> None:
        called: list[str] = []

        async def on_job(job: CronJob) -> None:
            called.append(job.id)

        svc = CronService(store_path=store_path, on_job=on_job)
        # Add job with next_run_at_ms in the past
        store = svc._load_store()
        now = _now_ms()
        job = CronJob(
            id="due1",
            name="due",
            enabled=True,
            schedule=CronSchedule(kind="every", every_ms=60000),
            payload=CronPayload(message="due"),
            state=CronJobState(next_run_at_ms=now - 1000),
        )
        store.jobs.append(job)
        svc._running = True
        await svc._on_timer()
        assert "due1" in called

    async def test_execute_job_error_recorded(self, store_path: Path) -> None:
        async def on_job(job: CronJob) -> None:
            raise RuntimeError("boom")

        svc = CronService(store_path=store_path, on_job=on_job)
        job = svc.add_job(
            name="err", schedule=CronSchedule(kind="every", every_ms=1000), message="err"
        )
        await svc._execute_job(job)
        assert job.state.last_status == "error"
        assert "boom" in (job.state.last_error or "")

    async def test_execute_at_job_deletes_after_run(self, store_path: Path) -> None:
        svc = CronService(store_path=store_path)
        future = _now_ms() + 1000
        job = svc.add_job(
            name="once",
            schedule=CronSchedule(kind="at", at_ms=future),
            message="once",
            delete_after_run=True,
        )
        # Force execute
        svc._store = svc._load_store()  # ensure store loaded
        await svc._execute_job(job)
        # Job removed from store
        assert not any(j.id == job.id for j in (svc._store.jobs if svc._store else []))

    async def test_execute_at_job_disables_when_not_delete(self, store_path: Path) -> None:
        svc = CronService(store_path=store_path)
        future = _now_ms() + 1000
        job = svc.add_job(
            name="once",
            schedule=CronSchedule(kind="at", at_ms=future),
            message="once",
            delete_after_run=False,
        )
        await svc._execute_job(job)
        assert job.enabled is False
        assert job.state.next_run_at_ms is None

    async def test_get_next_wake_ms_no_jobs(self, service: CronService) -> None:
        service._load_store()
        assert service._get_next_wake_ms() is None

    async def test_arm_timer_no_wake(self, service: CronService) -> None:
        service._running = True
        service._load_store()
        service._arm_timer()  # no jobs → no timer
        assert service._timer_task is None

    async def test_on_timer_no_store(self, service: CronService) -> None:
        service._running = True
        service._store = None
        # Should not raise even if store is None initially — _load_store creates empty
        await service._on_timer()

    async def test_running_service_honors_external_disable(self, tmp_path: Path) -> None:
        """A second CronService instance disabling a job externally is respected by the runner."""
        store_path = tmp_path / "cron" / "jobs.json"
        called: list[str] = []

        async def on_job(job: CronJob) -> None:
            called.append(job.id)

        service = CronService(store_path=store_path, on_job=on_job)
        job = service.add_job(
            name="external-disable",
            schedule=CronSchedule(kind="every", every_ms=200),
            message="hello",
        )
        await service.start()
        try:
            await asyncio.sleep(0.05)  # ensure mtime will differ
            external = CronService(store_path=store_path)
            updated = external.enable_job(job.id, enabled=False)
            assert updated is not None
            assert updated.enabled is False

            await asyncio.sleep(0.35)
            assert called == []
        finally:
            service.stop()


# ---------------------------------------------------------------------------
# CronTool
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service(tmp_path: Path) -> CronService:
    return CronService(store_path=tmp_path / "jobs.json")


@pytest.fixture
def backend(mock_service: CronService) -> LocalCronBackend:
    return LocalCronBackend(mock_service)


@pytest.fixture
def tool(backend: LocalCronBackend) -> CronTool:
    t = CronTool(backend=backend)
    t.set_context("cli", "user1")
    return t


class TestCronToolProperties:
    def test_name(self, tool: CronTool) -> None:
        assert tool.name == "cron"

    def test_description(self, tool: CronTool) -> None:
        assert "Schedule" in tool.description

    def test_parameters_schema(self, tool: CronTool) -> None:
        params = tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "required" in params


class TestCronToolExecuteAdd:
    async def test_add_every(self, tool: CronTool) -> None:
        result = await tool.execute(action="add", message="ping", every_seconds=60)
        assert "Created job" in result

    async def test_add_cron_expr(self, tool: CronTool) -> None:
        result = await tool.execute(action="add", message="daily", cron_expr="0 9 * * *", tz="UTC")
        assert "Created job" in result

    async def test_add_at(self, tool: CronTool) -> None:
        result = await tool.execute(action="add", message="once", at="2099-01-01T10:00:00")
        assert "Created job" in result

    async def test_add_no_message_error(self, tool: CronTool) -> None:
        result = await tool.execute(action="add")
        assert "Error" in result

    async def test_add_no_schedule_error(self, tool: CronTool) -> None:
        result = await tool.execute(action="add", message="hi")
        assert "Error" in result

    async def test_add_no_context_error(self, tmp_path: Path) -> None:
        svc = CronService(store_path=tmp_path / "j.json")
        t = CronTool(backend=LocalCronBackend(svc))
        # No set_context
        result = await t.execute(action="add", message="hi", every_seconds=60)
        assert "Error" in result

    async def test_add_tz_without_cron_error(self, tool: CronTool) -> None:
        result = await tool.execute(action="add", message="hi", every_seconds=60, tz="UTC")
        assert "Error" in result

    async def test_add_invalid_tz_error(self, tool: CronTool) -> None:
        result = await tool.execute(
            action="add", message="hi", cron_expr="0 9 * * *", tz="Not/Real"
        )
        assert "Error" in result

    async def test_add_invalid_at_format(self, tool: CronTool) -> None:
        result = await tool.execute(action="add", message="hi", at="not-a-date")
        assert "Error" in result

    async def test_add_blocked_in_cron_context(self, tool: CronTool) -> None:
        token = tool.set_cron_context(True)
        try:
            result = await tool.execute(action="add", message="hi", every_seconds=60)
            assert "Error" in result
        finally:
            tool.reset_cron_context(token)

    async def test_add_with_skills(self, tool: CronTool) -> None:
        result = await tool.execute(
            action="add", message="task", every_seconds=60, skills=["foo", "bar"]
        )
        assert "Created job" in result


class TestCronToolExecuteList:
    async def test_list_empty(self, tool: CronTool) -> None:
        result = await tool.execute(action="list")
        assert "No scheduled jobs" in result

    async def test_list_with_jobs(self, tool: CronTool) -> None:
        await tool.execute(action="add", message="ping", every_seconds=60)
        result = await tool.execute(action="list")
        assert "ping" in result


class TestCronToolExecuteRemove:
    async def test_remove_existing(self, tool: CronTool) -> None:
        await tool.execute(action="add", message="rm", every_seconds=60)
        jobs = await tool._backend.list_jobs()
        result = await tool.execute(action="remove", job_id=jobs[0].id)
        assert "Removed" in result

    async def test_remove_nonexistent(self, tool: CronTool) -> None:
        result = await tool.execute(action="remove", job_id="nope")
        assert "not found" in result

    async def test_remove_no_job_id(self, tool: CronTool) -> None:
        result = await tool.execute(action="remove")
        assert "Error" in result


class TestCronToolExecuteUpdate:
    async def test_update_message(self, tool: CronTool) -> None:
        await tool.execute(action="add", message="old", every_seconds=60)
        jobs = await tool._backend.list_jobs()
        result = await tool.execute(action="update", job_id=jobs[0].id, message="new msg")
        assert "Updated" in result

    async def test_update_nonexistent(self, tool: CronTool) -> None:
        result = await tool.execute(action="update", job_id="nope")
        assert "not found" in result

    async def test_update_no_job_id(self, tool: CronTool) -> None:
        result = await tool.execute(action="update")
        assert "Error" in result

    async def test_update_cron_expr(self, tool: CronTool) -> None:
        await tool.execute(action="add", message="old", every_seconds=60)
        jobs = await tool._backend.list_jobs()
        result = await tool.execute(
            action="update", job_id=jobs[0].id, cron_expr="0 * * * *", tz="America/New_York"
        )
        assert "Updated" in result
        updated = await tool._backend.get(jobs[0].id)
        assert updated is not None
        assert updated.schedule.kind == "cron"
        assert updated.schedule.expr == "0 * * * *"
        assert updated.schedule.tz == "America/New_York"

    async def test_update_cron_expr_invalid_tz(self, tool: CronTool) -> None:
        await tool.execute(action="add", message="old", every_seconds=60)
        jobs = await tool._backend.list_jobs()
        result = await tool.execute(
            action="update", job_id=jobs[0].id, cron_expr="0 * * * *", tz="Not/Real"
        )
        assert "Error" in result
        assert "unknown timezone" in result

    async def test_update_tz_without_cron_expr(self, tool: CronTool) -> None:
        await tool.execute(action="add", message="old", every_seconds=60)
        jobs = await tool._backend.list_jobs()
        result = await tool.execute(action="update", job_id=jobs[0].id, tz="UTC")
        assert "Error" in result
        assert "tz can only be used with cron_expr" in result


class TestCronToolExecuteEnable:
    async def test_enable_existing(self, tool: CronTool) -> None:
        await tool.execute(action="add", message="en", every_seconds=60)
        jobs = await tool._backend.list_jobs()
        result = await tool.execute(action="disable", job_id=jobs[0].id)
        assert "Disabled" in result
        result = await tool.execute(action="enable", job_id=jobs[0].id)
        assert "Enabled" in result

    async def test_enable_nonexistent(self, tool: CronTool) -> None:
        result = await tool.execute(action="enable", job_id="nope")
        assert "not found" in result

    async def test_enable_no_job_id(self, tool: CronTool) -> None:
        result = await tool.execute(action="enable")
        assert "Error" in result


class TestCronToolUnknownAction:
    async def test_unknown_action(self, tool: CronTool) -> None:
        result = await tool.execute(action="frobnicate")
        assert "Unknown action" in result


class TestCronServiceInternalEdgeCases:
    def test_recompute_next_runs_no_store(self, service: CronService) -> None:
        service._store = None
        service._recompute_next_runs()  # should not raise

    def test_get_next_wake_ms_no_store(self, service: CronService) -> None:
        service._store = None
        assert service._get_next_wake_ms() is None

    async def test_arm_timer_cancels_existing_task(self, service: CronService) -> None:
        # Plant a fake timer task
        task = MagicMock()
        service._timer_task = task
        service._running = True
        service._load_store()
        service._arm_timer()  # no jobs → calls cancel on old task and returns
        task.cancel.assert_called_once()

    def test_update_job_channel_and_to(self, service: CronService) -> None:
        job = service.add_job(
            name="u", schedule=CronSchedule(kind="every", every_ms=1000), message="u"
        )
        updated = service.update_job(job.id, channel="sms", to="+1234567890")
        assert updated is not None
        assert updated.payload.channel == "sms"
        assert updated.payload.to == "+1234567890"


class TestCronToolCronContext:
    def test_set_and_reset_cron_context(self, tool: CronTool) -> None:
        assert tool._in_cron_context.get() is False
        token = tool.set_cron_context(True)
        assert tool._in_cron_context.get() is True
        tool.reset_cron_context(token)
        assert tool._in_cron_context.get() is False

    def test_reset_with_non_token_noop(self, tool: CronTool) -> None:
        tool.reset_cron_context("not-a-token")  # should not raise


# ---------------------------------------------------------------------------
# CronBackend protocol
# ---------------------------------------------------------------------------


class TestCronBackendProtocol:
    def test_local_backend_satisfies_protocol(self, backend: LocalCronBackend) -> None:
        assert isinstance(backend, CronBackend)

    async def test_backend_add_and_list(self, backend: LocalCronBackend) -> None:
        job = await backend.add(
            name="test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="hello",
        )
        assert job.id
        assert job.payload.message == "hello"
        jobs = await backend.list_jobs()
        assert len(jobs) == 1

    async def test_backend_get(self, backend: LocalCronBackend) -> None:
        job = await backend.add(
            name="get-me",
            schedule=CronSchedule(kind="every", every_ms=1000),
            message="get",
        )
        found = await backend.get(job.id)
        assert found is not None
        assert found.id == job.id
        assert await backend.get("nonexistent") is None

    async def test_backend_update(self, backend: LocalCronBackend) -> None:
        job = await backend.add(
            name="upd",
            schedule=CronSchedule(kind="every", every_ms=1000),
            message="old",
        )
        updated = await backend.update(job.id, message="new", skills=["s1"])
        assert updated is not None
        assert updated.payload.message == "new"
        assert updated.payload.skills == ["s1"]

    async def test_backend_remove(self, backend: LocalCronBackend) -> None:
        job = await backend.add(
            name="rm",
            schedule=CronSchedule(kind="every", every_ms=1000),
            message="rm",
        )
        assert await backend.remove(job.id) is True
        assert await backend.remove("nope") is False

    async def test_backend_enable(self, backend: LocalCronBackend) -> None:
        job = await backend.add(
            name="en",
            schedule=CronSchedule(kind="every", every_ms=1000),
            message="en",
        )
        disabled = await backend.enable(job.id, enabled=False)
        assert disabled is not None
        assert disabled.enabled is False
        assert await backend.list_jobs() == []


# ---------------------------------------------------------------------------
# wake_mode + flush_deferred (heartbeat coalescing)
# ---------------------------------------------------------------------------


class TestWakeModePersistence:
    def test_default_wake_mode_is_now(self, service: CronService) -> None:
        job = service.add_job(
            name="default",
            schedule=CronSchedule(kind="every", every_ms=1000),
            message="hi",
        )
        assert job.wake_mode == "now"

    def test_wake_mode_round_trips_through_disk(
        self, service: CronService, store_path: Path
    ) -> None:
        service.add_job(
            name="batched",
            schedule=CronSchedule(kind="every", every_ms=5000),
            message="hi",
            wake_mode="next-heartbeat",
        )
        # Re-read from disk via a fresh service.
        fresh = CronService(store_path=store_path)
        store = fresh._load_store()
        assert len(store.jobs) == 1
        assert store.jobs[0].wake_mode == "next-heartbeat"

    def test_wake_mode_missing_in_legacy_json_defaults_now(self, store_path: Path) -> None:
        """A pre-0.9 cron.json without ``wakeMode`` keys should
        load every job as ``wake_mode="now"`` so existing chips
        upgrade transparently."""
        legacy = {
            "version": 1,
            "jobs": [
                {
                    "id": "old1",
                    "name": "legacy",
                    "enabled": True,
                    "schedule": {"kind": "every", "everyMs": 1000},
                    "payload": {"kind": "agent_turn", "message": "hi", "deliver": False},
                    "state": {"nextRunAtMs": None},
                    "createdAtMs": 0,
                    "updatedAtMs": 0,
                    "deleteAfterRun": False,
                    # Note: no "wakeMode" key.
                }
            ],
        }
        store_path.write_text(json.dumps(legacy), encoding="utf-8")
        svc = CronService(store_path=store_path)
        store = svc._load_store()
        assert store.jobs[0].wake_mode == "now"


class TestFlushDeferred:
    @pytest.mark.asyncio
    async def test_flush_empty_returns_zero(self, service: CronService) -> None:
        assert await service.flush_deferred() == 0

    @pytest.mark.asyncio
    async def test_due_now_job_fires_immediately(self, store_path: Path) -> None:
        """``wake_mode="now"`` is unchanged behaviour — fired
        from ``_on_timer``, never queued."""
        fired: list[str] = []

        async def on_job(job: CronJob) -> str | None:
            fired.append(job.id)
            return None

        svc = CronService(store_path=store_path, on_job=on_job)
        # Insert a job already past-due so _on_timer picks it up.
        store = svc._load_store()
        now = _now_ms()
        store.jobs.append(
            CronJob(
                id="j1",
                name="now-job",
                schedule=CronSchedule(kind="every", every_ms=1000),
                state=CronJobState(next_run_at_ms=now - 100),
                wake_mode="now",
            )
        )
        await svc._on_timer()
        assert fired == ["j1"]
        assert svc._deferred == []

    @pytest.mark.asyncio
    async def test_due_next_heartbeat_job_is_deferred_not_fired(self, store_path: Path) -> None:
        """``wake_mode="next-heartbeat"`` queues the job onto the
        deferred list; ``on_job`` doesn't fire from ``_on_timer``."""
        fired: list[str] = []

        async def on_job(job: CronJob) -> str | None:
            fired.append(job.id)
            return None

        svc = CronService(store_path=store_path, on_job=on_job)
        store = svc._load_store()
        now = _now_ms()
        store.jobs.append(
            CronJob(
                id="j2",
                name="batched-job",
                schedule=CronSchedule(kind="every", every_ms=1000),
                state=CronJobState(next_run_at_ms=now - 100),
                wake_mode="next-heartbeat",
            )
        )
        await svc._on_timer()
        assert fired == []
        assert len(svc._deferred) == 1
        assert svc._deferred[0].id == "j2"

    @pytest.mark.asyncio
    async def test_flush_deferred_fires_queued_jobs(self, store_path: Path) -> None:
        fired: list[str] = []

        async def on_job(job: CronJob) -> str | None:
            fired.append(job.id)
            return None

        svc = CronService(store_path=store_path, on_job=on_job)
        store = svc._load_store()
        now = _now_ms()
        for jid in ("a", "b", "c"):
            store.jobs.append(
                CronJob(
                    id=jid,
                    name=jid,
                    schedule=CronSchedule(kind="every", every_ms=1000),
                    state=CronJobState(next_run_at_ms=now - 100),
                    wake_mode="next-heartbeat",
                )
            )
        await svc._on_timer()
        assert fired == []
        assert len(svc._deferred) == 3

        count = await svc.flush_deferred()
        assert count == 3
        assert fired == ["a", "b", "c"]
        assert svc._deferred == []

    @pytest.mark.asyncio
    async def test_deferred_advances_schedule(self, store_path: Path) -> None:
        """A deferred job's ``next_run_at_ms`` must advance even
        though the callback didn't fire — otherwise ``_on_timer``
        would re-queue it on every tick."""
        svc = CronService(store_path=store_path, on_job=None)
        store = svc._load_store()
        now = _now_ms()
        original_next = now - 100
        store.jobs.append(
            CronJob(
                id="reschedule",
                name="reschedule",
                schedule=CronSchedule(kind="every", every_ms=10_000),
                state=CronJobState(next_run_at_ms=original_next),
                wake_mode="next-heartbeat",
            )
        )
        await svc._on_timer()
        new_next = svc._store.jobs[0].state.next_run_at_ms  # type: ignore[union-attr]
        assert new_next is not None
        assert new_next > original_next

    @pytest.mark.asyncio
    async def test_flush_deferred_clears_before_callback(self, store_path: Path) -> None:
        """If ``on_job`` re-enters the service somehow, it must
        not see the same deferred jobs again."""
        seen_after_first_call: list[CronJob] = []

        async def on_job(job: CronJob) -> str | None:
            # Snapshot deferred queue from inside the callback.
            seen_after_first_call.append(job)
            return None

        svc = CronService(store_path=store_path, on_job=on_job)
        svc._deferred = [
            CronJob(
                id="x",
                name="x",
                schedule=CronSchedule(kind="every", every_ms=1000),
                state=CronJobState(),
                wake_mode="next-heartbeat",
            )
        ]
        await svc.flush_deferred()
        # Deferred list cleared synchronously before any callback ran.
        assert svc._deferred == []
        assert len(seen_after_first_call) == 1

    @pytest.mark.asyncio
    async def test_flush_deferred_at_one_shot_with_delete_after_run_drops_job(
        self, store_path: Path
    ) -> None:
        """``at`` + ``delete_after_run`` jobs are removed from the
        store at flush time so they don't linger as disabled
        records forever."""

        async def on_job(job: CronJob) -> str | None:
            return None

        svc = CronService(store_path=store_path, on_job=on_job)
        store = svc._load_store()
        now = _now_ms()
        store.jobs.append(
            CronJob(
                id="oneshot",
                name="oneshot",
                schedule=CronSchedule(kind="at", at_ms=now - 100),
                state=CronJobState(next_run_at_ms=now - 100),
                wake_mode="next-heartbeat",
                delete_after_run=True,
            )
        )
        await svc._on_timer()
        assert len(svc._deferred) == 1
        await svc.flush_deferred()
        # Job removed from store; ``at`` + ``delete_after_run`` is one-shot.
        assert svc._store.jobs == []  # type: ignore[union-attr]


class TestHeartbeatTick:
    @pytest.mark.asyncio
    async def test_heartbeat_loop_flushes_deferred(self, store_path: Path) -> None:
        """The internal heartbeat loop, when configured, should
        call ``flush_deferred`` on its own cadence."""
        fired: list[str] = []

        async def on_job(job: CronJob) -> str | None:
            fired.append(job.id)
            return None

        # 50ms heartbeat — short enough for the test to observe.
        svc = CronService(
            store_path=store_path,
            on_job=on_job,
            heartbeat_interval_ms=50,
        )
        # Pre-populate deferred queue and start the service.
        svc._deferred = [
            CronJob(
                id="hb1",
                name="hb1",
                schedule=CronSchedule(kind="every", every_ms=1000),
                state=CronJobState(),
                wake_mode="next-heartbeat",
            )
        ]
        await svc.start()
        try:
            # Wait long enough for one heartbeat tick.
            await asyncio.sleep(0.15)
        finally:
            svc.stop()
            # Allow cancellation to settle.
            await asyncio.sleep(0)
        assert "hb1" in fired

    @pytest.mark.asyncio
    async def test_no_heartbeat_interval_means_no_internal_flushing(self, store_path: Path) -> None:
        """Without ``heartbeat_interval_ms``, deferred jobs sit
        forever until something else flushes — server use case."""
        fired: list[str] = []

        async def on_job(job: CronJob) -> str | None:
            fired.append(job.id)
            return None

        svc = CronService(store_path=store_path, on_job=on_job)  # no heartbeat
        svc._deferred = [
            CronJob(
                id="never",
                name="never",
                schedule=CronSchedule(kind="every", every_ms=1000),
                state=CronJobState(),
                wake_mode="next-heartbeat",
            )
        ]
        await svc.start()
        try:
            await asyncio.sleep(0.1)
        finally:
            svc.stop()
            await asyncio.sleep(0)
        assert fired == []
        assert len(svc._deferred) == 1


class TestCronToolWakeMode:
    @pytest.mark.asyncio
    async def test_tool_passes_wake_mode_to_backend(
        self, backend: LocalCronBackend, service: CronService
    ) -> None:
        tool = CronTool(backend=backend)
        tool.set_context(channel="serial", chat_id="default")
        result = await tool.execute(
            action="add",
            message="batched daily summary",
            every_seconds=86400,
            wake_mode="next-heartbeat",
        )
        assert "Created job" in result
        jobs = service.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].wake_mode == "next-heartbeat"

    @pytest.mark.asyncio
    async def test_tool_rejects_unknown_wake_mode(self, backend: LocalCronBackend) -> None:
        tool = CronTool(backend=backend)
        tool.set_context(channel="serial", chat_id="default")
        result = await tool.execute(
            action="add",
            message="hi",
            every_seconds=60,
            wake_mode="urgent",  # not valid
        )
        assert result.startswith("Error")
        assert "wake_mode" in result

    @pytest.mark.asyncio
    async def test_tool_default_is_now_for_backwards_compat(
        self, backend: LocalCronBackend, service: CronService
    ) -> None:
        tool = CronTool(backend=backend)
        tool.set_context(channel="serial", chat_id="default")
        await tool.execute(
            action="add",
            message="urgent",
            every_seconds=60,
        )
        jobs = service.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].wake_mode == "now"
        assert len(await backend.list_jobs(include_disabled=True)) == 1
