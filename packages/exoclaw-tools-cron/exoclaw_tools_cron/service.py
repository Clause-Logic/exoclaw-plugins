"""Cron service for scheduling agent tasks.

Contains two public classes:

- ``CronService`` — the low-level engine (JSON storage + asyncio timer loop).
  Methods are sync where possible, used directly by the timer internals.

- ``LocalCronBackend`` — async wrapper that implements ``CronBackend`` protocol.
  This is the interface callers (``CronTool``) should depend on.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal

from exoclaw._compat import Path, get_logger
from exoclaw.utils import create_isolated_task

from exoclaw_tools_cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore

if TYPE_CHECKING:
    # ``datetime`` is used only by the cron-expression branch; MP
    # gets it via ``mip install datetime`` (or the firmware sim's
    # micropython-lib stage). Moved under TYPE_CHECKING so plain
    # ``every`` / ``at`` schedules don't pull it in.
    from datetime import datetime  # noqa: F401

logger = get_logger()


def _short_id() -> str:
    """8-char hex id. ``uuid`` isn't on MicroPython and isn't in
    micropython-lib; ``os.urandom`` is the cross-runtime CSPRNG
    that produces equivalent randomness."""
    return os.urandom(4).hex()


def _mtime_or_none(path: Path) -> float | None:
    """Return ``path.stat().st_mtime`` on CPython, ``None`` on
    MicroPython.

    The mtime check is cache-invalidation for external edits to
    ``cron.json`` — useful when a sidecar process (test harness,
    ops tool) modifies the file. On a chip there's only one
    process touching the file, so skipping the check is safe and
    avoids needing ``Path.stat`` on the MP shim."""
    # ``stat`` lives on ``pathlib.Path`` (CPython) but not the
    # ``exoclaw._compat`` MP ``Path`` shim. ``getattr`` keeps ty
    # off our back on the union and lets MP fall through cleanly.
    stat_fn = getattr(path, "stat", None)
    if stat_fn is None:
        return None
    try:
        return stat_fn().st_mtime
    except OSError:
        return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """Compute next run time in ms."""
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        # Next interval from now
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            from croniter import croniter

            # Use caller-provided reference time for deterministic scheduling
            base_time = now_ms / 1000
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_time, tz=tz)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except (ImportError, ValueError):
            # ``croniter`` / ``zoneinfo`` aren't available on
            # MicroPython; chip users get ``every`` and ``at``
            # schedules without bringing in the cron-expression
            # parser. Server users hit the import path normally.
            return None

    return None


def _validate_schedule_for_add(schedule: CronSchedule) -> None:
    """Validate schedule fields that would otherwise create non-runnable jobs."""
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")

    if schedule.kind == "cron" and schedule.tz:
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(schedule.tz)
        except Exception:
            raise ValueError(f"unknown timezone '{schedule.tz}'") from None


class CronService:
    """Service for managing and executing scheduled jobs.

    ``heartbeat_interval_ms``: when non-None, the service runs an
    internal periodic tick that flushes any jobs queued by
    ``wake_mode="next-heartbeat"`` (see ``CronJob.wake_mode``).
    Without a heartbeat interval, deferred jobs accumulate
    indefinitely until something else calls ``flush_deferred()``
    (e.g. an external timer, a manual flush from the agent, a
    wake-from-deep-sleep handler, etc.).

    The heartbeat is intentionally a property of the cron service
    rather than a separate plugin: in openclaw's design heartbeat
    is the clock that flushes coalesced firings, not a parallel
    file-poll service. Cron is the source of truth for *what* runs;
    heartbeat is *how aggressively* deferred work wakes the agent.
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
        heartbeat_interval_ms: int | None = None,
    ):
        self.store_path = store_path
        self.on_job = on_job
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self._store: CronStore | None = None
        self._last_mtime: float = 0.0
        self._timer_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        # Jobs whose scheduled time arrived but that opted into
        # heartbeat coalescing — held here until ``flush_deferred``.
        self._deferred: list[CronJob] = []
        self._running = False

    def _load_store(self) -> CronStore:
        """Load jobs from disk. Reloads automatically if file was modified externally."""
        if self._store and self.store_path.exists():
            mtime = _mtime_or_none(self.store_path)
            if mtime is not None and mtime != self._last_mtime:
                logger.info("cron_store_reloaded")
                self._store = None
        if self._store:
            return self._store

        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                jobs = []
                for j in data.get("jobs", []):
                    jobs.append(
                        CronJob(
                            id=j["id"],
                            name=j["name"],
                            enabled=j.get("enabled", True),
                            schedule=CronSchedule(
                                kind=j["schedule"]["kind"],
                                at_ms=j["schedule"].get("atMs"),
                                every_ms=j["schedule"].get("everyMs"),
                                expr=j["schedule"].get("expr"),
                                tz=j["schedule"].get("tz"),
                            ),
                            payload=CronPayload(
                                kind=j["payload"].get("kind", "agent_turn"),
                                message=j["payload"].get("message", ""),
                                deliver=j["payload"].get("deliver", False),
                                channel=j["payload"].get("channel"),
                                to=j["payload"].get("to"),
                                skills=j["payload"].get("skills", []),
                                stateless=j["payload"].get("stateless", False),
                                model=j["payload"].get("model"),
                            ),
                            state=CronJobState(
                                next_run_at_ms=j.get("state", {}).get("nextRunAtMs"),
                                last_run_at_ms=j.get("state", {}).get("lastRunAtMs"),
                                last_status=j.get("state", {}).get("lastStatus"),
                                last_error=j.get("state", {}).get("lastError"),
                            ),
                            created_at_ms=j.get("createdAtMs", 0),
                            updated_at_ms=j.get("updatedAtMs", 0),
                            delete_after_run=j.get("deleteAfterRun", False),
                            wake_mode=j.get("wakeMode", "now"),
                        )
                    )
                self._store = CronStore(jobs=jobs)
            except Exception as e:
                logger.warning("cron_store_load_failed", error=e)
                self._store = CronStore()
        else:
            self._store = CronStore()

        return self._store

    def _save_store(self) -> None:
        """Save jobs to disk."""
        if not self._store:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self._store.version,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "enabled": j.enabled,
                    "schedule": {
                        "kind": j.schedule.kind,
                        "atMs": j.schedule.at_ms,
                        "everyMs": j.schedule.every_ms,
                        "expr": j.schedule.expr,
                        "tz": j.schedule.tz,
                    },
                    "payload": {
                        "kind": j.payload.kind,
                        "message": j.payload.message,
                        "deliver": j.payload.deliver,
                        "channel": j.payload.channel,
                        "to": j.payload.to,
                        "skills": j.payload.skills,
                        "stateless": j.payload.stateless,
                        "model": j.payload.model,
                    },
                    "state": {
                        "nextRunAtMs": j.state.next_run_at_ms,
                        "lastRunAtMs": j.state.last_run_at_ms,
                        "lastStatus": j.state.last_status,
                        "lastError": j.state.last_error,
                    },
                    "createdAtMs": j.created_at_ms,
                    "updatedAtMs": j.updated_at_ms,
                    "deleteAfterRun": j.delete_after_run,
                    "wakeMode": j.wake_mode,
                }
                for j in self._store.jobs
            ],
        }

        # MicroPython's ``json.dumps`` doesn't accept the
        # ``indent=`` or ``ensure_ascii=`` kwargs CPython ships.
        # Drop both — readable indenting is nice for ops debugging
        # but isn't required, and the default ``ensure_ascii=True``
        # is fine for cron job storage.
        self.store_path.write_text(json.dumps(data), encoding="utf-8")
        mtime = _mtime_or_none(self.store_path)
        if mtime is not None:
            self._last_mtime = mtime

    async def start(self) -> None:
        """Start the cron service."""
        self._running = True
        self._load_store()
        self._recompute_next_runs()
        self._save_store()
        self._arm_timer()
        if self.heartbeat_interval_ms is not None:
            self._heartbeat_task = create_isolated_task(self._heartbeat_loop())
        logger.info(
            "cron_started",
            **{
                "job.count": len(self._store.jobs if self._store else []),
                "heartbeat.interval_ms": self.heartbeat_interval_ms,
            },
        )

    def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Periodic tick that flushes ``next-heartbeat`` deferred jobs.

        Runs only when ``heartbeat_interval_ms`` is set on the
        service. The chip use case wants this; servers with always-on
        agents can leave it ``None`` and never coalesce."""
        interval_s = (self.heartbeat_interval_ms or 0) / 1000
        if interval_s <= 0:
            return
        while self._running:
            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                break
            if not self._running:
                break
            try:
                fired = await self.flush_deferred()
                if fired > 0:
                    logger.info("cron_heartbeat_flush", **{"deferred.fired": fired})
            except Exception as e:
                logger.error("cron_heartbeat_flush_failed", error=e)

    def _recompute_next_runs(self) -> None:
        """Recompute next run times for all enabled jobs."""
        if not self._store:
            return
        now = _now_ms()
        for job in self._store.jobs:
            if job.enabled:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, now)

    def _get_next_wake_ms(self) -> int | None:
        """Get the earliest next run time across all jobs."""
        if not self._store:
            return None
        times = [
            j.state.next_run_at_ms for j in self._store.jobs if j.enabled and j.state.next_run_at_ms
        ]
        return min(times) if times else None

    def _arm_timer(self) -> None:
        """Schedule the next timer tick."""
        if self._timer_task:
            self._timer_task.cancel()

        next_wake = self._get_next_wake_ms()
        if not next_wake or not self._running:
            return

        delay_ms = max(0, next_wake - _now_ms())
        delay_s = delay_ms / 1000

        async def tick() -> None:
            await asyncio.sleep(delay_s)
            if self._running:
                await self._on_timer()

        # Isolate from caller contextvars — most importantly DBOS's
        # ``_dbos_context_var``. Without this, a cron added from inside
        # a ``@DBOS.workflow`` leaks the parent workflow's
        # ``DBOSContext`` (workflow_id + function_id snapshot) into
        # this task; when the timer fires, the on_job callback's
        # downstream workflow is recorded as a child step of the
        # long-finished parent and crashes with
        # ``DBOSUnexpectedStepError``.
        self._timer_task = create_isolated_task(tick())

    async def _on_timer(self) -> None:
        """Handle timer tick - run due jobs.

        Jobs with ``wake_mode="next-heartbeat"`` are queued onto
        ``self._deferred`` and their schedule advanced as if they
        had run, but the ``on_job`` callback is held back until
        ``flush_deferred()`` is called (typically from the
        heartbeat tick). Jobs with ``wake_mode="now"`` (default)
        fire the callback immediately as before.
        """
        self._load_store()
        if not self._store:
            return

        now = _now_ms()
        due_jobs = [
            j
            for j in self._store.jobs
            if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms
        ]

        for job in due_jobs:
            if job.wake_mode == "next-heartbeat":
                self._defer_job(job)
            else:
                await self._execute_job(job)

        self._save_store()
        self._arm_timer()

    def _defer_job(self, job: CronJob) -> None:
        """Queue a job onto the heartbeat-flush list and advance its
        schedule as if it had run.

        The state-advance has to happen here (not in
        ``flush_deferred``) so the next ``_on_timer`` doesn't see
        the job as still-due and re-queue it on every tick. The
        callback fires later from ``flush_deferred``."""
        self._deferred.append(job)
        now = _now_ms()
        job.state.last_run_at_ms = now
        job.updated_at_ms = now
        if job.schedule.kind == "at":
            if job.delete_after_run:
                # One-shot ``at`` jobs are removed at flush time
                # to keep the deferred reference valid; for now
                # disable so the timer doesn't re-fire.
                job.enabled = False
            else:
                job.enabled = False
            job.state.next_run_at_ms = None
        else:
            job.state.next_run_at_ms = _compute_next_run(job.schedule, now)

    async def flush_deferred(self) -> int:
        """Fire ``on_job`` for every job queued by ``next-heartbeat``.

        Returns the number of jobs fired. Safe to call when the
        deferred list is empty (no-op, returns 0). Idempotent —
        clears the queue before invoking callbacks so re-entry from
        within ``on_job`` won't see the same jobs twice.
        """
        if not self._deferred:
            return 0
        batch = self._deferred
        self._deferred = []
        for job in batch:
            start_ms = _now_ms()
            try:
                if self.on_job:
                    await self.on_job(job)
                job.state.last_status = "ok"
                job.state.last_error = None
                logger.info(
                    "cron_job_executed",
                    **{"job.name": job.name, "job.id": job.id, "wake_mode": job.wake_mode},
                )
            except Exception as e:
                job.state.last_status = "error"
                job.state.last_error = str(e)
                logger.error("cron_job_failed", **{"job.name": job.name}, error=e)
            job.state.last_run_at_ms = start_ms
            job.updated_at_ms = _now_ms()
            # Drop one-shot ``at`` + ``delete_after_run`` jobs now
            # that they've actually fired.
            if job.schedule.kind == "at" and job.delete_after_run and self._store is not None:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]
        self._save_store()
        return len(batch)

    async def _execute_job(self, job: CronJob) -> None:
        """Execute a single job."""
        start_ms = _now_ms()

        try:
            if self.on_job:
                await self.on_job(job)

            job.state.last_status = "ok"
            job.state.last_error = None
            logger.info("cron_job_executed", **{"job.name": job.name, "job.id": job.id})

        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.error("cron_job_failed", **{"job.name": job.name}, error=e)

        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = _now_ms()

        # Handle one-shot jobs
        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]  # type: ignore[union-attr]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        else:
            # Compute next run
            job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())

    # ========== Public API ==========

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """List all jobs."""
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float("inf"))

    def get_job(self, job_id: str) -> CronJob | None:
        """Get a single job by ID without sorting."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                return job
        return None

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
        skills: list[str] | None = None,
        stateless: bool = False,
        model: str | None = None,
        wake_mode: Literal["now", "next-heartbeat"] = "now",
    ) -> CronJob:
        """Add a new job. ``wake_mode`` controls whether the firing
        wakes the agent immediately (``"now"``) or coalesces into
        the next heartbeat tick (``"next-heartbeat"``)."""
        store = self._load_store()
        _validate_schedule_for_add(schedule)
        now = _now_ms()

        job = CronJob(
            id=_short_id(),
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
                skills=skills or [],
                stateless=stateless,
                model=model,
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
            wake_mode=wake_mode,
        )

        store.jobs.append(job)
        self._save_store()
        self._arm_timer()

        logger.info(
            "cron_job_added",
            **{"job.name": name, "job.id": job.id, "wake_mode": wake_mode},
        )
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        store = self._load_store()
        before = len(store.jobs)
        store.jobs = [j for j in store.jobs if j.id != job_id]
        removed = len(store.jobs) < before

        if removed:
            self._save_store()
            self._arm_timer()
            logger.info("cron_job_removed", **{"job.id": job_id})

        return removed

    def update_job(
        self,
        job_id: str,
        message: str | None = None,
        deliver: bool | None = None,
        channel: str | None = None,
        to: str | None = None,
        skills: list[str] | None = None,
        stateless: bool | None = None,
        model: str | None = None,
        schedule: CronSchedule | None = None,
        wake_mode: Literal["now", "next-heartbeat"] | None = None,
    ) -> CronJob | None:
        """Update fields on an existing job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                if message is not None:
                    job.payload.message = message
                    job.name = message[:30]
                if deliver is not None:
                    job.payload.deliver = deliver
                if channel is not None:
                    job.payload.channel = channel
                if to is not None:
                    job.payload.to = to
                if skills is not None:
                    job.payload.skills = skills
                if stateless is not None:
                    job.payload.stateless = stateless
                if model is not None:
                    job.payload.model = model
                if schedule is not None:
                    _validate_schedule_for_add(schedule)
                    job.schedule = schedule
                    job.state.next_run_at_ms = _compute_next_run(schedule, _now_ms())
                if wake_mode is not None:
                    job.wake_mode = wake_mode
                job.updated_at_ms = _now_ms()
                self._save_store()
                self._arm_timer()
                logger.info("cron_job_updated", **{"job.name": job.name, "job.id": job.id})
                return job
        return None

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable or disable a job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                job.enabled = enabled
                job.updated_at_ms = _now_ms()
                if enabled:
                    job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())
                else:
                    job.state.next_run_at_ms = None
                self._save_store()
                self._arm_timer()
                return job
        return None

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        """Manually run a job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                if not force and not job.enabled:
                    return False
                await self._execute_job(job)
                self._save_store()
                self._arm_timer()
                return True
        return False

    def status(self) -> dict[str, object]:
        """Get service status."""
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }


class LocalCronBackend:
    """Async ``CronBackend`` implementation backed by ``CronService``.

    Wraps the sync CronService methods as async coroutines so that
    ``CronTool`` can depend on the protocol without caring whether the
    backend is local (JSON file + timer) or remote (e.g. Temporal).
    """

    def __init__(self, service: CronService) -> None:
        self._svc = service

    # -- CronBackend protocol ------------------------------------------------

    async def add(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        *,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
        skills: list[str] | None = None,
        stateless: bool = False,
        model: str | None = None,
        wake_mode: Literal["now", "next-heartbeat"] = "now",
    ) -> CronJob:
        return self._svc.add_job(
            name=name,
            schedule=schedule,
            message=message,
            deliver=deliver,
            channel=channel,
            to=to,
            delete_after_run=delete_after_run,
            skills=skills,
            stateless=stateless,
            model=model,
            wake_mode=wake_mode,
        )

    async def list_jobs(self, *, include_disabled: bool = False) -> list[CronJob]:
        return self._svc.list_jobs(include_disabled=include_disabled)

    async def get(self, job_id: str) -> CronJob | None:
        return self._svc.get_job(job_id)

    async def update(
        self,
        job_id: str,
        *,
        message: str | None = None,
        schedule: CronSchedule | None = None,
        deliver: bool | None = None,
        channel: str | None = None,
        to: str | None = None,
        skills: list[str] | None = None,
        stateless: bool | None = None,
        model: str | None = None,
        wake_mode: Literal["now", "next-heartbeat"] | None = None,
    ) -> CronJob | None:
        return self._svc.update_job(
            job_id,
            message=message,
            schedule=schedule,
            deliver=deliver,
            channel=channel,
            to=to,
            skills=skills,
            stateless=stateless,
            model=model,
            wake_mode=wake_mode,
        )

    async def remove(self, job_id: str) -> bool:
        return self._svc.remove_job(job_id)

    async def enable(self, job_id: str, enabled: bool = True) -> CronJob | None:
        return self._svc.enable_job(job_id, enabled=enabled)

    async def flush_deferred(self) -> int:
        """Fire all jobs queued via ``wake_mode="next-heartbeat"``.

        Forwarded to ``CronService.flush_deferred`` for callers
        that don't have direct access to the service (e.g. host
        firmware wiring an external timer to the cron backend)."""
        return await self._svc.flush_deferred()
