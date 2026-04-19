"""CronBackend protocol — the execution engine seam for cron scheduling.

Implementations handle storage, scheduling, and job lifecycle. The ``CronTool``
delegates all persistence and scheduling concerns through this protocol,
allowing alternative backends (e.g. Temporal Schedules) to reuse the full
plugin feature set without reimplementing the tool.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from exoclaw_tools_cron.types import CronJob, CronSchedule


@runtime_checkable
class CronBackend(Protocol):
    """Execution engine for scheduled jobs."""

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
    ) -> CronJob: ...

    async def list_jobs(self, *, include_disabled: bool = False) -> list[CronJob]: ...

    async def get(self, job_id: str) -> CronJob | None: ...

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
    ) -> CronJob | None: ...

    async def remove(self, job_id: str) -> bool: ...

    async def enable(self, job_id: str, enabled: bool = True) -> CronJob | None: ...
