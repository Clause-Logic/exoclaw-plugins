"""Cron tool for scheduling reminders and tasks."""

from typing import Any

from exoclaw._compat import TaskLocal as ContextVar
from exoclaw.agent.tools.protocol import ToolBase, ToolContext

from exoclaw_tools_cron.protocol import CronBackend
from exoclaw_tools_cron.types import CronSchedule


class CronTool(ToolBase):
    """Tool to schedule reminders and recurring tasks."""

    def __init__(self, backend: CronBackend):
        self._backend = backend
        # Per-task destination via ContextVars — same pattern as
        # MessageTool / SpawnTool (see test_cron_concurrency_bugs.py).
        # A non-context call after a context-bound call would otherwise
        # inherit the prior caller's destination from instance attrs.
        self._channel_var: ContextVar[str] = ContextVar(f"cron_tool_channel_{id(self)}", default="")
        self._chat_id_var: ContextVar[str] = ContextVar(f"cron_tool_chat_id_{id(self)}", default="")
        self._in_cron_context: ContextVar[bool] = ContextVar("cron_in_context", default=False)

    @property
    def _channel(self) -> str:
        return self._channel_var.get()

    @property
    def _chat_id(self) -> str:
        return self._chat_id_var.get()

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current session context for delivery (per-task)."""
        self._channel_var.set(channel)
        self._chat_id_var.set(chat_id)

    def set_cron_context(self, active: bool) -> object:
        """Mark whether the tool is executing inside a cron job callback."""
        return self._in_cron_context.set(active)

    def reset_cron_context(self, token: object) -> None:
        """Restore previous cron context.

        ``token`` is whatever ``set_cron_context`` returned —
        ``contextvars.Token`` on CPython, ``exoclaw._compat._Token``
        on MicroPython. Both reject foreign types with
        ``TypeError``; swallow that so passing garbage is a clean
        noop on either runtime (the original implementation
        gated this with an ``isinstance(token, Token)`` check
        which doesn't work on MP — ``contextvars.Token`` doesn't
        exist there)."""
        if token is None:
            return
        try:
            self._in_cron_context.reset(token)  # type: ignore[arg-type]
        except TypeError:
            # Foreign token type (test passes a string, etc.) —
            # noop matches the prior behaviour.
            pass

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return "Schedule reminders and recurring tasks. Actions: add, list, remove, update, enable, disable."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove", "update", "enable", "disable"],
                    "description": "Action to perform",
                },
                "message": {"type": "string", "description": "Reminder message (for add)"},
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds (for recurring tasks)",
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression like '0 9 * * *' (for scheduled tasks)",
                },
                "tz": {
                    "type": "string",
                    "description": "IANA timezone for cron expressions (e.g. 'America/Vancouver')",
                },
                "at": {
                    "type": "string",
                    "description": "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00')",
                },
                "job_id": {"type": "string", "description": "Job ID (for remove/update)"},
                "deliver": {
                    "type": "boolean",
                    "description": "Whether to deliver the response to the user (for update)",
                },
                "to": {
                    "type": "string",
                    "description": "Delivery destination for this job (for update). Channel-specific address — consult the active channel skill for the format.",
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skill names to load into context when this job runs (e.g. ['email-backlog'])",
                },
                "stateless": {
                    "type": "boolean",
                    "description": "Run without session history (default false — keeps full context)",
                },
                "model": {
                    "type": "string",
                    "description": "Override the agent's default model for this job's turn (e.g. 'openrouter/google/gemma-4-26b-a4b-it'). Defaults to the agent's main model.",
                },
            },
            "required": ["action"],
        }

    async def execute_with_context(
        self,
        ctx: ToolContext,
        action: str,
        **kwargs: Any,
    ) -> str:
        self._channel_var.set(ctx.channel)
        self._chat_id_var.set(ctx.chat_id)
        return await self.execute(action=action, **kwargs)

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        deliver: bool | None = None,
        to: str | None = None,
        skills: list[str] | None = None,
        stateless: bool | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "add":
            if self._in_cron_context.get():
                return "Error: cannot schedule new jobs from within a cron job execution"
            return await self._add_job(
                message, every_seconds, cron_expr, tz, at, skills, stateless, model
            )
        elif action == "list":
            return await self._list_jobs()
        elif action == "remove":
            return await self._remove_job(job_id)
        elif action == "update":
            return await self._update_job(
                job_id, message or None, deliver, to, skills, stateless, model
            )
        elif action == "enable":
            return await self._enable_job(job_id, enabled=True)
        elif action == "disable":
            return await self._enable_job(job_id, enabled=False)
        return f"Unknown action: {action}"

    async def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
        skills: list[str] | None = None,
        stateless: bool | None = None,
        model: str | None = None,
    ) -> str:
        if not message:
            return "Error: message is required for add"
        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)"
        if tz and not cron_expr:
            return "Error: tz can only be used with cron_expr"
        if tz:
            from zoneinfo import ZoneInfo

            try:
                ZoneInfo(tz)
            except (KeyError, Exception):
                return f"Error: unknown timezone '{tz}'"

        # Build schedule
        delete_after = False
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
        elif at:
            from datetime import datetime

            try:
                dt = datetime.fromisoformat(at)
            except ValueError:
                return f"Error: invalid ISO datetime format '{at}'. Expected format: YYYY-MM-DDTHH:MM:SS"
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            return "Error: either every_seconds, cron_expr, or at is required"

        job = await self._backend.add(
            name=message[:30],
            schedule=schedule,
            message=message,
            deliver=True,
            channel=self._channel,
            to=self._chat_id,
            delete_after_run=delete_after,
            skills=skills,
            stateless=stateless or False,
            model=model,
        )
        return f"Created job '{job.name}' (id: {job.id})"

    async def _list_jobs(self) -> str:
        jobs = await self._backend.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines = [f"- {j.name} (id: {j.id}, {j.schedule.kind})" for j in jobs]
        return "Scheduled jobs:\n" + "\n".join(lines)

    async def _update_job(
        self,
        job_id: str | None,
        message: str | None,
        deliver: bool | None,
        to: str | None,
        skills: list[str] | None,
        stateless: bool | None = None,
        model: str | None = None,
    ) -> str:
        if not job_id:
            return "Error: job_id is required for update"
        job = await self._backend.update(
            job_id,
            message=message,
            deliver=deliver,
            to=to,
            skills=skills,
            stateless=stateless,
            model=model,
        )
        if job:
            return f"Updated job {job_id}"
        return f"Job {job_id} not found"

    async def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"
        if await self._backend.remove(job_id):
            return f"Removed job {job_id}"
        return f"Job {job_id} not found"

    async def _enable_job(self, job_id: str | None, *, enabled: bool) -> str:
        if not job_id:
            return "Error: job_id is required for enable/disable"
        job = await self._backend.enable(job_id, enabled=enabled)
        if job:
            state = "enabled" if enabled else "disabled"
            return f"{state.capitalize()} job {job_id}"
        return f"Job {job_id} not found"
