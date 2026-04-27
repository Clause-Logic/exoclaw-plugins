"""Cron types — hand-written ``__init__`` for cross-runtime parity.

These types use plain ``__init__`` rather than the ``@dataclass``
decorator. Reason: MicroPython 1.27 doesn't populate
``__annotations__`` for variable annotations, so the runtime
``@dataclass`` decorator on MP synthesises an ``__init__`` with no
fields. The dual-class pattern (``@dataclass`` on CPython, plain
class on MP) works for the runtime but creates a union of two
class objects that ``ty`` can't resolve cleanly when types nest
(``CronJob.schedule: CronSchedule | None`` cascades into 32+
``invalid-argument-type`` diagnostics).

Plain class on both runtimes side-steps that — same call surface,
no ``@dataclass`` magic, no union resolution headaches. We give
up auto-synthesised ``__repr__`` / ``__eq__`` (cron types aren't
compared for equality anywhere; identity comparison via ``id``
field is the convention in callers).
"""

from typing import Any, Literal


class CronSchedule:
    """Schedule definition for a cron job."""

    def __init__(
        self,
        kind: Literal["at", "every", "cron"],
        at_ms: int | None = None,
        every_ms: int | None = None,
        expr: str | None = None,
        tz: str | None = None,
    ) -> None:
        self.kind = kind
        # For "at": timestamp in ms.
        self.at_ms = at_ms
        # For "every": interval in ms.
        self.every_ms = every_ms
        # For "cron": cron expression (e.g. "0 9 * * *").
        self.expr = expr
        # Timezone for cron expressions.
        self.tz = tz


class CronPayload:
    """What to do when the job runs."""

    def __init__(
        self,
        kind: Literal["system_event", "agent_turn"] = "agent_turn",
        message: str = "",
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        skills: list[str] | None = None,
        stateless: bool = False,
        model: str | None = None,
    ) -> None:
        self.kind = kind
        self.message = message
        # Deliver response to channel.
        self.deliver = deliver
        self.channel = channel  # e.g. "whatsapp"
        self.to = to  # e.g. phone number
        # Skills to load into context when this job runs.
        self.skills = skills if skills is not None else []
        # Run without session history (default False — stateful, backwards compatible).
        self.stateless = stateless
        # Override the agent's default model for this job's turn (None = inherit).
        self.model = model


class CronJobState:
    """Runtime state of a job."""

    def __init__(
        self,
        next_run_at_ms: int | None = None,
        last_run_at_ms: int | None = None,
        last_status: Literal["ok", "error", "skipped"] | None = None,
        last_error: str | None = None,
    ) -> None:
        self.next_run_at_ms = next_run_at_ms
        self.last_run_at_ms = last_run_at_ms
        self.last_status: Any = last_status
        self.last_error = last_error


class CronJob:
    """A scheduled job."""

    def __init__(
        self,
        id: str,
        name: str,
        enabled: bool = True,
        schedule: CronSchedule | None = None,
        payload: CronPayload | None = None,
        state: CronJobState | None = None,
        created_at_ms: int = 0,
        updated_at_ms: int = 0,
        delete_after_run: bool = False,
    ) -> None:
        self.id = id
        self.name = name
        self.enabled = enabled
        self.schedule = schedule if schedule is not None else CronSchedule(kind="every")
        self.payload = payload if payload is not None else CronPayload()
        self.state = state if state is not None else CronJobState()
        self.created_at_ms = created_at_ms
        self.updated_at_ms = updated_at_ms
        self.delete_after_run = delete_after_run


class CronStore:
    """Persistent store for cron jobs."""

    def __init__(
        self,
        version: int = 1,
        jobs: list[CronJob] | None = None,
    ) -> None:
        self.version = version
        self.jobs = jobs if jobs is not None else []
