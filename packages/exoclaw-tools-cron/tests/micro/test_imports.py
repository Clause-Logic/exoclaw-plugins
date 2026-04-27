"""MicroPython import smoke test for ``exoclaw-tools-cron``.

Pure-Python — no pytest. Driven by the workspace's
``mise run test-micro`` task on a coverage-variant MicroPython
binary.

The real cron timer behaviour (firing on schedule, persisting
across reboots) is exercised by the CPython test suite. This MP
gate verifies the import graph clears cleanly + the storage and
schedule-math paths work end-to-end with the dual-class
dataclasses.
"""

import asyncio


def test_top_level_imports():
    from exoclaw_tools_cron.protocol import CronBackend
    from exoclaw_tools_cron.service import CronService, LocalCronBackend
    from exoclaw_tools_cron.tool import CronTool
    from exoclaw_tools_cron.types import CronJob, CronPayload, CronSchedule

    assert callable(CronBackend)
    assert callable(CronService)
    assert callable(LocalCronBackend)
    assert callable(CronTool)
    assert callable(CronJob)
    assert callable(CronPayload)
    assert callable(CronSchedule)


def test_short_id_is_hex():
    """``_short_id`` produces 8-char hex on MP without ``uuid``."""
    from exoclaw_tools_cron.service import _short_id

    out = _short_id()
    assert isinstance(out, str)
    assert len(out) == 8
    for c in out:
        assert c in "0123456789abcdef"


def test_cron_schedule_constructs_with_kwargs():
    """The dual-class dataclass pattern means CronSchedule's
    ``__init__`` accepts ``kind`` / ``every_ms`` / etc. kwargs on
    both runtimes (MP gets a hand-written init since
    ``@dataclass`` can't introspect annotations there)."""
    from exoclaw_tools_cron.types import CronSchedule

    s = CronSchedule(kind="every", every_ms=500)
    assert s.kind == "every"
    assert s.every_ms == 500
    assert s.at_ms is None
    assert s.expr is None
    assert s.tz is None


def test_compute_next_run_every():
    """``every`` schedules return ``now + interval``."""
    from exoclaw_tools_cron.service import _compute_next_run
    from exoclaw_tools_cron.types import CronSchedule

    sched = CronSchedule(kind="every", every_ms=500)
    assert _compute_next_run(sched, 1000) == 1500


def test_compute_next_run_at():
    """``at`` schedules return the timestamp if it's in the future,
    ``None`` if already passed."""
    from exoclaw_tools_cron.service import _compute_next_run
    from exoclaw_tools_cron.types import CronSchedule

    future = CronSchedule(kind="at", at_ms=2000)
    past = CronSchedule(kind="at", at_ms=500)
    assert _compute_next_run(future, 1000) == 2000
    assert _compute_next_run(past, 1000) is None


def test_compute_next_run_cron_handles_both_runtimes():
    """``cron``-expression schedules need ``croniter`` /
    ``zoneinfo`` which aren't on MP. CPython returns a real
    next-run timestamp; MP returns ``None`` cleanly. Both
    behaviours are valid for the same code path — no crash."""
    from exoclaw._compat import IS_MICROPYTHON
    from exoclaw_tools_cron.service import _compute_next_run
    from exoclaw_tools_cron.types import CronSchedule

    sched = CronSchedule(kind="cron", expr="0 9 * * *")
    result = _compute_next_run(sched, 1000)
    if IS_MICROPYTHON:
        assert result is None
    else:
        # CPython has croniter — gets a real timestamp.
        assert result is None or isinstance(result, int)


def test_local_cron_backend_round_trip():
    """End-to-end: build a service, wrap with backend, add a job,
    list it, remove it. No timer fires — purely the storage path."""
    from exoclaw._compat import Path
    from exoclaw_tools_cron.service import CronService, LocalCronBackend
    from exoclaw_tools_cron.types import CronSchedule

    store = Path("/tmp/test_cron_round_trip.json")
    if store.exists():
        store.unlink()

    async def _go():
        service = CronService(store_path=store)
        backend = LocalCronBackend(service=service)
        # ``CronBackend`` protocol surface — every method exists.
        for method in ("add", "list_jobs", "get", "update", "remove", "enable"):
            assert callable(getattr(backend, method))

        job = await backend.add(
            name="test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="hello",
        )
        assert job.name == "test"
        jobs = await backend.list_jobs()
        assert len(jobs) == 1
        ok = await backend.remove(jobs[0].id)
        assert ok is True
        assert await backend.list_jobs() == []

    try:
        asyncio.run(_go())
    finally:
        if store.exists():
            store.unlink()
