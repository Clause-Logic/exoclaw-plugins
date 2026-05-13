"""MicroPython smoke test for ``exoclaw-turn-budget``.

Pure-Python — no pytest. Driven by the workspace's ``mise run test-micro``
task on a coverage-variant MicroPython binary. Verifies that the runtime
branches (dual-class config, plain-class Enforcement constants,
``time.time``-based day-key) all import and behave on MP.
"""


def test_top_level_imports() -> None:
    from exoclaw_turn_budget import (
        BudgetWrapper,
        DailyBudgetConfig,
        DailyBudgetTracker,
        Enforcement,
        TurnBudgetConfig,
        TurnBudgetPolicy,
        TurnBudgetTracker,
    )

    # Constants accessible.
    assert Enforcement.OBSERVE == "observe"
    assert Enforcement.WARN == "warn"
    assert Enforcement.CUTOFF == "cutoff"
    assert Enforcement.FALLBACK == "fallback"

    # Classes are constructible.
    assert callable(BudgetWrapper)
    assert callable(TurnBudgetTracker)
    assert callable(DailyBudgetTracker)
    assert callable(TurnBudgetPolicy)
    assert callable(TurnBudgetConfig)
    assert callable(DailyBudgetConfig)


def test_turn_config_constructs_with_defaults() -> None:
    """Dual-class pattern — MP branch is hand-written ``__init__``,
    CPython is ``@dataclass``. Both must accept the same kwargs."""
    from exoclaw_turn_budget import TurnBudgetConfig

    c = TurnBudgetConfig()
    assert c.iteration_budget == 50
    assert c.token_budget == 1_500_000
    assert c.warning_thresholds == (0.5, 0.8, 0.9)
    assert c.enforcement == "cutoff"
    assert c.fallback_model is None
    # Tool-strip fields default to disabled / empty so existing deploys
    # behave identically until they opt in.
    assert c.tool_strip_threshold is None
    assert c.tool_strip_disallow == ()
    assert c.cached_token_weight == 0.1


def test_turn_config_accepts_tool_strip_overrides() -> None:
    from exoclaw_turn_budget import TurnBudgetConfig

    c = TurnBudgetConfig(
        tool_strip_threshold=0.8,
        tool_strip_disallow=("exec",),
    )
    assert c.tool_strip_threshold == 0.8
    assert c.tool_strip_disallow == ("exec",)


def test_turn_tracker_should_strip_tools() -> None:
    from exoclaw_turn_budget import TurnBudgetConfig, TurnBudgetTracker

    cfg = TurnBudgetConfig(
        iteration_budget=10,
        token_budget=None,
        warning_thresholds=(),
        tool_strip_threshold=0.5,
    )
    t = TurnBudgetTracker(cfg)
    assert t.should_strip_tools() is False
    for _ in range(5):
        t.record({"total_tokens": 0})
    assert t.should_strip_tools() is True
    assert "Tools are disabled" in t.tool_strip_message()


def test_turn_config_constructs_with_overrides() -> None:
    from exoclaw_turn_budget import Enforcement, TurnBudgetConfig

    c = TurnBudgetConfig(
        iteration_budget=10,
        token_budget=None,
        enforcement=Enforcement.FALLBACK,
        fallback_model="cheap",
    )
    assert c.iteration_budget == 10
    assert c.token_budget is None
    assert c.enforcement == "fallback"
    assert c.fallback_model == "cheap"


def test_daily_config_constructs() -> None:
    from exoclaw_turn_budget import DailyBudgetConfig

    c = DailyBudgetConfig()
    assert c.daily_budget == 35_000_000
    assert c.primary_models == ()
    assert c.enforcement == "fallback"
    assert c.reset_hour_utc == 0


def test_turn_tracker_basic_recording() -> None:
    from exoclaw_turn_budget import TurnBudgetConfig, TurnBudgetTracker

    t = TurnBudgetTracker(TurnBudgetConfig(iteration_budget=5))
    t.record({"total_tokens": 100})
    t.record({"total_tokens": 200})
    assert t.iterations_seen == 2
    assert t.total_tokens == 300
    assert t.is_at_limit() is False
    t.reset()
    assert t.iterations_seen == 0


def test_turn_tracker_token_fallback_summing() -> None:
    """``record_response`` falls back to prompt+completion when no
    ``total_tokens`` field is present — MP's ``json`` shim never adds the
    field if the upstream response omits it."""
    from exoclaw_turn_budget import TurnBudgetConfig, TurnBudgetTracker

    t = TurnBudgetTracker(TurnBudgetConfig())
    t.record({"prompt_tokens": 100, "completion_tokens": 50})
    assert t.total_tokens == 150


def test_daily_tracker_day_boundary() -> None:
    """Day-key is computed from ``time.time()`` — no datetime/timezone
    dependencies that MP doesn't ship. Use an injected fake clock."""
    from exoclaw_turn_budget import DailyBudgetConfig, DailyBudgetTracker

    clock = [1_700_000_000.0]

    def fake_clock() -> float:
        return clock[0]

    t = DailyBudgetTracker(
        DailyBudgetConfig(daily_budget=1000),
        clock=fake_clock,
    )
    t.record({"total_tokens": 500})
    assert t.total_tokens == 500
    # Advance > 24h — new day-key.
    clock[0] += 25 * 3600
    t.maybe_auto_reset()
    assert t.total_tokens == 0


def test_daily_tracker_primary_model_filter() -> None:
    from exoclaw_turn_budget import DailyBudgetConfig, DailyBudgetTracker

    t = DailyBudgetTracker(DailyBudgetConfig(daily_budget=1000, primary_models=("glm-5.1",)))
    t.record({"total_tokens": 100}, model="glm-5.1")
    t.record({"total_tokens": 999}, model="other")
    assert t.total_tokens == 100


def test_policy_resets_on_iteration_zero() -> None:
    """Async ``should_continue`` exercised by manually driving the
    coroutine — MP ships ``asyncio`` but not ``asyncio.run`` everywhere,
    so we use ``send`` directly to keep the test runtime-portable."""
    from exoclaw_turn_budget import TurnBudgetConfig, TurnBudgetPolicy, TurnBudgetTracker

    tracker = TurnBudgetTracker(TurnBudgetConfig(iteration_budget=10))
    tracker.record({"total_tokens": 999})
    policy = TurnBudgetPolicy(tracker)

    coro = policy.should_continue(0, [])
    try:
        coro.send(None)
    except StopIteration as exc:
        result = exc.value
    else:
        result = None
    assert result is True
    assert tracker.iterations_seen == 0
    assert tracker.total_tokens == 0


def test_in_memory_store_roundtrips() -> None:
    """``InMemoryBudgetStore`` is the default for the daily tracker — it
    must work on MP since chip deploys can't pull in a filesystem store."""
    from exoclaw_turn_budget import InMemoryBudgetStore

    s = InMemoryBudgetStore()
    assert s.load() is None
    s.save({"day_key": 1, "total_tokens": 42})
    loaded = s.load()
    assert loaded is not None
    assert loaded["day_key"] == 1
    assert loaded["total_tokens"] == 42
    s.clear()
    assert s.load() is None


def test_daily_tracker_persists_to_store() -> None:
    """Tracker calls ``store.save`` on every record so a chip restart
    (when paired with a durable store at the host level) can recover.
    On MP itself the in-memory store is the only viable choice — this
    test just verifies the tracker→store wiring works without crashing
    on the chip's typing shim."""
    from exoclaw_turn_budget import (
        DailyBudgetConfig,
        DailyBudgetTracker,
        InMemoryBudgetStore,
    )

    clock = [1_700_000_000.0]

    def fake_clock() -> float:
        return clock[0]

    store = InMemoryBudgetStore()
    t = DailyBudgetTracker(
        DailyBudgetConfig(daily_budget=1000),
        clock=fake_clock,
        store=store,
    )
    t.record({"total_tokens": 250})

    loaded = store.load()
    assert loaded is not None
    assert loaded["total_tokens"] == 250

    # Fresh tracker hydrates from the same store.
    recovered = DailyBudgetTracker(
        DailyBudgetConfig(daily_budget=1000),
        clock=fake_clock,
        store=store,
    )
    assert recovered.total_tokens == 250
