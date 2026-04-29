# exoclaw-turn-budget

Per-turn and per-day token + iteration budgets for [exoclaw](https://github.com/Clause-Logic/exoclaw), with progressive warnings before any cutoff and configurable enforcement actions.

## Why

Provider weekly quotas are measured in tokens (or "prompts" that translate to tokens under the hood). A single runaway turn — say, an unbounded research loop chaining hundreds of expensive `web_search` calls — can consume an entire week's quota in 90 minutes.

A hard `max_iterations` cap stops the loop but kills useful work mid-task with no warning. This plugin instead provides:

- **Turn budget** — caps iterations and tokens per *individual* turn. Warnings inject as `user`-role messages at configurable thresholds (default 50/80/90%) so the agent can wrap up gracefully. Default enforcement is `CUTOFF`.
- **Daily budget** — tracks cumulative tokens for the configured primary models (e.g., the GLM family for z.ai's pooled weekly quota), auto-resets at the UTC day boundary. Default enforcement is `FALLBACK` — silently demote to a cheaper model when the daily allotment is spent so the bot keeps working through end-of-day instead of going dark.

Both layers are built on a single shared `BudgetWrapper` primitive and share the same `Enforcement` enum: `OBSERVE`, `WARN`, `CUTOFF`, `FALLBACK`.

## Install

```
pip install exoclaw-turn-budget
```

## Usage

Stack the wrappers (turn → daily → real provider):

```python
from exoclaw import Exoclaw
from exoclaw_turn_budget import (
    BudgetWrapper,
    DailyBudgetConfig,
    DailyBudgetTracker,
    Enforcement,
    TurnBudgetConfig,
    TurnBudgetPolicy,
    TurnBudgetTracker,
)

turn_tracker = TurnBudgetTracker(TurnBudgetConfig(
    iteration_budget=50,
    token_budget=1_500_000,
    warning_thresholds=(0.5, 0.8, 0.9),
    enforcement=Enforcement.CUTOFF,
))

daily_tracker = DailyBudgetTracker(DailyBudgetConfig(
    daily_budget=35_000_000,
    primary_models=("glm-4.7", "glm-5.1"),  # what counts against the daily pool
    enforcement=Enforcement.FALLBACK,
    fallback_model="minimax/minimax-m2.7",
))

# Stack: real provider → daily layer → turn layer → AgentLoop
provider = BudgetWrapper(real_provider, daily_tracker)
provider = BudgetWrapper(provider, turn_tracker)

app = Exoclaw(
    provider=provider,
    iteration_policy=TurnBudgetPolicy(turn_tracker),  # bridges turn cutoff into the loop
    conversation=conversation,
)
```

The wrapper picks up `enforcement` and `fallback_model` from `tracker.config` automatically — no need to set them in two places.

## Turn budget configuration

| Parameter | Default | Description |
|---|---|---|
| `iteration_budget` | `50` | Hard cap on LLM iterations within one turn. `None` to disable. |
| `token_budget` | `1_500_000` | Hard cap on total tokens (input + output, summed across iterations). `None` to disable. |
| `warning_thresholds` | `(0.5, 0.8, 0.9)` | Fractions at which to inject a warning message. Each fires at most once per turn. |
| `enforcement` | `Enforcement.CUTOFF` | Action at 100% utilization. See "Enforcement modes" below. |
| `fallback_model` | `None` | Required when `enforcement == FALLBACK`. |
| `warning_template` | see config | Template for threshold warnings. Substitutions: `{scope}`, `{pct}`, `{used}`, `{cap}`, `{unit}`. |
| `cutoff_template` | see config | Template for the cutoff message used by `CUTOFF`/`WARN`. Same substitutions. |

Both budgets are consumed simultaneously — whichever exhausts first triggers enforcement. The substitution variables describe whichever axis is closer to exhaustion when the message fires (so `{used}/{cap} {unit}` reads as either `40/50 iterations` or `1200000/1500000 tokens`).

## Daily budget configuration

| Parameter | Default | Description |
|---|---|---|
| `daily_budget` | `35_000_000` | Tokens allowed per day for the primary models. Aim slightly below `weekly_quota / 7`. |
| `primary_models` | `()` | Models that count against the budget. Empty tuple means "all models count". |
| `warning_thresholds` | `()` | Fractions at which to inject a warning. Default empty (silent). |
| `enforcement` | `Enforcement.FALLBACK` | Action at 100% utilization. |
| `fallback_model` | `None` | Required when `enforcement == FALLBACK`. |
| `reset_hour_utc` | `0` | Hour of day (0–23, UTC) at which the budget rolls over. |
| `warning_template` / `cutoff_template` / `fallback_template` | see config | Same substitutions as the turn config. `fallback_template` is used at the cutoff point when `enforcement == FALLBACK`. |

Tokens spent on the `fallback_model` do *not* count against the daily budget — when `primary_models` is set, only matches deplete it.

## Enforcement modes

| Mode | Behavior at 100% utilization |
|---|---|
| `OBSERVE` | Track only — nothing happens. Useful for measurement before flipping on real enforcement. |
| `WARN` | Inject the cutoff message as a one-time `user`-role notice, then forward unchanged. Continues working past the limit. |
| `CUTOFF` | Synthesize a final response containing the cutoff message and stop. The inner provider is *not* called. |
| `FALLBACK` | Rewrite the `model` argument to `fallback_model` and forward. Tokens used on the fallback don't count against the budget. |

Threshold warnings (50/80/90%) are orthogonal — they always fire if `warning_thresholds` is non-empty, regardless of the enforcement mode chosen for the cutoff.

## How warnings reach the agent

When utilization crosses a threshold, the next `chat()` call has the warning *appended* as a synthetic `user`-role message at the end of `messages`. The agent sees:

```
[Budget notice] You've used 80% of your turn budget
(40/50 iterations). Wrap up your current line of work and
respond to the user soon.
```

The injection is ephemeral — only that one model call sees it, but the agent's response (the wrap-up plan it adopts) gets persisted as normal. Each threshold fires at most once per turn; subsequent iterations don't repeat the same warning. Threshold warnings are also suppressed once the budget is exhausted, so the agent doesn't get a stale "you're at 50%" notice after it has already hit the limit.

## Subagents

Each `AgentLoop` instance gets its own `TurnBudgetTracker`. To budget a parent turn together with its subagent spawns, share the same tracker (and policy) when constructing the subagent loop. The `DailyBudgetTracker` is naturally shared across all loops in a process since it's keyed on wall-clock time, not turn boundaries.

## Composition with other policies

`TurnBudgetPolicy` only handles turn-boundary detection and the optional cutoff. Pair it with [`exoclaw-loop-detection`](../exoclaw-loop-detection) when you also want pattern-based detection of degenerate tool-call loops — they don't conflict, but `exoclaw` only accepts a single `iteration_policy`. A meta-policy that delegates to both is left as an exercise.

## MicroPython compatibility

This package opts into the workspace's MP CI gate via `[tool.exoclaw] mp_compat = true`. The runtime branches (plain-class `Enforcement` constants instead of `enum.Enum`, dual `@dataclass` / hand-written `__init__` configs, `time.time()` day-key) are exercised by `tests/micro/test_imports.py`.
