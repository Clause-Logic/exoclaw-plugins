# exoclaw-turn-budget

Per-turn iteration + token budget for [exoclaw](https://github.com/Clause-Logic/exoclaw), with progressive warnings before the cutoff.

## Why

Provider weekly quotas are measured in tokens (or "prompts" that translate to tokens under the hood). A single runaway turn — say, an unbounded research loop — can consume an entire week's quota in 90 minutes by chaining hundreds of expensive `web_search` calls.

A hard `max_iterations` cap stops the loop but kills useful work mid-task with no warning. This plugin instead:

- Lets you cap **iterations**, **tokens**, or both per turn.
- Injects a `user`-role warning message at configurable thresholds (default 50%, 80%, 90%) so the agent sees it coming and can wrap up gracefully.
- Falls back to a hard cutoff at 100% if the agent ignored the warnings.
- Auto-resets at the start of every turn — no plumbing required.

## Install

```
pip install exoclaw-turn-budget
```

## Usage

The plugin has three pieces — a shared tracker, a provider wrapper that feeds it usage data + injects warnings, and a policy that owns the cutoff decision:

```python
from exoclaw import Exoclaw
from exoclaw_turn_budget import (
    BudgetTrackingProvider,
    TurnBudgetConfig,
    TurnBudgetPolicy,
    TurnBudgetTracker,
)

config = TurnBudgetConfig(
    iteration_budget=50,
    token_budget=500_000,
    warning_thresholds=(0.5, 0.8, 0.9),
)
tracker = TurnBudgetTracker(config)

app = Exoclaw(
    provider=BudgetTrackingProvider(real_provider, tracker),
    iteration_policy=TurnBudgetPolicy(tracker),
    conversation=conversation,
)
```

Both budgets are consumed simultaneously; whichever exhausts first triggers the cutoff. Set either to `None` to disable that axis.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `iteration_budget` | `50` | Hard cap on LLM iterations within one turn. `None` to disable. |
| `token_budget` | `500_000` | Hard cap on total tokens (input + output, summed across iterations). `None` to disable. |
| `warning_thresholds` | `(0.5, 0.8, 0.9)` | Fractions at which to inject a warning message. Each fires at most once per turn. |
| `warning_template` | see config | Template with `{pct}`, `{iter_used}`, `{iter_budget}`, `{token_used}`, `{token_budget}`. |
| `cutoff_template` | see config | Template for the final `on_limit_reached` message. Same substitutions. |

## How warnings reach the agent

When usage crosses a threshold, the next `chat()` call has the warning prepended as a `user`-role message. The agent sees:

```
[Turn budget notice] You've used 80% of your turn budget
(40/50 iterations, 410000/500000 tokens). Wrap up your current
line of work and respond to the user soon.
```

The warning is ephemeral — only that one model call sees it, but the agent's response (the wrap-up plan it adopts) gets persisted as normal. Each threshold fires once per turn; subsequent iterations don't repeat the same warning.

## Subagents

Each `AgentLoop` instance gets its own tracker. To budget a parent turn together with its subagent spawns, share the same `TurnBudgetTracker` and `TurnBudgetPolicy` instances when constructing the subagent loop.

## Composition with other policies

`TurnBudgetPolicy` only handles per-turn iteration/token cutoff. Pair it with [`exoclaw-loop-detection`](../exoclaw-loop-detection) when you also want pattern-based detection of degenerate tool-call loops — they don't conflict, but `exoclaw` only accepts a single `iteration_policy`. A meta-policy that delegates to both is left as an exercise.
