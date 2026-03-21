# exoclaw-loop-detection

Pattern-based loop detection for [exoclaw](https://github.com/Clause-Logic/exoclaw) — replaces hard iteration caps with smart termination.

## Why

The default `max_iterations=40` cap kills productive runs that happen to use many tools. This plugin detects *degenerate* patterns instead:

- **Repeat** — same tool + same args called N times in a row
- **Ping-pong** — alternating between two identical calls (A→B→A→B…)
- **Circuit breaker** — absolute safety net (default 200)

Productive runs with many *different* tool calls are never interrupted.

## Install

```
pip install exoclaw-loop-detection
```

## Usage

```python
from exoclaw import Exoclaw
from exoclaw_loop_detection import LoopDetectionPolicy, LoopDetectionConfig

policy = LoopDetectionPolicy(LoopDetectionConfig(
    critical_threshold=20,      # stop after 20 identical repeats
    global_circuit_breaker=200, # absolute safety net
))

app = Exoclaw(
    provider=provider,
    conversation=conversation,
    iteration_policy=policy,
)
```

The policy records tool calls automatically when the loop invokes `should_continue`. To also track arguments for fingerprinting, call `policy.record(name, args)` from an `on_tool_calls` hook or a custom executor's `execute_tool`.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `history_size` | 30 | Tool calls retained for pattern matching |
| `warning_threshold` | 10 | (Reserved) threshold for injecting a warning |
| `critical_threshold` | 20 | Repeated calls to trigger termination |
| `global_circuit_breaker` | 200 | Absolute iteration cap |
| `detect_repeat` | True | Enable same-tool-same-args streak detection |
| `detect_ping_pong` | True | Enable alternating pattern detection |
