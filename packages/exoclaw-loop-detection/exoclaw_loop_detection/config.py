"""Configuration for loop detection thresholds and detectors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopDetectionConfig:
    """Tuneable knobs for the loop detection policy.

    Attributes:
        history_size: Maximum number of tool calls retained for pattern matching.
        warning_threshold: After this many repeated identical calls, a warning
            is injected as a tool result so the LLM can self-correct.
        critical_threshold: After this many repeated identical calls (or
            ping-pong cycles), the loop is terminated.
        global_circuit_breaker: Absolute iteration cap — safety net regardless
            of whether a pattern is detected.
        detect_repeat: Enable same-tool-same-args streak detection.
        detect_ping_pong: Enable alternating A-B-A-B pattern detection.
    """

    history_size: int = 30
    warning_threshold: int = 10
    critical_threshold: int = 20
    global_circuit_breaker: int = 200
    detect_repeat: bool = True
    detect_ping_pong: bool = True
