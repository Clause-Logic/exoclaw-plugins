"""Enforcement strategy constants for budget limits.

Plain string class attributes — not ``enum.Enum`` — because MicroPython
1.27 doesn't ship the ``enum`` module. Comparison via ``==`` works
identically to an ``Enum`` so call sites are unaffected.
"""

from __future__ import annotations


class Enforcement:
    """What happens when a budget is exhausted (utilization >= 100%).

    Threshold warnings (50%, 80%, 90% — configured via ``warning_thresholds``)
    are orthogonal to enforcement and always fire if configured, regardless
    of which enforcement mode is selected.

    * ``OBSERVE``  — Track usage only. No action taken at the limit.
                     Useful for measurement before turning on real enforcement.
    * ``WARN``     — Inject the cutoff message as a warning to the model on
                     the first call past the limit, but allow the call to
                     proceed normally. Continues working past the limit.
    * ``CUTOFF``   — Synthesize a final response containing the cutoff
                     message and stop the agent loop. Subsequent calls
                     against the same exhausted budget keep returning the
                     synthetic response (so daily-level cutoff blocks the
                     bot until the day reset).
    * ``FALLBACK`` — Rewrite the ``model`` argument to the configured
                     ``fallback_model`` and forward to the underlying
                     provider. Tokens used on the fallback model do not
                     count against the budget.
    """

    OBSERVE = "observe"
    WARN = "warn"
    CUTOFF = "cutoff"
    FALLBACK = "fallback"
