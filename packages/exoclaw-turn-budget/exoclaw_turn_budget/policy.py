"""IterationPolicy adapter for the turn-level budget.

Two responsibilities:

1. **Turn-boundary detection** — the agent loop calls ``should_continue(0, ...)``
   at the start of every turn. That's the only signal we have to know a
   new turn has started; the policy resets the paired ``TurnBudgetTracker``
   on that call.
2. **Cutoff (when ``Enforcement.CUTOFF``)** — return ``False`` from
   ``should_continue`` once the tracker reports it's at the limit, so the
   loop terminates via exoclaw's existing ``on_max_iterations`` /
   ``iteration_limit`` log path. (When the wrapper is also configured with
   ``CUTOFF``, the wrapper's synthetic response would have ended the loop
   first; the policy returning ``False`` is a defense-in-depth fallback.)

For ``OBSERVE``, ``WARN``, and ``FALLBACK`` enforcement modes the policy
never blocks — it only handles the per-turn reset.
"""

from __future__ import annotations

from exoclaw_turn_budget.enforcement import Enforcement
from exoclaw_turn_budget.tracker import TurnBudgetTracker


class TurnBudgetPolicy:
    """Pairs with a ``TurnBudgetTracker`` to bridge the agent loop's
    iteration counter into per-turn budget enforcement.
    """

    def __init__(
        self,
        tracker: TurnBudgetTracker,
        enforcement: str | None = None,
    ) -> None:
        self._tracker = tracker
        # Default to whatever the tracker's config says, so users only need
        # to set enforcement in one place.
        self._enforcement = enforcement if enforcement is not None else tracker.config.enforcement

    async def should_continue(self, iteration: int, tools_used: list[str]) -> bool:
        if iteration == 0:
            self._tracker.reset()
            return True
        if self._enforcement == Enforcement.CUTOFF and self._tracker.is_at_limit():
            return False
        return True

    async def on_limit_reached(self, iteration: int, tools_used: list[str]) -> str:
        return self._tracker.at_limit_message()
