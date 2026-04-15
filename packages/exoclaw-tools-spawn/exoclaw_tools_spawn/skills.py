"""Deprecated entry-point shim — delegates to ``exoclaw_subagent.skills``.

Existing host configs that list ``exoclaw-tools-spawn`` in their
skills package list keep getting the spawn skill via this entry
point. The actual SKILL.md and loader live in ``exoclaw_subagent``.
"""

from exoclaw_subagent.skills import spawn

__all__ = ["spawn"]
