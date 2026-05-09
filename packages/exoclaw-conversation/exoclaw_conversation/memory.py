"""Memory system for persistent agent memory.

Two-artifact backend: ``MEMORY.md`` (long-term facts) + ``HISTORY.md``
(grep-searchable log). The store is *stateless with respect to
sessions* — it summarizes a list of messages and writes the artifacts.
Boundary advancement, sidecar persistence, and "what to summarize when"
all live in the ``ConsolidationPolicy``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from exoclaw._compat import Path, get_logger

from .helpers import ensure_dir

logger = get_logger()

if TYPE_CHECKING:
    from exoclaw.providers.protocol import LLMProvider


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log).

    Stateless with respect to sessions. ``summarize`` produces artifacts
    from a message list and returns the new history-log entry text;
    callers are responsible for tracking which messages have been
    summarized.
    """

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._provider = provider
        self._model = model

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content)

    def append_history(self, entry: str) -> None:
        # ``encoding="utf-8"`` kwarg dropped — MicroPython's ``open``
        # doesn't accept it. Text mode is always UTF-8 on both
        # runtimes, so the result is identical.
        with open(str(self.history_file), "a") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def summarize(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        """Summarize ``messages`` via the configured LLM and persist
        ``MEMORY.md`` + ``HISTORY.md`` artifacts.

        Returns the new ``history_entry`` text on success — the policy
        uses it as its rolling preamble. Returns ``None`` if the
        provider is missing, the model declines to call the tool, or
        the call fails.
        """
        if not messages:
            return ""

        lines = []
        for m in messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(
                f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}"
            )

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        if self._provider is None or self._model is None:
            logger.warning("memory_consolidation_skipped", reason="no_provider")
            return None

        try:
            response = await self._provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,  # type: ignore[arg-type]
                model=self._model,
            )

            if not response.has_tool_calls:
                logger.warning("memory_consolidation_skipped", reason="no_tool_call")
                return None

            args = response.tool_calls[0].arguments
            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            # Some providers return arguments as a list (handle edge case)
            if isinstance(args, list):
                if args and isinstance(args[0], dict):
                    args = args[0]
                else:
                    logger.warning("memory_consolidation_skipped", reason="unexpected_args_list")
                    return None
            if not isinstance(args, dict):
                logger.warning(
                    "memory_consolidation_skipped",
                    reason="unexpected_args_type",
                    **{"args.type": type(args).__name__},
                )
                return None

            history_entry: str = ""
            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry)
                self.append_history(entry)
                history_entry = entry
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update)
                if update != current_memory:
                    self.write_long_term(update)

            logger.info(
                "memory_consolidated",
                **{
                    "message.summarized": len(messages),
                    "history_entry.chars": len(history_entry),
                },
            )
            return history_entry
        except Exception:
            logger.exception("memory_consolidation_failed")
            return None
