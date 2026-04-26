"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from .helpers import ensure_dir

logger = structlog.get_logger()

if TYPE_CHECKING:
    from exoclaw.providers.protocol import LLMProvider

    from .protocols import HistoryStore
    from .session.manager import Session


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
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider | None = None,
        model: str | None = None,
        history: "HistoryStore | None" = None,
    ):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._provider = provider
        self._model = model
        # Optional HistoryStore reference. Used by consolidate() and the
        # consolidate_messages boundary-repair pass to read messages from
        # disk when ``session.messages`` is empty (streaming_history mode).
        self._history = history

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def consolidate(
        self,
        session: Session,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via LLM tool call.

        Returns True on success (including no-op), False on failure.

        Legacy interface — reads messages from session.messages in RAM.
        Prefer consolidate_messages() which accepts pre-loaded messages.
        """

        # Streaming-aware loader: when session.messages is empty and the
        # store is wired in, read the consolidation slice from disk so
        # streaming_history sessions still consolidate correctly.
        def _load_slice(start: int, end: int) -> list[dict[str, Any]]:
            if session.messages:
                offset = getattr(session, "_messages_offset", 0)
                rel_start = max(start - offset, 0)
                rel_end = end - offset
                return list(session.messages[rel_start:rel_end])
            if self._history is not None:
                return self._history.load_range(session.key, start, end)
            return []

        if archive_all:
            keep_count = 0
            old_messages = _load_slice(0, session.total_messages)
        else:
            keep_count = memory_window // 2
            total = getattr(session, "total_messages", len(session.messages))
            if total <= keep_count:
                return True
            if total - session.last_consolidated <= 0:
                return True
            old_messages = _load_slice(session.last_consolidated, total - keep_count)
            if not old_messages:
                return True

        return await self.consolidate_messages(
            session,
            old_messages=old_messages,
            archive_all=archive_all,
            memory_window=memory_window,
        )

    async def consolidate_messages(
        self,
        session: Session,
        *,
        old_messages: list[dict[str, Any]],
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate the given messages into MEMORY.md + HISTORY.md via LLM tool call.

        Unlike consolidate(), this accepts pre-loaded messages so the caller
        can load them from disk without keeping the full history in RAM.

        Returns True on success (including no-op), False on failure.
        """
        if not old_messages:
            return True

        keep_count = 0 if archive_all else memory_window // 2

        lines = []
        for m in old_messages:
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
            return False

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
                return False

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
                    return False
            if not isinstance(args, dict):
                logger.warning(
                    "memory_consolidation_skipped",
                    reason="unexpected_args_type",
                    **{"args.type": type(args).__name__},
                )
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            total = getattr(session, "total_messages", len(session.messages))
            if archive_all:
                session.last_consolidated = 0
            else:
                # Advance the boundary past any tool_use/tool_result group it
                # would split. Leaving a tool_result in the kept tail whose
                # tool_call_id lives in the archived region causes providers
                # like MiniMax to reject the next request with
                # "tool result's tool id(...) not found". Prefer sacrificing
                # a few messages at the boundary (they're neither summarized
                # nor kept) over producing an invalid conversation shape.
                boundary = total - keep_count
                offset = getattr(session, "_messages_offset", 0)
                # Streaming-aware read: when session.messages is empty,
                # load the boundary window from disk so the repair pass
                # can still see the messages it needs to inspect.
                if session.messages:
                    repair_window: list[dict[str, Any]] = list(session.messages)
                    window_offset = offset
                elif self._history is not None:
                    # Need messages around [boundary-1, total). A small
                    # window is enough — repair only walks forward.
                    window_start = max(boundary - 1, 0)
                    repair_window = self._history.load_range(session.key, window_start, total)
                    window_offset = window_start
                else:
                    repair_window = []
                    window_offset = offset
                while boundary < total:
                    rel = boundary - window_offset
                    if rel < 0 or rel >= len(repair_window):
                        break
                    curr = repair_window[rel]
                    prev = repair_window[rel - 1] if rel > 0 else None
                    if curr.get("role") == "tool":
                        boundary += 1
                        continue
                    if (
                        prev is not None
                        and prev.get("role") == "assistant"
                        and prev.get("tool_calls")
                    ):
                        boundary += 1
                        continue
                    break
                session.last_consolidated = boundary
            logger.info(
                "memory_consolidated",
                **{
                    "message.total": total,
                    "message.consolidated": session.last_consolidated,
                    "message.kept": keep_count,
                },
            )
            return True
        except Exception:
            logger.exception("memory_consolidation_failed")
            return False
