"""DefaultConversation — file-backed implementation of the Conversation protocol."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from exoclaw._compat import Path, WeakValueDictionary, bind_log_contextvars, get_logger
from exoclaw.utils import create_isolated_task

from .protocols import ConsolidationPolicy, HistoryStore, MemoryBackend, PromptBuilder
from .session.manager import Session

logger = get_logger()

if TYPE_CHECKING:
    from exoclaw.bus.protocol import Bus
    from exoclaw.providers.protocol import LLMProvider

# Injected before each user message at call time; stripped before persisting.
_RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

_TOOL_RESULT_MAX_CHARS = 500


class DefaultConversation:
    """
    File-backed conversation state manager.

    Implements the exoclaw Conversation protocol without inheriting from any
    exoclaw class.

    Accepts HistoryStore, MemoryBackend, and PromptBuilder as constructor
    arguments so each layer can be replaced independently. Use
    DefaultConversation.create() for the standard file-backed setup.

    - build_prompt: builds messages via PromptBuilder, triggers background
      consolidation when unconsolidated history exceeds memory_window.
    - record: saves new turn messages (stripping runtime context, truncating
      large tool results) into the JSONL session file.
    - clear: archives the current session to memory and resets to fresh state.
    - list_sessions: lists all sessions from the sessions directory.
    """

    def __init__(
        self,
        history: HistoryStore,
        memory: MemoryBackend,
        prompt: PromptBuilder,
        memory_window: int = 100,
        consolidation_policy: ConsolidationPolicy | None = None,
        bus: Bus | None = None,
    ):
        self.history = history
        self.memory = memory
        self.prompt = prompt
        self.memory_window = memory_window
        self._consolidation_policy = consolidation_policy
        self._bus: Bus | None = bus

        self._consolidating: set[str] = set()
        self._consolidation_tasks: set[asyncio.Task[Any]] = set()
        self._consolidation_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        # Turn context set by build_prompt(), read by record() for hook firing.
        self._turn_channel: str | None = None
        self._turn_chat_id: str | None = None
        self._turn_session_id: str | None = None

    @classmethod
    def create(
        cls,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        memory_window: int = 100,
        skill_packages: list[str] | None = None,
        consolidation_policy: ConsolidationPolicy | None = None,
        builtin_skills_dir: Path | None = None,
        allowed_skills: list[str] | None = None,
    ) -> DefaultConversation:
        """Construct with the standard file-backed implementations.

        ``builtin_skills_dir`` is the deployment-bundled skills root
        — intrinsic-to-this-deployment skills that ship with the
        firmware / server image and sit alongside workspace
        (agent-managed) skills in the loader. Version-locked with
        the deployment tag. On a chip this is typically the staged
        ``.stage/skills/`` directory copied into the firmware image;
        on a server it's an installer-relative path. ``None`` leaves
        only workspace + entry-point package skills visible.

        ``allowed_skills`` forwards to ``SkillsLoader.allowed_names``
        to restrict the visible surface to a known whitelist when
        the deployment-bundled set is meant to be exhaustive."""
        from .context import ContextBuilder
        from .memory import MemoryStore
        from .session.manager import SessionManager

        memory = MemoryStore(workspace, provider, model)
        return cls(
            history=SessionManager(workspace),
            memory=memory,
            prompt=ContextBuilder(
                workspace,
                memory=memory,
                skill_packages=skill_packages,
                builtin_skills_dir=builtin_skills_dir,
                allowed_skills=allowed_skills,
            ),
            memory_window=memory_window,
            consolidation_policy=consolidation_policy,
        )

    async def build_prompt(
        self,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        turn_context: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the full messages list to send to the LLM."""
        # Track turn context for hook firing in record().
        self._turn_channel = channel
        self._turn_chat_id = chat_id
        self._turn_session_id = session_id

        skills: list[str] | None = kwargs.get("skills")
        # ``isolated`` is an optional bool. Accept only actual booleans so
        # stringy values (e.g. the literal "false" from a misconfigured
        # upstream) can't silently enable isolation via Python truthiness —
        # ``bool("false")`` is True. Anything else is a caller bug; raise
        # rather than guess.
        if "isolated" not in kwargs:
            isolated: bool = False
        else:
            isolated_value = kwargs.get("isolated")
            if isinstance(isolated_value, bool):
                isolated = isolated_value
            else:
                raise TypeError(f"'isolated' must be a bool, got {type(isolated_value).__name__}")

        session = self.history.get_or_create(session_id)

        unconsolidated = session.total_messages - session.last_consolidated
        bind_log_contextvars(
            **{
                "session.total_messages": session.total_messages,
                "session.last_consolidated": session.last_consolidated,
                "session.unconsolidated": unconsolidated,
                "session.has_summary": bool(session.metadata.get("summary")),
                "memory.window": self.memory_window,
                "consolidation.active": session_id in self._consolidating,
                "skill.requested": ",".join(skills) if skills else "",
                "hook.active": channel == "hook",
                "isolated": isolated,
            }
        )

        # Skip consolidation entirely in isolated mode — the whole point is
        # that the caller treats this invocation as a stateless function,
        # so there's no history worth summarizing.
        if not isolated:
            # Trigger background consolidation when policy says so (or default: history is long)
            should = await self._should_consolidate(session)
            if should and session_id not in self._consolidating:
                self._consolidating.add(session_id)
                lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())

                async def _consolidate_and_unlock() -> None:
                    try:
                        async with lock:
                            success = await self._consolidate_memory(session)
                            if success:
                                self.history.save_metadata(session)
                    finally:
                        self._consolidating.discard(session_id)
                        _task = asyncio.current_task()
                        if _task is not None:
                            self._consolidation_tasks.discard(_task)

                # Isolate from caller contextvars — when build_prompt
                # is reached from inside a DBOS workflow, the spawned
                # consolidation task would otherwise inherit
                # DBOSContext and any future workflow-aware call in
                # the consolidation path would be misclassified.
                _task = create_isolated_task(_consolidate_and_unlock())
                self._consolidation_tasks.add(_task)

        # Isolated mode skips session history entirely — the LLM sees only
        # [system(minimal), user(current_message)]. Keeping history would
        # reintroduce contamination from earlier turns on the same
        # session_key (e.g. many cron-fired enrichments sharing a key).
        history: list[dict[str, Any]]
        if isolated:
            history = []
        else:
            # Route through the store rather than session.get_history so a
            # streaming-enabled HistoryStore reads the unconsolidated tail
            # from disk on demand instead of holding it in session.messages.
            history = self.history.read_history(session_id, max_messages=self.memory_window)

        extra_context: str | None = None
        if plugin_context:
            extra_context = "\n\n".join(plugin_context)

        # Inject per-session summary from consolidation policy (if present).
        # Isolated mode skips this too for the same reason — no carryover.
        effective_turn_context = list(turn_context or [])
        if not isolated:
            summary = session.metadata.get("summary")
            if summary:
                effective_turn_context.insert(0, f"## Previous Session Summary\n{summary}")

        messages = self.prompt.build_messages(
            history=history,
            current_message=message,
            skill_names=skills,
            media=media,
            channel=channel,
            chat_id=chat_id,
            extra_context=extra_context,
            turn_context=effective_turn_context or None,
            isolated=isolated,
        )

        # Bind skill/tool context after build_messages (which resolves active skills/tools)
        get_tools = getattr(self.prompt, "get_active_optional_tools", None)
        skills_loader = getattr(self.prompt, "skills", None)
        if get_tools and skills_loader:
            active_tools = get_tools()
            always_skills = skills_loader.get_always_skills()
            extra_skills = [s for s in (skills or []) if s not in always_skills]
            active_skills = always_skills + extra_skills
            bind_log_contextvars(
                **{
                    "skill.always": ",".join(always_skills),
                    "skill.active": ",".join(active_skills),
                    "skill.active.count": len(active_skills),
                    "tool.optional.active": ",".join(sorted(active_tools)),
                    "tool.optional.active.count": len(active_tools),
                }
            )

        return messages

    def load_persisted_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return the same history slice ``build_prompt`` uses for a session.

        Synchronous, with no additional transformations beyond
        ``session.get_history(...)`` — returns the unconsolidated tail
        that ``build_prompt`` would read from history (trimmed to
        ``memory_window``, with leading non-user messages dropped to
        avoid orphan tool-result blocks). Unlike ``build_prompt``,
        this method:

        * does NOT build the system prompt, runtime context, or render
          the new user message. Prefix/suffix assembly stays in
          ``build_prompt`` where it belongs;
        * does NOT trigger consolidation as a side effect;
        * does NOT touch the async event loop — it's a synchronous
          history lookup so an executor's ``PriorSource`` closure
          (phase 2b of docs/memory-model.md) can call it from the
          in-progress LLM iteration.

        Intended caller: an executor that installs
        ``set_prior_source(lambda: conv.load_persisted_history(session_id))``
        as the lazy prior after the initial ``build_prompt`` runs.
        Successive ``load_messages`` calls then re-read the session
        state rather than holding a Python-heap list between LLM
        iterations.

        Structure-wise: because this method skips the system prompt /
        runtime context / new-user-message assembly, callers that need
        those still have to source them somewhere (typically by
        caching the initial ``build_prompt`` return's prefix and
        suffix). The split lives in the executor so it can compose
        ``[*prefix, *load_persisted_history(session_id), *suffix]``
        per iteration — prefix and suffix are small and cheap to hold.
        """
        return self.history.read_history(session_id, max_messages=self.memory_window)

    async def append(
        self,
        session_id: str,
        message: dict[str, Any],
    ) -> None:
        """Persist a single message as it is produced during a turn.

        Implements ``AppendableConversation`` from exoclaw>=0.19.0. The
        agent loop calls this after each assistant response, tool
        result, and the incoming user message. ``_prepare_turn`` runs
        per-message (tool-result truncation, runtime-context tag
        stripping, etc.) so calling it with a one-element list is the
        same shape as the end-of-turn batch path.

        No hooks fire here — ``post_turn`` owns the end-of-turn hook
        trigger so consolidation / agent_end callbacks only run once
        per turn, not per message.
        """
        session = self.history.get_or_create(session_id)
        prepared = self._prepare_turn(session, [message])
        # ``_prepare_turn`` may return an empty list (empty assistant,
        # runtime-context-only user) — in which case there is nothing
        # to flush and ``save_append`` would write a metadata-only
        # file on first call for the session. Skip.
        if prepared:
            self.history.save_append(session, prepared)

    async def post_turn(self, session_id: str) -> None:
        """Fire end-of-turn hooks after all messages have been persisted.

        Called once per turn by the agent loop when the append path is
        active. Hook turns (``channel="hook"``) are skipped to prevent
        recursion — an agent_end hook that calls back into the bot
        would otherwise retrigger itself.
        """
        if self._bus and self._turn_channel != "hook":
            await self._fire_agent_hooks(session_id)

    async def record(
        self,
        session_id: str,
        new_messages: list[dict[str, Any]],
    ) -> None:
        """Persist the messages produced during one turn.

        Legacy end-of-turn batch path. The agent loop calls this only
        when the Conversation doesn't satisfy ``AppendableConversation``
        (i.e. doesn't implement ``append``). This implementation does
        — see above — so under a current-version agent loop this
        method isn't called during normal turns. Kept for external
        callers that still drive persistence via ``record`` directly.
        """
        session = self.history.get_or_create(session_id)
        prepared = self._prepare_turn(session, new_messages)
        self.history.save_append(session, prepared)

        # Fire agent_end hooks via the bus.  Hook turns use channel="hook"
        # and are skipped to prevent recursion.
        if self._bus and self._turn_channel != "hook":
            await self._fire_agent_hooks(session_id)

    async def clear(self, session_id: str) -> bool:
        """Archive current session to memory and start fresh. Returns True on success."""
        session = self.history.get_or_create(session_id)
        lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())
        self._consolidating.add(session_id)
        try:
            async with lock:
                # Under streaming_history session.messages is empty — read
                # the unconsolidated tail from disk so clear()-time
                # archival still has something to consolidate.
                if getattr(self.history, "streaming_history", False):
                    snapshot = self.history.load_range(
                        session_id, session.last_consolidated, session.total_messages
                    )
                else:
                    snapshot = list(session.messages)
                if snapshot:
                    temp = Session(key=session_id)
                    temp.messages = list(snapshot)
                    success = await self._consolidate_memory(temp, archive_all=True)
                    if not success:
                        return False
        except Exception:
            logger.exception("session_clear_failed", **{"session.id": session_id})
            return False
        finally:
            self._consolidating.discard(session_id)

        session.clear()
        self.history.save(session)
        self.history.invalidate(session_id)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all known sessions."""
        return self.history.list_sessions()

    def active_tools(self) -> set[str]:
        """Return optional tool names activated by the current turn's skills."""
        return self.prompt.get_active_optional_tools()

    async def _fire_agent_hooks(self, session_id: str) -> None:
        """Discover agent_end hooks and publish them as inbound messages on the bus."""
        from exoclaw.bus.events import InboundMessage

        skills_loader = getattr(self.prompt, "skills", None)
        if skills_loader is None:
            return

        hooks = skills_loader.get_agent_hooks("agent_end")
        if not hooks:
            return

        chat_id = self._turn_chat_id or session_id
        for hook in hooks:
            try:
                await self._bus.publish_inbound(  # type: ignore[union-attr]
                    InboundMessage(
                        channel="hook",
                        sender_id=f"hook:{hook.skill_name}:agent_end",
                        chat_id=chat_id,
                        content=hook.prompt,
                        metadata={
                            "_hook_turn": True,
                            "hook_name": "agent_end",
                            "hook_skill": hook.skill_name,
                            "hook_tools": hook.tools,
                            "hook_skills": hook.skills,
                            "source_session_id": session_id,
                        },
                    )
                )
            except Exception:
                logger.warning(
                    "agent_hook_publish_failed",
                    hook_skill=hook.skill_name,
                    exc_info=True,
                )

    async def _should_consolidate(self, session: Session) -> bool:
        """Check whether consolidation should run."""
        if self._consolidation_policy is not None:
            return await self._consolidation_policy.should_consolidate(
                session, memory_window=self.memory_window
            )
        unconsolidated = session.total_messages - session.last_consolidated
        return unconsolidated >= self.memory_window

    async def _consolidate_memory(self, session: Session, archive_all: bool = False) -> bool:
        """Delegate to ConsolidationPolicy if present, otherwise MemoryBackend.

        Loads the consolidation range from disk so we don't need the full
        message history in RAM.
        """
        if self._consolidation_policy is not None:
            return await self._consolidation_policy.consolidate(
                session,
                archive_all=archive_all,
                memory_window=self.memory_window,
            )

        # Load only the messages that need consolidating from disk
        if archive_all:
            old_messages = self._load_consolidation_range(session, archive_all=True)
        else:
            old_messages = self._load_consolidation_range(session, archive_all=False)

        if not old_messages:
            return True

        return await self.memory.consolidate_messages(
            session,
            old_messages=old_messages,
            archive_all=archive_all,
            memory_window=self.memory_window,
        )

    def _load_consolidation_range(
        self, session: Session, *, archive_all: bool
    ) -> list[dict[str, Any]]:
        """Load the message range to consolidate from disk."""
        if archive_all:
            loaded = self.history.load_range(session.key, 0, session.total_messages)
            return loaded or list(session.messages)

        keep_count = self.memory_window // 2
        if session.total_messages <= keep_count:
            return []
        end = session.total_messages - keep_count
        start = session.last_consolidated
        if start >= end:
            return []
        loaded = self.history.load_range(session.key, start, end)
        return loaded or session.messages[: end - start]

    def _prepare_turn(
        self, session: Session, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare turn messages, truncating large tool results.

        Appends to session.messages (in-memory view) and returns the
        list of prepared entries for disk persistence.
        """
        from datetime import datetime

        prepared: list[dict[str, Any]] = []

        for m in messages:
            entry = dict(m)
            role = entry.get("role")
            content = entry.get("content")

            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context

            if (
                role == "tool"
                and isinstance(content, str)
                and len(content) > _TOOL_RESULT_MAX_CHARS
            ):
                entry["content"] = content[:_TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"

            elif role == "user":
                if isinstance(content, str) and content.startswith(_RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if (
                            c.get("type") == "text"
                            and isinstance(c.get("text"), str)
                            and c["text"].startswith(_RUNTIME_CONTEXT_TAG)
                        ):
                            continue
                        if c.get("type") == "image_url" and c.get("image_url", {}).get(
                            "url", ""
                        ).startswith("data:image/"):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered

            entry.setdefault("timestamp", datetime.now().isoformat())
            new_total = session.total_messages + 1
            # Under streaming_history we don't grow session.messages — the
            # whole point is that the unconsolidated tail lives only on
            # disk between turns. ``save_append`` still flushes ``entry``
            # to JSONL; ``total_messages`` advances so the next
            # ``read_history`` reads through the new boundary.
            if not getattr(self.history, "streaming_history", False):
                session.messages.append(entry)
            session._total_messages = new_total
            prepared.append(entry)

        session.updated_at = datetime.now()
        return prepared
