"""DefaultConversation — file-backed implementation of the Conversation protocol."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from exoclaw._compat import Path, WeakValueDictionary, bind_log_contextvars, get_logger
from exoclaw.utils import create_isolated_task

from . import _consolidation_state as state_io
from .protocols import ConsolidationPolicy, HistoryStore, MemoryBackend, PromptBuilder
from .session.manager import Session, _repair_and_project

logger = get_logger()

if TYPE_CHECKING:
    from exoclaw.bus.protocol import Bus
    from exoclaw.providers.protocol import LLMProvider

# Injected before each user message at call time; stripped before persisting.
_RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

_TOOL_RESULT_MAX_CHARS = 500


class _NoOpPolicy:
    """Inline policy used when a caller doesn't configure one. Streams
    the session log unchanged and runs no consolidation."""

    def transform(self, reader, *, budget=None):  # type: ignore[no-untyped-def]
        async def _passthrough():  # type: ignore[no-untyped-def]
            async for m in reader.stream():
                yield m

        return _passthrough()

    async def on_turn_complete(self, reader) -> None:  # type: ignore[no-untyped-def]
        return None


_NOOP_POLICY = _NoOpPolicy()


class DefaultConversation:
    """File-backed conversation state manager.

    Implements the exoclaw Conversation protocol without inheriting from any
    exoclaw class.

    Accepts ``HistoryStore``, ``MemoryBackend``, ``PromptBuilder``, and
    ``ConsolidationPolicy`` as constructor arguments so each layer can be
    replaced independently.

    The session log is append-only — this class never rewrites or
    truncates message data on disk. The ``ConsolidationPolicy`` owns
    the *view* the LLM sees: it transforms a streaming reader over the
    log into the message list, persisting its own state in a sidecar
    next to the session file.

    - ``build_prompt``: ``policy.transform(reader)`` →
      ``PromptBuilder.build_messages``.
    - ``append`` / ``record``: appends new turn messages to disk.
    - ``post_turn``: schedules ``policy.on_turn_complete`` as
      background work for periodic consolidation. Callers run this
      after each turn.
    - ``recover_from_overflow``: reactive seam consumed by
      ``AgentLoop`` on ``ContextWindowExceededError`` — asks the
      policy to advance its sidecar by one chunk, then re-emits the
      compacted view.
    - ``clear``: resets the session log to a metadata-only header
      and removes the policy sidecar. No automatic archival;
      archive explicitly first if needed.
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
        self._consolidation_policy: ConsolidationPolicy = consolidation_policy or _NOOP_POLICY  # type: ignore[assignment]
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

        bind_log_contextvars(
            **{
                "memory.window": self.memory_window,
                "consolidation.active": session_id in self._consolidating,
                "skill.requested": ",".join(skills) if skills else "",
                "hook.active": channel == "hook",
                "isolated": isolated,
            }
        )

        # Isolated mode skips session history entirely — the LLM sees only
        # [system(minimal), user(current_message)]. Keeping history would
        # reintroduce contamination from earlier turns on the same
        # session_key (e.g. many cron-fired enrichments sharing a key).
        history: list[dict[str, Any]]
        if isolated:
            history = []
        else:
            reader = self.history.reader(session_id)
            # Async list comprehensions (PEP 530) don't parse on
            # MicroPython 1.27 — build via explicit ``async for``.
            history = []
            async for _m in self._consolidation_policy.transform(reader):
                history.append(_m)
            # Project to LLM-input shape (strip ``timestamp`` and other
            # persistence-only fields) and repair any tool_call /
            # tool_result orphans inside the policy's view. Don't
            # peel leading non-user — the rolling-summary preamble
            # the policy emits is a leading system message that must
            # survive.
            history = _repair_and_project(history)

        extra_context: str | None = None
        if plugin_context:
            extra_context = "\n\n".join(plugin_context)

        messages = self.prompt.build_messages(
            history=history,
            current_message=message,
            skill_names=skills,
            media=media,
            channel=channel,
            chat_id=chat_id,
            extra_context=extra_context,
            turn_context=list(turn_context) if turn_context else None,
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

    async def post_turn(
        self,
        session_id: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        await_maintenance: bool = False,
    ) -> None:
        """End-of-turn callback: schedule policy maintenance and fire
        agent_end hooks.

        By default ``policy.on_turn_complete`` runs in a background
        task isolated from the caller's contextvars, so a long-lived
        worker doesn't propagate caller context into the consolidation
        pass. Hook turns (``channel="hook"``) skip both maintenance
        and hook firing to prevent recursion.

        Parameters
        ----------
        channel, chat_id:
            Override the turn context recorded by ``build_prompt``.
            Required when ``build_prompt`` and ``post_turn`` are
            invoked on different ``Conversation`` instances — e.g.
            stateless workflow runners or durable execution
            environments that reconstruct the conversation per step.
            When ``None``, falls back to instance state.
        await_maintenance:
            When True, run maintenance synchronously inline instead
            of scheduling a background task. Required for callers
            that can't keep a background task alive past the
            surrounding function call (stateless workflow steps,
            short-lived invocations). The synchronous path also
            preserves the caller's contextvars so that the
            maintenance pass's logs/traces are attributed to the
            invoking step.
        """
        effective_channel = channel if channel is not None else self._turn_channel
        if effective_channel == "hook":
            return

        if await_maintenance:
            await self._run_maintenance(session_id)
        else:
            self._schedule_maintenance(session_id)

        if self._bus:
            await self._fire_agent_hooks(session_id, chat_id=chat_id)

    async def record(
        self,
        session_id: str,
        new_messages: list[dict[str, Any]],
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        await_maintenance: bool = False,
    ) -> None:
        """Persist the messages produced during one turn.

        Legacy end-of-turn batch path. The agent loop calls this only
        when the Conversation doesn't satisfy ``AppendableConversation``
        (i.e. doesn't implement ``append``). This implementation does
        — see above — so under a current-version agent loop this
        method isn't called during normal turns. Kept for external
        callers that still drive persistence via ``record`` directly.

        ``channel``, ``chat_id``, and ``await_maintenance`` mirror the
        options on ``post_turn`` for callers that drive persistence
        through ``record`` without a paired ``post_turn`` invocation.
        See ``post_turn`` for the rationale on each.
        """
        session = self.history.get_or_create(session_id)
        prepared = self._prepare_turn(session, new_messages)
        self.history.save_append(session, prepared)

        # Legacy batch path — fire end-of-turn machinery directly.
        effective_channel = channel if channel is not None else self._turn_channel
        if effective_channel != "hook":
            if await_maintenance:
                await self._run_maintenance(session_id)
            else:
                self._schedule_maintenance(session_id)
            if self._bus:
                await self._fire_agent_hooks(session_id, chat_id=chat_id)

    async def clear(self, session_id: str) -> bool:
        """Reset the session log to a metadata-only header and remove
        the policy sidecar. Returns True on success.

        Note: the JSONL file is rewritten (not unlinked) — file-backed
        ``HistoryStore`` implementations preserve the file so
        ``list_sessions`` still surfaces the empty session.

        No automatic archival — if a caller wants the session
        summarized to long-term memory before it disappears, they
        must drive that explicitly via the ``MemoryBackend``
        (``memory.summarize(messages, ...)``) before ``clear``.
        """
        session = self.history.get_or_create(session_id)
        try:
            session.clear()
            self.history.save(session)
            self.history.invalidate(session_id)
            # Best-effort sidecar cleanup. Sidecars live next to the
            # session JSONL by convention; if a custom HistoryStore
            # uses a different layout, the policy is responsible for
            # cleanup via its own hook.
            sessions_dir = getattr(self.history, "sessions_dir", None)
            if sessions_dir is not None:
                state_io.delete_state(sessions_dir, session_id)
            return True
        except Exception:
            logger.exception("session_clear_failed", **{"session.id": session_id})
            return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all known sessions."""
        return self.history.list_sessions()

    def active_tools(self) -> set[str]:
        """Return optional tool names activated by the current turn's skills."""
        return self.prompt.get_active_optional_tools()

    async def recover_from_overflow(self, session_id: str) -> list[dict[str, Any]] | None:
        """Reactive overflow-recovery seam consumed by ``AgentLoop``
        (via ``Executor.recover_from_overflow``) on
        ``ContextWindowExceededError``.

        Asks the consolidation policy to advance its sidecar by one
        chunk, then re-assembles the prompt from the post-recovery
        view. Returns the new message list (caller passes to
        ``executor.set_messages`` and retries) or ``None`` when the
        policy can't make progress.

        The returned list is ``[system_prompt, *recovered_view]`` —
        no new user message is appended. The in-flight turn's
        messages are already in the active log (persisted via
        ``append`` as the turn produced them) and surface naturally
        through the policy's transform.
        """
        recover = getattr(self._consolidation_policy, "recover_from_overflow", None)
        if recover is None:
            return None

        reader = self.history.reader(session_id)
        advanced = await recover(reader)
        if not advanced:
            return None

        # Re-materialize the active view through the policy. Includes
        # the rolling summary preamble (if any) plus the tail past
        # the freshly-advanced ``summarized_through`` pointer. Built
        # via explicit ``async for`` because async list comprehensions
        # don't parse on MicroPython 1.27.
        recovered_view: list[dict[str, Any]] = []
        async for _m in self._consolidation_policy.transform(reader):
            recovered_view.append(_m)
        # Same projection/repair as ``build_prompt`` — strip
        # persistence-only fields and repair tool-pair orphans
        # before handing the list to the executor for retry.
        recovered_view = _repair_and_project(recovered_view)

        # Prepend the system prompt manually rather than going through
        # ``build_messages`` — the latter would append an empty user
        # message at the end (its current_message argument). For
        # recovery there's no fresh user input; the active log already
        # carries the in-flight turn's user message and any
        # tool-call/result messages produced so far.
        #
        # ``build_system_prompt`` lives on the default ``ContextBuilder``
        # but isn't on the ``PromptBuilder`` protocol — feature-detect
        # via getattr so custom builders that don't expose it fall
        # through to the no-system-prompt path. The resulting prompt
        # is still valid (just leaner); custom builders that care
        # should add the method.
        get_sys_prompt = getattr(self.prompt, "build_system_prompt", None)
        if get_sys_prompt is not None:
            system_content = get_sys_prompt()
            # PEP 448 list-unpack ``[a, *xs]`` doesn't parse on
            # MicroPython 1.27. Build via list concat instead — same
            # result, runs cross-runtime.
            return [{"role": "system", "content": system_content}] + list(recovered_view)
        return list(recovered_view)

    # ─── Internal: maintenance + hook plumbing ───

    async def _run_maintenance(self, session_id: str) -> None:
        """Run ``policy.on_turn_complete`` synchronously inline.

        Public-but-underscored entry point shared by
        ``_schedule_maintenance`` (fire-and-forget background task)
        and the ``await_maintenance=True`` path on ``post_turn`` /
        ``record`` (synchronous inline). Idempotent per session —
        re-entrant calls while another maintenance pass is in flight
        return immediately.
        """
        if session_id in self._consolidating:
            return
        self._consolidating.add(session_id)
        lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())
        reader = self.history.reader(session_id)
        try:
            async with lock:
                await self._consolidation_policy.on_turn_complete(reader)
        except Exception:
            logger.exception("policy_on_turn_complete_failed", **{"session.id": session_id})
        finally:
            self._consolidating.discard(session_id)

    def _schedule_maintenance(self, session_id: str) -> None:
        """Spawn a background task that runs ``_run_maintenance``.

        Used by the default ``post_turn`` path for callers whose
        process lifetime extends past the surrounding function call.
        Isolates the maintenance task from caller contextvars to keep
        consolidation logs/traces separate from the in-flight turn.
        """

        async def _run_and_untrack() -> None:
            try:
                await self._run_maintenance(session_id)
            finally:
                _task = asyncio.current_task()
                if _task is not None:
                    self._consolidation_tasks.discard(_task)

        _task = create_isolated_task(_run_and_untrack())
        self._consolidation_tasks.add(_task)

    async def _fire_agent_hooks(
        self,
        session_id: str,
        *,
        chat_id: str | None = None,
    ) -> None:
        """Discover agent_end hooks and publish them as inbound messages on the bus.

        ``chat_id`` overrides ``self._turn_chat_id`` when callers
        invoke ``post_turn`` / ``record`` from a different instance
        than the one that ran ``build_prompt``.
        """
        from exoclaw.bus.events import InboundMessage

        skills_loader = getattr(self.prompt, "skills", None)
        if skills_loader is None:
            return

        hooks = skills_loader.get_agent_hooks("agent_end")
        if not hooks:
            return

        chat_id = chat_id or self._turn_chat_id or session_id
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

    def _prepare_turn(
        self, session: Session, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare turn messages, truncating large tool results.

        Returns the list of prepared entries for disk persistence.
        Does not mutate ``session.messages`` — the session log is
        append-only on disk; in-memory caching is a back-compat
        artifact maintained by ``SessionManager`` for non-streaming
        deployments and is not relied on here.
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
            # Compute new total BEFORE appending — when ``_total_messages``
            # is 0 (fresh session) ``total_messages`` derives from
            # ``len(messages)``, so reading it after the append would
            # double-count the new entry.
            new_total = session.total_messages + 1
            # Append to in-memory tail too for non-streaming HistoryStore
            # implementations that still use ``session.messages`` for
            # bookkeeping. Streaming-aware stores opt out via the flag.
            if not getattr(self.history, "streaming_history", False):
                session.messages.append(entry)
            session._total_messages = new_total
            prepared.append(entry)

        session.updated_at = datetime.now()
        return prepared
