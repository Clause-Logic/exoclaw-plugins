"""DBOS-backed durable executor for exoclaw.

Drop-in replacement for DirectExecutor. Every LLM call and tool execution
is a DBOS step, automatically checkpointed to SQLite. If the process
restarts mid-turn, DBOS replays completed steps from the journal.

Architecture follows the same pattern as standd_agent's TemporalExecutor:
the agent loop runs inside a @DBOS.workflow(), and each chat/tool call
is a @DBOS.step().

Usage in nanobot wiring:
    from exoclaw_executor_dbos import run_durable_turn, DBOSExecutor

    # In message processing, instead of calling AgentLoop._process_message:
    await run_durable_turn(session_id, message, ...)
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from dbos import DBOS, Queue, SetWorkflowID
from exoclaw.agent.conversation import Conversation
from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import LLMResponse, ToolCallRequest
from uuid_utils import uuid7

from .intents import (
    StartChildWorkflow,
    _bind_intent_buffer,
    _release_intent_buffer,
)

# Source for per-turn prior-history messages — phase 2b of
# exoclaw/docs/memory-model.md. ``DBOSExecutor.load_messages``
# invokes the source on every LLM iteration; installing a
# disk-backed closure (via ``set_prior_source``) lets prior not be
# heap-resident between iterations. The back-compat
# ``set_messages`` path snapshots the list in a closure, matching
# pre-phase-2b RAM behaviour.
_PriorSource = Callable[[], list[dict[str, object]]]


def _empty_prior_source() -> list[dict[str, object]]:
    """Default prior source before any seed call. Returns an empty
    list so a premature ``load_messages`` doesn't raise
    ``LookupError``."""
    return []


def _build_lazy_prior_source(
    *,
    full: list[dict[str, object]],
    history_snapshot: list[dict[str, object]],
    reload_history: Callable[[], list[dict[str, object]]],
) -> _PriorSource | None:
    """Construct a disk-backed ``PriorSource`` for a turn's prior.

    Locates ``history_snapshot`` inside ``full`` by dict ``id()``
    matching — ``session.get_history`` and ``Conversation.build_prompt``
    share dict refs into ``session.messages``, so id matching is
    reliable when it works. When it doesn't (empty history, isolated
    mode, a PromptBuilder that deep-copies), returns ``None`` so the
    caller falls back to the closure-over-list path.

    The returned source closes over ``prefix`` and ``suffix`` (small,
    stable for the turn) and invokes ``reload_history`` on each
    call. RAM savings come from the history slice — usually the bulk
    of prompt size — not being heap-resident between LLM iterations.
    """
    if not history_snapshot:
        return None
    history_ids = {id(m) for m in history_snapshot}
    first_idx: int | None = None
    for i, m in enumerate(full):
        if id(m) in history_ids:
            first_idx = i
            break
    if first_idx is None:
        return None
    last_idx = first_idx + len(history_snapshot) - 1
    if last_idx >= len(full):
        return None
    # Verify the whole slice matches history by identity, not just
    # the first hit. A ``PromptBuilder`` that replaces some history
    # dicts while leaving others shared (e.g. tool-result compaction)
    # would partially overlap the id set at the right starting index
    # but the slice itself mixes original + transformed dicts. Using
    # this source would then re-inject the untransformed history on
    # later iterations and diverge from what the initial LLM call saw.
    # Bail to the snapshot path if any dict in the slice isn't one of
    # the history refs.
    slice_ids = [id(m) for m in full[first_idx : last_idx + 1]]
    expected_ids = [id(m) for m in history_snapshot]
    if slice_ids != expected_ids:
        return None
    prefix = list(full[:first_idx])
    suffix = list(full[last_idx + 1 :])

    def _source() -> list[dict[str, object]]:
        return [*prefix, *reload_history(), *suffix]

    return _source


# ── Serialization helpers ────────────────────────────────────────────────────


def _response_to_dict(resp: LLMResponse) -> dict[str, Any]:
    return dataclasses.asdict(resp)


def _dict_to_response(d: dict[str, Any]) -> LLMResponse:
    d = dict(d)  # don't mutate caller's dict
    tool_calls = [ToolCallRequest(**tc) for tc in d.pop("tool_calls", [])]
    return LLMResponse(tool_calls=tool_calls, **d)


# ── Per-task context for non-serializable refs ───────────────────────────────
# ContextVars are safe for concurrent workflows — each asyncio Task gets
# its own copy, so parallel turns don't stomp on each other.

_conversation_var: contextvars.ContextVar[Conversation | None] = contextvars.ContextVar(
    "_conversation_var", default=None
)
_provider_var: contextvars.ContextVar[LLMProvider | None] = contextvars.ContextVar(
    "_provider_var", default=None
)
_registry_var: contextvars.ContextVar[ToolRegistry | None] = contextvars.ContextVar(
    "_registry_var", default=None
)


# ── DBOS step functions ──────────────────────────────────────────────────────
# Module-level so DBOS can register and replay them.


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2)
async def _chat_step(
    messages: list[dict[str, Any]],
    tools_json: str | None,
    model: str | None,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Durable LLM call. Result is cached by DBOS on completion."""
    provider = _provider_var.get()
    if provider is None:
        raise RuntimeError("provider not set — call set_turn_context() before running turns")
    tools = json.loads(tools_json) if tools_json else None
    resp = await provider.chat(
        messages=messages,
        tools=tools,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    return _response_to_dict(resp)


@DBOS.step(retries_allowed=True, max_attempts=2, interval_seconds=1)
async def _tool_step(
    name: str,
    params: dict[str, Any],
    ctx_data: dict[str, Any] | None,
) -> str:
    """Durable tool execution. Result is cached by DBOS on completion."""
    registry = _registry_var.get()
    if registry is None:
        raise RuntimeError("registry not set — call set_turn_context() before running turns")
    ctx = ToolContext(**ctx_data) if ctx_data else None
    return await registry.execute(name, params, ctx)


@DBOS.step()
async def _append_message_step(
    session_id: str,
    message: dict[str, Any],
) -> None:
    """Persist a single turn message via the Conversation, journaled.

    Wrapped in a ``@DBOS.step`` so recovery replays the journaled
    completion without re-invoking ``conversation.append`` — a plain
    JSONL-append without idempotency would otherwise double-write on
    crash replay. Conversation is read from a ContextVar set by
    ``DBOSExecutor.append_message`` rather than passed in: Conversation
    isn't JSON-serializable so it can't go through step arguments.

    Callers (``DBOSExecutor.append_message``) only invoke this step
    when the agent loop has already confirmed the Conversation
    implements ``AppendableConversation``. Reaching the step with a
    non-appendable conversation or no conversation at all is a wiring
    bug — fail loudly rather than silently drop the persistence and
    lose the message.
    """
    conversation = _conversation_var.get()
    if conversation is None:
        raise RuntimeError(
            "conversation not set on _conversation_var — "
            "DBOSExecutor.append_message should have set it before invoking this step"
        )
    fn = getattr(conversation, "append", None)
    if not callable(fn):
        raise TypeError(
            f"conversation {type(conversation).__name__} has no callable ``append`` — "
            "only implementations of AppendableConversation may reach this step"
        )
    await fn(session_id, message)


@DBOS.step()
async def _post_turn_step(session_id: str) -> None:
    """Fire end-of-turn hooks via the Conversation, journaled.

    Same ContextVar pattern as ``_append_message_step`` — the step
    carries ``session_id`` in its arguments and reads the conversation
    from the process-local ContextVar. Missing ``post_turn`` is a
    wiring bug: fail loudly rather than skip end-of-turn hooks.
    """
    conversation = _conversation_var.get()
    if conversation is None:
        raise RuntimeError(
            "conversation not set on _conversation_var — "
            "DBOSExecutor.post_turn should have set it before invoking this step"
        )
    fn = getattr(conversation, "post_turn", None)
    if not callable(fn):
        raise TypeError(
            f"conversation {type(conversation).__name__} has no callable ``post_turn`` — "
            "only implementations of AppendableConversation may reach this step"
        )
    await fn(session_id)


@DBOS.step()
async def _mint_turn_id_step() -> str:
    """Replay-safe turn id minted via uuidv7.

    Wraps the only non-deterministic part — the uuidv7 mint itself —
    in a DBOS step so the same id is journaled on first run and
    returned from the journal on workflow recovery. Without the step
    boundary, a recovered workflow would mint a *new* turn id on
    replay and break ``turn.root_id`` correlation across the crash.
    """
    return str(uuid7())


# ── Workflow registry for deferred-intent dispatch ───────────────────────────
# Workflows that get started via StartChildWorkflow intents register
# themselves here at import time. The executor resolves the intent's
# ``workflow_key`` to the actual decorated function so intent producers
# (e.g. DBOSSubagentSpawner) never need to import DBOS workflow refs.

_WorkflowRef = Callable[..., Coroutine[Any, Any, Any]]
_workflow_registry: dict[str, _WorkflowRef] = {}
_queue_registry: dict[str, "Queue"] = {}


def register_intent_workflow(key: str, workflow: _WorkflowRef) -> None:
    """Register a DBOS workflow function under a string key.

    Called at module import time by packages that ship workflows
    intended to be started via ``StartChildWorkflow`` intents (e.g.
    ``exoclaw_executor_dbos.subagent``).
    """
    _workflow_registry[key] = workflow


def register_intent_queue(key: str, queue: "Queue") -> None:
    """Attach a DBOS ``Queue`` to a previously-registered intent workflow.

    When present, intent dispatch uses ``queue.enqueue_async(...)``
    instead of ``DBOS.start_workflow_async(...)`` so the queue's
    concurrency / rate-limit config governs execution. Passing no queue
    (the default) preserves the original direct-start behavior.

    Intended call site is the spawner's ``__init__`` — the spawner owns
    its queue config (e.g. ``max_concurrent``) and attaches it here
    once, at wiring time.
    """
    _queue_registry[key] = queue


def unregister_intent_queue(key: str) -> None:
    """Detach a queue from an intent workflow. Used by tests for cleanup."""
    _queue_registry.pop(key, None)


# ── DBOSExecutor ─────────────────────────────────────────────────────────────


class DBOSExecutor:
    """Executor that routes AgentLoop operations through DBOS steps.

    Must be used inside a @DBOS.workflow() — see run_durable_turn().
    Sets ContextVar refs so steps can access provider/registry safely
    across concurrent workflows.
    """

    # Signals to AgentLoop._process_message that the executor will publish
    # the final reply to the bus from inside the workflow. When True and
    # the caller opted in via ``publish_response=True``, _process_message
    # returns None so _dispatch doesn't double-publish.
    handles_response_send: bool = True

    def __init__(self) -> None:
        # Per-turn buffer split into prior (read-only, seeded by
        # build_prompt or compaction) and delta (appended to mid-turn).
        # Mirrors ``DirectExecutor``'s phase 2a shape — keeps prior
        # replaceable by a lazy ``PriorSource`` (phase 2b: see
        # ``set_prior_source`` below) so the history slice can be
        # re-read on each LLM iteration rather than heap-resident.
        #
        # Both vars are per-instance ContextVars: per-instance so two
        # executors in the same task don't share state (unusual in
        # production, common in tests); per-task via the ContextVar
        # machinery so concurrent turns on the same executor don't
        # trample each other's buffer (e.g. a periodic background
        # task firing while a user-initiated turn is still running).
        # The buffer does not need to be durable across DBOS recovery
        # because ``run_durable_turn`` encapsulates the whole turn.
        self._prior_var: contextvars.ContextVar[_PriorSource] = contextvars.ContextVar(
            f"dbos_executor_prior_{id(self)}"
        )
        self._delta_var: contextvars.ContextVar[list[dict[str, object]]] = contextvars.ContextVar(
            f"dbos_executor_delta_{id(self)}"
        )

    def __deepcopy__(self, memo: dict) -> DBOSExecutor:
        # ContextVar objects are not deep-copyable (TypeError: cannot
        # pickle '_contextvars.ContextVar'). ``ToolContext.executor``
        # is a reference to this singleton, and ``execute_tool`` calls
        # ``dataclasses.asdict(ctx)`` which deep-copies every field as
        # part of step-argument serialization. Returning self preserves
        # identity — callers comparing executor references still see
        # the same object, and no tool result is ever mutated through
        # the copy.
        return self

    def _get_prior_source(self) -> _PriorSource:
        try:
            return self._prior_var.get()
        except LookupError:
            self._prior_var.set(_empty_prior_source)
            return _empty_prior_source

    def _get_prior(self) -> list[dict[str, object]]:
        return self._get_prior_source()()

    def _get_delta(self) -> list[dict[str, object]]:
        try:
            return self._delta_var.get()
        except LookupError:
            buf: list[dict[str, object]] = []
            self._delta_var.set(buf)
            return buf

    def append_messages(self, messages: list[dict[str, object]]) -> None:
        self._get_delta().extend(messages)

    def load_messages(self) -> list[dict[str, object]]:
        # Concat into a new list — callers treat the return as owned.
        # Same ownership contract as DirectExecutor.load_messages.
        return [*self._get_prior(), *self._get_delta()]

    def set_messages(self, messages: list[dict[str, object]]) -> None:
        # Back-compat: capture the list in a snapshotting closure and
        # install it as the prior source. Matches the pre-phase-2b
        # behaviour — the list stays heap-resident for the turn. Use
        # ``set_prior_source`` directly when a cheaper re-read is
        # available.
        snapshot = list(messages)
        self.set_prior_source(lambda: snapshot)

    def set_prior_source(self, source: _PriorSource) -> None:
        """Install a lazy source for prior-history messages.

        Each ``load_messages`` invokes ``source()`` to materialise
        prior fresh, so the history slice need not stay heap-resident
        between LLM iterations. Phase 2b of
        exoclaw/docs/memory-model.md.

        Also clears delta — sequential turns on the same task share
        the ContextVar binding; without the clear the next turn would
        see the prior turn's delta leaked into its ``load_messages``
        return.
        """
        self._prior_var.set(source)
        self._delta_var.set([])

    async def chat(
        self,
        provider: LLMProvider,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        _provider_var.set(provider)
        tools_json = json.dumps(tools) if tools else None
        result = await _chat_step(
            messages=list(messages),
            tools_json=tools_json,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return _dict_to_response(result)

    async def execute_tool(
        self,
        registry: ToolRegistry,
        name: str,
        params: dict[str, object],
        ctx: ToolContext | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> str:
        _registry_var.set(registry)
        ctx_data = dataclasses.asdict(ctx) if ctx else None

        # Bind a fresh intent buffer for this step. Tools running inside
        # the step append child-workflow start requests via
        # ``try_queue_child_workflow``; we drain and dispatch them after
        # the step body exits, when we are back in workflow context.
        buffer, token = _bind_intent_buffer()
        try:
            result = await _tool_step(
                name=name,
                params=dict(params),
                ctx_data=ctx_data,
            )
        finally:
            _release_intent_buffer(token)

        if buffer:
            await self._dispatch_intents(buffer)

        return result

    async def _dispatch_intents(self, intents: list[StartChildWorkflow]) -> None:
        """Start child workflows for queued intents from workflow context.

        Legal here because ``execute_tool`` runs from inside a parent
        ``@DBOS.workflow()`` (``run_durable_turn``) and the wrapping
        ``_tool_step`` has already exited. Step retries dispatch the same
        intents with the same ``workflow_id``s — DBOS dedups duplicate
        ``start_workflow_async`` calls with identical workflow IDs, so
        retry safety is automatic.
        """
        for intent in intents:
            workflow = _workflow_registry.get(intent.workflow_key)
            if workflow is None:
                raise RuntimeError(
                    f"No DBOS workflow registered for intent key "
                    f"{intent.workflow_key!r}. Did the providing module fail "
                    f"to import, or did you forget to call "
                    f"register_intent_workflow()?"
                )
            queue = _queue_registry.get(intent.workflow_key)
            with SetWorkflowID(intent.workflow_id):
                if queue is not None:
                    await queue.enqueue_async(workflow, **intent.kwargs)
                else:
                    await DBOS.start_workflow_async(workflow, **intent.kwargs)

    async def build_prompt(
        self,
        conversation: Conversation,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        **kwargs: list[str] | None,
    ) -> list[dict[str, object]]:
        messages = await conversation.build_prompt(
            session_id,
            message,
            channel=channel,
            chat_id=chat_id,
            media=media,
            plugin_context=plugin_context,
            **kwargs,
        )
        # Phase 2b auto-wire: when the Conversation exposes a sync
        # ``load_persisted_history(session_id)``, install a disk-backed
        # prior source so the history slice is re-read per LLM
        # iteration instead of being heap-resident for the whole turn.
        # Falls back to the snapshot path (``set_messages``) if the
        # conversation doesn't expose the method or if the slice
        # detection fails (deep-copying PromptBuilder, isolated mode,
        # etc.). See exoclaw/docs/memory-model.md phase 2b.
        loader = getattr(conversation, "load_persisted_history", None)
        if callable(loader) and not asyncio.iscoroutinefunction(loader):
            history_snapshot = loader(session_id)
            source = _build_lazy_prior_source(
                full=messages,
                history_snapshot=history_snapshot,
                reload_history=lambda: loader(session_id),
            )
            if source is not None:
                self.set_prior_source(source)
                return messages
        self.set_messages(messages)
        return messages

    async def append_message(
        self,
        conversation: Conversation,
        session_id: str,
        message: dict[str, object],
    ) -> None:
        """Per-message persistence, journaled via ``@DBOS.step``.

        The actual ``conversation.append`` call runs inside
        ``_append_message_step`` so its completion is written to the
        DBOS journal — recovery then skips it rather than re-appending
        the same message to the session JSONL (which would double-
        write, since ``DefaultConversation.append`` is not idempotent
        at the filesystem level).

        Matches PR #44's posture for the final-reply send: accept
        at-least-once semantics on the window between the step body
        completing and the journal committing; that window is ~ms and
        the resulting duplicate JSONL line is recoverable by the
        session loader while a silent drop would not be.
        """
        _conversation_var.set(conversation)
        await _append_message_step(session_id, message)

    async def post_turn(
        self,
        conversation: Conversation,
        session_id: str,
    ) -> None:
        """End-of-turn hooks, journaled."""
        _conversation_var.set(conversation)
        await _post_turn_step(session_id)

    async def record(
        self,
        conversation: Conversation,
        session_id: str,
        new_messages: list[dict[str, object]],
    ) -> None:
        await conversation.record(session_id, new_messages)

    async def clear(
        self,
        conversation: Conversation,
        session_id: str,
    ) -> bool:
        return await conversation.clear(session_id)

    async def run_turn(
        self,
        loop: Any,
        session_id: str,
        message: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
        on_progress: Any = None,
        model: str | None = None,
        publish_response: bool = False,
        **kwargs: Any,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Run a full agent turn inside a DBOS workflow.

        Called by AgentLoop.process_turn() when the executor provides this
        method. Sets the loop context (for crash recovery) and wraps the
        turn in a @DBOS.workflow() so it is recoverable on restart.

        When ``publish_response`` is True the workflow also publishes the
        final reply to the bus via a durable step, so the send survives
        OOM kills mid-turn. Callers who read the returned content
        directly (e.g. ``process_direct`` for subagents) leave it False.
        """
        from .turn import run_durable_turn, set_loop_context

        # Ensure the loop reference is available for DBOS recovery
        set_loop_context(loop)

        from . import turn

        turn._on_progress = on_progress

        wfid = f"turn:{session_id}:{uuid7().hex}"
        with SetWorkflowID(wfid):
            return await run_durable_turn(
                session_id,
                message,
                channel=channel or "",
                chat_id=chat_id or "",
                media=media,
                plugin_context=plugin_context,
                model=model,
                publish_response=publish_response,
            )

    async def run_hook(
        self,
        fn: Callable[..., Awaitable[object]],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        return await fn(*args, **kwargs)

    async def mint_turn_id(self) -> str:
        """Mint a replay-safe per-turn id via a DBOS step.

        ``AgentLoop._process_turn_inline`` calls this once at the top
        of every turn (added in exoclaw 0.15) and binds the result as
        ``turn.id`` in structlog's contextvars. Wrapping ``uuid7()``
        in a ``@DBOS.step()`` is what makes it survive workflow
        recovery: on the first execution DBOS records the value to
        the step journal, and on replay the body is skipped and the
        recorded value is returned. Without this, a recovered
        workflow would rebind a new ``turn.id`` and downstream log
        lines emitted before vs after the crash would land in two
        different ``turn.root_id`` buckets.
        """
        return await _mint_turn_id_step()
