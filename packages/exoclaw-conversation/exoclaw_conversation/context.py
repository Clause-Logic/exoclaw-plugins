"""Context builder for assembling agent prompts."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from exoclaw._compat import IS_MICROPYTHON, Path, guess_image_mime, platform_summary

from .helpers import detect_image_mime
from .memory import MemoryStore
from .protocols import MemoryBackend
from .skills import SkillsLoader

if TYPE_CHECKING:
    # collections.abc isn't available on every MicroPython build; gating
    # the import behind TYPE_CHECKING keeps these as string annotations at
    # runtime while still giving CPython type-checkers what they need.
    from collections.abc import Awaitable, Callable


def _b64encode(data: bytes) -> str:
    """Base64-encode bytes to ASCII text. Cross-runtime: CPython
    uses ``base64.b64encode``; MicroPython uses ``binascii.b2a_base64``
    (the ``base64`` module isn't part of the unix-port standard
    library)."""
    if IS_MICROPYTHON:
        import binascii

        return binascii.b2a_base64(data, newline=False).decode("ascii")
    import base64

    return base64.b64encode(data).decode("ascii")


_RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
_COMPACTION_MARKER = "[compacted — tool output removed to free context]"
_RECOVERY_HARD_CLEAR_MARKER = "[Old tool result content cleared]"
_RECOVERY_SUMMARY_PREFIX = (
    "Summary of prior conversation (older messages were summarized to free context):\n\n"
)
_CHARS_PER_TOKEN = 3  # conservative estimate
_DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count from messages using character heuristic."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("text", "")))
        # Count tool call arguments too
        for tc in m.get("tool_calls", []):
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                if isinstance(fn, dict):
                    total += len(str(fn.get("arguments", "")))
    return total // _CHARS_PER_TOKEN


def compact_tool_results(
    messages: list[dict[str, Any]],
    context_window: int,
    headroom: float = 0.75,
) -> list[dict[str, Any]]:
    """Replace old tool results with compaction marker when context exceeds budget.

    Compacts from oldest to newest, skipping the most recent tool results
    (within the last 4 messages) to preserve the active conversation.
    """
    budget = int(context_window * headroom)
    current = _estimate_tokens(messages)
    if current <= budget:
        return messages

    # Find tool results eligible for compaction (skip last 4 non-system messages)
    non_system = [i for i, m in enumerate(messages) if m.get("role") != "system"]
    protected = set(non_system[-4:]) if len(non_system) >= 4 else set(non_system)

    compactable = []
    for i, m in enumerate(messages):
        if i in protected:
            continue
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            content = m["content"]
            if content != _COMPACTION_MARKER and len(content) > 100:
                compactable.append(i)

    # Compact oldest first until under budget
    result = list(messages)
    for i in compactable:
        if _estimate_tokens(result) <= budget:
            break
        # Avoid ``{**result[i], ...}`` — MicroPython 1.27 doesn't
        # support PEP 448 dict-unpacking in dict literals. Plain
        # copy + assign works on both runtimes.
        copy = dict(result[i])
        copy["content"] = _COMPACTION_MARKER
        result[i] = copy

    return result


def drop_oldest_half(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Emergency compression: keep system prompt + last half of conversation.

    Repairs orphaned tool results that lost their parent assistant message.
    """
    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    half = len(non_system) // 2
    kept = non_system[half:]

    # Repair: drop tool results without a preceding assistant tool_call
    repaired: list[dict[str, Any]] = []
    for m in kept:
        if m.get("role") == "tool":
            tool_call_id = m.get("tool_call_id")
            has_parent = any(
                r.get("role") == "assistant"
                and any(
                    tc.get("id") == tool_call_id
                    for tc in (r.get("tool_calls") or [])
                    if isinstance(tc, dict)
                )
                for r in repaired
            )
            if not has_parent:
                continue
        repaired.append(m)

    return system + repaired


def truncate_oldest_tool_results(
    messages: list[dict[str, Any]],
    target_tokens: int,
    *,
    placeholder: str = _RECOVERY_HARD_CLEAR_MARKER,
) -> tuple[list[dict[str, Any]], int]:
    """Recovery-time tool-result hard-clear: replace oldest tool results until under budget.

    Unlike ``compact_tool_results``, this does NOT protect the most recent
    messages — it's used after a context-window-exceeded error, when the
    preventive compaction was insufficient and we need to free space at any
    cost. Mirrors openclaw's hard-clear recovery step.

    Returns ``(new_messages, cleared_count)``. ``cleared_count`` is 0 when
    nothing was eligible — callers can use that to detect when recovery
    can't make further progress.
    """
    if _estimate_tokens(messages) <= target_tokens:
        return messages, 0

    # Eligible: any tool message with substantive content (not already cleared)
    eligible = []
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if content == placeholder or content == _COMPACTION_MARKER:
            continue
        if len(content) <= 100:
            continue
        eligible.append(i)

    result = list(messages)
    cleared = 0
    for i in eligible:
        if _estimate_tokens(result) <= target_tokens:
            break
        copy = dict(result[i])
        copy["content"] = placeholder
        result[i] = copy
        cleared += 1

    return result, cleared


async def summarize_old_chunks(
    messages: list[dict[str, Any]],
    target_tokens: int,
    summarizer: Callable[[list[dict[str, Any]]], Awaitable[str]],
    *,
    keep_recent: int = 4,
    summarizer_max_input_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Recovery-time summarization: replace older history with one summary message.

    Eligible messages are non-system messages excluding the last
    ``keep_recent`` non-system messages (the active conversation that the
    model needs to act on). The eligible block is passed to ``summarizer``
    which returns a summary string; the block is replaced with a single
    user-role summary message.

    If the eligible block exceeds ``summarizer_max_input_tokens``, only the
    oldest portion that fits is summarized (caller should re-invoke for
    further reductions). Default cap is ``target_tokens // 2``.

    Returns ``(new_messages, summarized)``. ``summarized=False`` means there
    was nothing eligible (e.g. only ``keep_recent`` messages remain) — the
    caller should fall back to a different strategy.

    Repairs orphaned tool results in both the summarized chunk (their parent
    assistant call disappears into the summary) and the kept tail.
    """
    if _estimate_tokens(messages) <= target_tokens:
        return messages, False

    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= keep_recent:
        return messages, False

    eligible = non_system[:-keep_recent] if keep_recent > 0 else list(non_system)
    tail = non_system[-keep_recent:] if keep_recent > 0 else []

    # Cap input to summarizer to keep its own context safe
    cap = (
        summarizer_max_input_tokens
        if summarizer_max_input_tokens is not None
        else target_tokens // 2
    )
    if cap > 0:
        chunk: list[dict[str, Any]] = []
        # Greedily take from oldest until we'd exceed cap
        for m in eligible:
            trial = chunk + [m]
            if _estimate_tokens(trial) > cap and chunk:
                break
            chunk.append(m)
        remaining_eligible = eligible[len(chunk) :]
    else:
        chunk = list(eligible)
        remaining_eligible = []

    if not chunk:
        return messages, False

    summary_text = await summarizer(chunk)
    if not summary_text:
        return messages, False

    summary_msg: dict[str, Any] = {
        "role": "user",
        "content": _RECOVERY_SUMMARY_PREFIX + summary_text,
    }

    rebuilt = system + [summary_msg] + remaining_eligible + tail

    # Repair: drop tool results whose parent assistant tool_call was
    # absorbed into the summary.
    repaired: list[dict[str, Any]] = []
    for m in rebuilt:
        if m.get("role") == "tool":
            tool_call_id = m.get("tool_call_id")
            has_parent = any(
                r.get("role") == "assistant"
                and any(
                    tc.get("id") == tool_call_id
                    for tc in (r.get("tool_calls") or [])
                    if isinstance(tc, dict)
                )
                for r in repaired
            )
            if not has_parent:
                continue
        repaired.append(m)

    return repaired, True


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    def __init__(
        self,
        workspace: Path,
        memory: MemoryBackend | None = None,
        context_window: int = 128_000,
        skill_packages: list[str] | None = None,
        builtin_skills_dir: Path | None = None,
        allowed_skills: list[str] | None = None,
    ):
        self.workspace = workspace
        self.skills = SkillsLoader(
            workspace,
            builtin_skills_dir=builtin_skills_dir,
            skill_packages=skill_packages,
            allowed_names=allowed_skills,
        )
        self.memory: MemoryBackend = memory if memory is not None else MemoryStore(workspace)
        self.context_window = context_window
        self._active_optional_tools: set[str] = set()

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        extra_context: str | None = None,
        isolated: bool = False,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills.

        ``isolated=True`` returns a minimal prompt containing a short
        functional preamble, the active skills' content, and any
        caller-provided ``extra_context``. It skips identity, bootstrap
        files (AGENTS.md / SOUL.md / USER.md / TOOLS.md / IDENTITY.md),
        long-term memory, bootstrap hooks, and the skills summary. The
        goal is "one call, one job" — use this when the caller is a
        deterministic script invoking the agent as a pure function
        (e.g. per-item feed enrichment). With the persona / memory /
        skill-menu removed, small open-weight models like gpt-oss stop
        contradicting the skill's directives under the weight of the
        broader bot context. ``extra_context`` is preserved because the
        caller explicitly asked for it — it's turn-volatile data they
        need the model to see.
        """
        always_skills = self.skills.get_always_skills()
        extra_skills = [s for s in (skill_names or []) if s not in always_skills]
        active_skills = always_skills + extra_skills
        self._active_optional_tools = self.skills.get_tools_for_skills(active_skills)

        if isolated:
            parts: list[str] = [
                "You are a worker. Follow the instructions below exactly. "
                "Do not invoke capabilities not explicitly requested."
            ]
            if active_skills:
                active_content = self.skills.load_skills_for_context(active_skills)
                if active_content:
                    parts.append(active_content)
            if extra_context:
                parts.append(f"# Retrieved Context\n\n{extra_context}")
            return "\n\n---\n\n".join(parts)

        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        if extra_context:
            parts.append(f"# Retrieved Context\n\n{extra_context}")

        if active_skills:
            active_content = self.skills.load_skills_for_context(active_skills)
            if active_content:
                parts.append(f"# Active Skills\n\n{active_content}")

        bootstrap_hooks = self.skills.get_bootstrap_injections()
        if bootstrap_hooks:
            parts.append("\n\n".join(bootstrap_hooks))

        # Show all enabled skills so the agent can load_skill any of them.
        # Active skills are already injected above; the summary gives the agent
        # names + descriptions for everything it *could* load on demand.
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills are available. To activate a skill and its tools, call the load_skill tool with the skill name.
Skills with available="false" need dependencies installed first.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def get_active_optional_tools(self) -> set[str]:
        """Return the optional tool names activated by the current turn's skills.

        Updated every time :meth:`build_system_prompt` is called.  Pass this
        method as ``optional_tools_fn`` when constructing ``AgentLoop`` so the
        loop surfaces the right tools per turn without knowing about skills.
        """
        return self._active_optional_tools

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        runtime = platform_summary()

        return f"""# exoclaw

You are exoclaw, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        # Format manually instead of ``strftime`` — micropython-lib's
        # datetime doesn't ship ``strftime`` and ``mip install
        # datetime`` on a chip would still pull the upstream impl.
        # ``isoformat`` + ``weekday`` are available on both runtimes.
        now = datetime.now()
        iso = now.isoformat()
        # ``isoformat`` returns ``YYYY-MM-DDTHH:MM:SS[.ffffff][+TZ]``.
        # Split on ``T`` and trim seconds for the human-facing block.
        date_part, _, time_part = iso.partition("T")
        hh_mm = time_part[:5]
        weekday = _DAY_NAMES[now.weekday()]
        # Skip timezone display — ``time.strftime`` isn't on MP and
        # boards typically run on UTC straight from NTP anyway. The
        # ISO date already implies UTC for the chip path.
        lines = [f"Current Time: {date_part} {hh_mm} ({weekday}) UTC"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return _RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        extra_context: str | None = None,
        turn_context: list[str] | None = None,
        isolated: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call.

        ``isolated=True`` strips session history and the runtime-context
        metadata prefix so the LLM sees ``[system, user]`` — a
        pure-function invocation without persona/memory/history
        carryover. Any provided ``turn_context`` is still prepended to
        ``current_message`` inside that user message (it's caller-
        supplied turn-volatile data, not implicit history), so the
        caller — typically a deterministic orchestrator hitting
        ``/agent/call`` — is responsible for putting everything the
        model needs into ``current_message`` and/or ``turn_context``.
        """
        # Prepend turn_context to the user message so the system prompt stays
        # stable across turns and benefits from prompt caching. Unlike
        # plugin_context (which goes into the system prompt via extra_context),
        # turn_context is per-turn volatile data (e.g. A-MEM retrieved notes)
        # that belongs alongside the user message, not in the cached prefix.
        effective_message = current_message
        if turn_context:
            ctx_block = "\n\n".join(turn_context)
            effective_message = f"{ctx_block}\n\n{current_message}"

        user_content = self._build_user_content(effective_message, media)

        if isolated:
            merged: str | list[dict[str, Any]] = user_content
            effective_history: list[dict[str, Any]] = []
        else:
            runtime_ctx = self._build_runtime_context(channel, chat_id)
            # Merge runtime context and user content into a single user message
            # to avoid consecutive same-role messages that some providers reject.
            if isinstance(user_content, str):
                merged = f"{runtime_ctx}\n\n{user_content}"
            else:
                merged = [{"type": "text", "text": runtime_ctx}] + user_content
            effective_history = history

        # MicroPython 1.27 doesn't support PEP 448 list-unpacking
        # inside list literals (``[a, *xs, b]``). Build the list
        # via append/extend instead — same shape on both runtimes.
        # Annotate the list explicitly because the user-message
        # ``content`` can be a ``str`` OR a ``list[dict]`` (image
        # attachments) and ty would otherwise infer the narrower
        # ``list[dict[str, str]]`` from the literal.
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    extra_context=extra_context,
                    isolated=isolated,
                ),
            },
        ]
        messages.extend(effective_history)
        messages.append({"role": "user", "content": merged})

        # Compact old tool results if approaching context budget
        return compact_tool_results(messages, self.context_window)

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or guess_image_mime(path)
            if not mime or not mime.startswith("image/"):
                continue
            b64 = _b64encode(raw)
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks
        messages.append(msg)
        return messages
