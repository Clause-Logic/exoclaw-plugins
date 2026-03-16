"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .helpers import detect_image_mime
from .memory import MemoryStore
from .protocols import MemoryBackend
from .skills import SkillsLoader

_RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
_COMPACTION_MARKER = "[compacted — tool output removed to free context]"
_CHARS_PER_TOKEN = 3  # conservative estimate


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
        result[i] = {**result[i], "content": _COMPACTION_MARKER}

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


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    def __init__(
        self,
        workspace: Path,
        memory: MemoryBackend | None = None,
        context_window: int = 128_000,
        skill_packages: list[str] | None = None,
    ):
        self.workspace = workspace
        self.skills = SkillsLoader(workspace, skill_packages=skill_packages)
        self.memory: MemoryBackend = memory if memory is not None else MemoryStore(workspace)
        self.context_window = context_window

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        extra_context: str | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        if extra_context:
            parts.append(f"# Retrieved Context\n\n{extra_context}")

        always_skills = self.skills.get_always_skills()
        extra_skills = [s for s in (skill_names or []) if s not in always_skills]
        active_skills = always_skills + extra_skills
        if active_skills:
            active_content = self.skills.load_skills_for_context(active_skills)
            if active_content:
                parts.append(f"# Active Skills\n\n{active_content}")

        bootstrap_hooks = self.skills.get_bootstrap_injections()
        if bootstrap_hooks:
            parts.append("\n\n".join(bootstrap_hooks))

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities.
- Skills with a file path: read the SKILL.md file using read_file to learn how to use them.
- Skills with source="package": already loaded — just use the tools they describe. Do NOT try to read_file them.
- Skills with available="false" need dependencies installed first.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

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
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
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
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged: str | list[dict[str, Any]] = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        messages = [
            {"role": "system", "content": self.build_system_prompt(skill_names, extra_context=extra_context)},
            *history,
            {"role": "user", "content": merged},
        ]

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
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
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
