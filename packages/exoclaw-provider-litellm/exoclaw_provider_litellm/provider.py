"""LiteLLM provider implementation for exoclaw."""

import hashlib
import os
import secrets
import string
import time
from typing import Any

import litellm
from litellm import acompletion
from loguru import logger

from exoclaw.providers.types import LLMResponse, ToolCallRequest

# Standard chat-completion message keys.
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"})
_ANTHROPIC_EXTRA_KEYS = frozenset({"thinking_blocks"})
_ALNUM = string.ascii_letters + string.digits


def _short_tool_id() -> str:
    """Generate a 9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace empty text content that causes provider 400 errors."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")

        if isinstance(content, str) and not content:
            clean = dict(msg)
            clean["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
            result.append(clean)
            continue

        if isinstance(content, list):
            filtered = [
                item for item in content
                if not (
                    isinstance(item, dict)
                    and item.get("type") in ("text", "input_text", "output_text")
                    and not item.get("text")
                )
            ]
            if len(filtered) != len(content):
                clean = dict(msg)
                if filtered:
                    clean["content"] = filtered
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    clean["content"] = None
                else:
                    clean["content"] = "(empty)"
                result.append(clean)
                continue

        if isinstance(content, dict):
            clean = dict(msg)
            clean["content"] = [content]
            result.append(clean)
            continue

        result.append(msg)
    return result


def _sanitize_request_messages(
    messages: list[dict[str, Any]],
    allowed_keys: frozenset[str],
) -> list[dict[str, Any]]:
    """Keep only provider-safe message keys and normalize assistant content."""
    sanitized = []
    for msg in messages:
        clean = {k: v for k, v in msg.items() if k in allowed_keys}
        if clean.get("role") == "assistant" and "content" not in clean:
            clean["content"] = None
        sanitized.append(clean)
    return sanitized


def _normalize_tool_call_id(tool_call_id: Any) -> Any:
    """Normalize tool_call_id to a provider-safe 9-char alphanumeric form."""
    if not isinstance(tool_call_id, str):
        return tool_call_id
    if len(tool_call_id) == 9 and tool_call_id.isalnum():
        return tool_call_id
    return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]


def _is_anthropic(model: str) -> bool:
    """Return True when the model is an Anthropic/Claude model."""
    lower = model.lower()
    return "claude" in lower or lower.startswith("anthropic/")


class LiteLLMProvider:
    """
    LLM provider using LiteLLM for multi-provider support.

    Implements the exoclaw LLMProvider protocol without inheriting from any
    exoclaw class.

    Supports OpenRouter, Anthropic, OpenAI, Gemini, and many other providers
    through a unified interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        if api_key and api_base:
            # Custom / gateway endpoint — set as OpenAI-compatible
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        elif api_key:
            # Best-effort: set common env vars so LiteLLM can pick up the key
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            os.environ.setdefault("OPENROUTER_API_KEY", api_key)

        if api_base:
            litellm.api_base = api_base

        litellm.suppress_debug_info = True
        litellm.drop_params = True

        if callbacks_env := os.environ.get("LITELLM_CALLBACKS"):
            litellm.callbacks = [c.strip() for c in callbacks_env.split(",") if c.strip()]

        self._llm_logging = os.environ.get("LLM_LOGGING", "").lower() == "true"
        self._llm_log_truncate = int(os.environ.get("LLM_LOG_TRUNCATE", "500"))

    def _sanitize_messages(
        self,
        messages: list[dict[str, Any]],
        extra_keys: frozenset[str] = frozenset(),
    ) -> list[dict[str, Any]]:
        """Strip non-standard keys and normalize tool call IDs."""
        allowed = _ALLOWED_MSG_KEYS | extra_keys
        sanitized = _sanitize_request_messages(messages, allowed)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return id_map.setdefault(value, _normalize_tool_call_id(value))

        for clean in sanitized:
            if isinstance(clean.get("tool_calls"), list):
                normalized_tool_calls = []
                for tc in clean["tool_calls"]:
                    if not isinstance(tc, dict):
                        normalized_tool_calls.append(tc)
                        continue
                    tc_clean = dict(tc)
                    tc_clean["id"] = map_id(tc_clean.get("id"))
                    normalized_tool_calls.append(tc_clean)
                clean["tool_calls"] = normalized_tool_calls

            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_id(clean["tool_call_id"])
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request via LiteLLM."""
        resolved_model = model or self.default_model
        extra_keys = _ANTHROPIC_EXTRA_KEYS if _is_anthropic(resolved_model) else frozenset()

        max_tokens = max(1, max_tokens)

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": self._sanitize_messages(
                _sanitize_empty_content(messages), extra_keys=extra_keys
            ),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.api_base:
            kwargs["api_base"] = self.api_base

        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if self._llm_logging:
            logger.info(
                "LLM request: model={} messages={} tools={}",
                resolved_model,
                len(kwargs["messages"]),
                len(tools) if tools else 0,
            )
            for msg in kwargs["messages"]:
                role = msg.get("role", "?")
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                text = str(content).replace("\n", "\\n")
                if self._llm_log_truncate >= 0:
                    text = text[:self._llm_log_truncate]
                logger.info("  [{}] {}", role, text)

        try:
            t0 = time.monotonic()
            response = await acompletion(**kwargs)
            elapsed = time.monotonic() - t0

            if self._llm_logging:
                usage = response.usage
                choice = response.choices[0]
                content = getattr(choice.message, "content", None) or ""
                tool_calls = getattr(choice.message, "tool_calls", None) or []
                cached_tokens = 0
                if usage:
                    details = getattr(usage, "prompt_tokens_details", None)
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0
                    if not cached_tokens:
                        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                cache_created = getattr(usage, "cache_creation_input_tokens", 0) if usage else 0
                logger.info(
                    "LLM response: model={} tokens={}+{}={} cached={} cache_created={} duration={:.2f}s finish={} tools={}",
                    resolved_model,
                    usage.prompt_tokens if usage else "?",
                    usage.completion_tokens if usage else "?",
                    usage.total_tokens if usage else "?",
                    cached_tokens,
                    cache_created,
                    elapsed,
                    choice.finish_reason,
                    [tc.function.name for tc in tool_calls],
                )

            return self._parse_response(response)
        except Exception as e:
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        content = message.content
        finish_reason = choice.finish_reason

        # Some providers split content and tool_calls across multiple choices.
        raw_tool_calls = []
        for ch in response.choices:
            msg = ch.message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                raw_tool_calls.extend(msg.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and msg.content:
                content = msg.content

        tool_calls = []
        for tc in raw_tool_calls:
            import json_repair
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)

            tool_calls.append(ToolCallRequest(
                id=_short_tool_id(),
                name=tc.function.name,
                arguments=args,
            ))

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        reasoning_content = getattr(message, "reasoning_content", None) or None
        thinking_blocks = getattr(message, "thinking_blocks", None) or None

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
