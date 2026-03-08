"""
Sanitization utilities for provider package authors.

These are optional helpers — provider packages may import them to handle
common edge cases (empty content, unknown message keys) that cause 400
errors with most LLM APIs.
"""

from typing import Any


def sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace empty text content that causes provider 400 errors."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")

        if isinstance(content, str) and not content:
            clean = dict(msg)
            clean["content"] = (
                None if (msg.get("role") == "assistant" and msg.get("tool_calls"))
                else "(empty)"
            )
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


def sanitize_message_keys(
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
