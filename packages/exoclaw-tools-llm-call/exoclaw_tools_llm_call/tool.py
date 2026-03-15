"""LLM call tool — single-shot LLM call with Jinja2 templated prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jinja2
from exoclaw.agent.tools.protocol import ToolBase
from exoclaw.providers.protocol import LLMProvider
from exoclaw.providers.types import ResponseFormat


def _file(path: str) -> str:
    """Jinja2 global: read a file and return its contents."""
    p = Path(path)
    if not p.exists():
        return f"[file not found: {path}]"
    return p.read_text()


def _render(template: str, vars: dict[str, Any] | None = None) -> str:
    """Render a Jinja2 template with file() global and optional vars."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.globals["file"] = _file
    tmpl = env.from_string(template)
    return tmpl.render(**(vars or {}))


def _build_response_format(schema: dict[str, Any]) -> ResponseFormat:
    """Build a ResponseFormatJSONSchema from a user-provided schema dict."""
    name = schema.pop("name", "response")
    strict = schema.pop("strict", None)
    json_schema: dict[str, Any] = {"name": name, "schema": schema}
    if strict is not None:
        json_schema["strict"] = strict
    return {"type": "json_schema", "json_schema": json_schema}


class LLMCallTool(ToolBase):
    """Single-shot LLM call with Jinja2 templated prompts.

    Configuration (constructor):
        provider:       LLMProvider instance
        allowed_models: list of model IDs the agent is allowed to use
        default_model:  fallback model when none specified

    Call-time (from the agent):
        prompt:   Jinja2 template string — use {{ var }}, {{ file('/path') }}
        vars:     dict of template variables (optional)
        model:    model ID to use (must be in allowed_models)
        schema:   JSON Schema for structured output (optional)
        output:   file path to write result to (optional, otherwise returned inline)
    """

    def __init__(
        self,
        provider: LLMProvider,
        allowed_models: list[str] | None = None,
        default_model: str | None = None,
    ) -> None:
        self._provider = provider
        self._allowed_models = allowed_models or []
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "llm_call"

    @property
    def description(self) -> str:
        models = ", ".join(self._allowed_models) if self._allowed_models else "any"
        return (
            "Make a single LLM call with a Jinja2 templated prompt. "
            "No tools or agent loop — just prompt in, text out. "
            "Use {{ var }} for variable substitution and {{ file('/path') }} "
            "to inline file contents. "
            f"Allowed models: {models}. "
            "Set schema for structured JSON output. "
            "Set output to a file path to write the result to disk "
            "instead of returning it inline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Jinja2 template for the prompt. "
                        "Use {{ var }} for substitution, {{ file('/path') }} to read files."
                    ),
                },
                "vars": {
                    "type": "object",
                    "description": "Template variables (optional)",
                },
                "model": {
                    "type": "string",
                    "description": (
                        f"Model to use. Allowed: {', '.join(self._allowed_models) or 'any'}. "
                        f"Default: {self._default_model or 'provider default'}"
                    ),
                },
                "schema": {
                    "type": "object",
                    "description": (
                        "JSON Schema for structured output. The LLM response will conform "
                        "to this schema. Include 'name' (optional, default 'response') and "
                        "standard JSON Schema properties (type, properties, required, etc.)."
                    ),
                },
                "output": {
                    "type": "string",
                    "description": "File path to write output to. If omitted, result returned inline.",
                },
            },
            "required": ["prompt"],
        }

    async def execute(
        self,
        prompt: str,
        vars: dict[str, Any] | None = None,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        output: str | None = None,
        **kwargs: Any,
    ) -> str:
        # Resolve model
        use_model = model or self._default_model
        if self._allowed_models and use_model and use_model not in self._allowed_models:
            return (
                f"Error: Model '{use_model}' not allowed. "
                f"Allowed: {', '.join(self._allowed_models)}"
            )

        # Render template
        try:
            rendered = _render(prompt, vars)
        except jinja2.TemplateError as e:
            return f"Error rendering template: {e}"

        # Call LLM
        messages = [{"role": "user", "content": rendered}]
        chat_kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": [],
            "model": use_model,
        }
        if schema:
            chat_kwargs["response_format"] = _build_response_format(dict(schema))

        try:
            response = await self._provider.chat(**chat_kwargs)
            text = response.content or ""
        except Exception as e:
            return f"Error calling LLM: {e}"

        # Write to file or return inline
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(text)
            return json.dumps({"output_path": output, "chars": len(text)})

        return text
