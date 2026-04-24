"""Direct-httpx OpenAI-compatible provider for exoclaw.

The key property: the request body is emitted as an ``AsyncIterable[bytes]``
into ``httpx.AsyncClient.post(url, content=_stream_body(...))`` so the full
JSON never materializes as one contiguous string. That's the peak-memory
reduction the ``docs/memory-model.md`` Step B plan is aimed at, delivered
without forking or upstreaming LiteLLM.

Per-model routing and fallback: each model name maps to exactly one
``Deployment`` (base URL + API key + optional extra headers), and each
model has an optional fallback chain. A retryable error on the primary
walks the chain; non-retryable errors (auth, 400-class) bubble up.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import string
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import json_repair
import structlog
from exoclaw.providers.types import (
    ContextWindowExceededError,
    LLMResponse,
    ResponseFormat,
    ToolCallRequest,
)

logger = structlog.get_logger()

_ALNUM = string.ascii_letters + string.digits

# Response-retryable HTTP status codes. 408/425 join the 5xx/429 set for
# safety — some providers emit them on transient queue pressure.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _short_tool_id() -> str:
    """9-char alnum id. OpenAI/Anthropic accept arbitrary strings for
    ``tool_calls[].id``; some providers (Mistral) reject longer/punctuated
    ids, so we keep to a safe subset that everything accepts."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


@dataclass(frozen=True)
class Deployment:
    """A single model → endpoint binding.

    ``extra_headers`` is merged into the request headers on every call;
    ``extra_body`` is merged into the JSON body at the top level (e.g. for
    OpenRouter's ``provider`` routing object, or a custom ``transforms``
    flag). Both are read-only after construction.
    """

    base_url: str
    api_key: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


class OpenAIStreamingProvider:
    """Direct-httpx provider speaking OpenAI chat-completions protocol.

    Implements the exoclaw ``LLMProvider`` protocol.
    """

    def __init__(
        self,
        default_model: str,
        deployments: dict[str, Deployment],
        fallbacks: dict[str, list[str]] | None = None,
        *,
        request_timeout: float = 120.0,
        stream_ttft_timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if default_model not in deployments:
            raise ValueError(
                f"default_model {default_model!r} not in deployments: {sorted(deployments)}"
            )
        for primary, chain in (fallbacks or {}).items():
            for fb in chain:
                if fb not in deployments:
                    raise ValueError(
                        f"fallback {fb!r} (for primary {primary!r}) not in deployments"
                    )

        self.default_model = default_model
        self._deployments = dict(deployments)
        self._fallbacks = {k: list(v) for k, v in (fallbacks or {}).items()}
        self._request_timeout = request_timeout
        self._stream_ttft_timeout = stream_ttft_timeout

        # Allow dependency injection for tests. In production one client
        # is reused across all requests so httpx can pool connections.
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=request_timeout)

        self._llm_logging = os.environ.get("LLM_LOGGING", "").lower() == "true"
        self._llm_log_truncate = int(os.environ.get("LLM_LOG_TRUNCATE", "500"))

    def get_default_model(self) -> str:
        return self.default_model

    async def close(self) -> None:
        """Close the underlying httpx client. Safe to call multiple times."""
        if self._owns_client:
            await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Send a chat completion. Walks the fallback chain on retryable
        errors; raises the last error if every model fails."""
        resolved = model or self.default_model
        chain = [resolved] + self._fallbacks.get(resolved, [])
        last_err: Exception | None = None

        for candidate in chain:
            deployment = self._deployments.get(candidate)
            if deployment is None:
                # Shouldn't happen — __init__ validates — but belt-and-braces.
                raise ValueError(f"no deployment for model {candidate!r}")

            try:
                return await self._chat_once(
                    deployment=deployment,
                    model=candidate,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                )
            except _RetryableError as e:
                # ``__cause__`` can be any BaseException; narrow to
                # Exception for the type checker and fall through to the
                # _RetryableError itself if the cause isn't a plain
                # Exception (it always is in practice — we only chain
                # from httpx errors).
                cause = e.__cause__
                last_err = cause if isinstance(cause, Exception) else e
                logger.warning(
                    "llm_fallback",
                    **{
                        "llm.model": candidate,
                        "llm.next": chain[chain.index(candidate) + 1]
                        if chain.index(candidate) + 1 < len(chain)
                        else None,
                        "error": str(last_err),
                    },
                )
                continue
            except ContextWindowExceededError:
                # Context-window errors don't get better on another model
                # in the same series. Surface immediately.
                raise

        # Exhausted chain.
        assert last_err is not None
        raise last_err

    async def _chat_once(
        self,
        *,
        deployment: Deployment,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        response_format: ResponseFormat | None,
    ) -> LLMResponse:
        """Single non-retried request to ``deployment`` for ``model``.
        Raises ``_RetryableError`` on status/network errors the caller
        should treat as fallback-eligible. Other errors bubble up."""
        url = deployment.base_url.rstrip("/") + "/chat/completions"
        headers = self._build_headers(deployment)

        # Request body is assembled as a dict of metadata + the messages
        # list, but we stream it out chunk-per-message so the full JSON
        # never lives in memory as one string. See ``_stream_body``.
        body_head: dict[str, Any] = {
            "model": model,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body_head["tools"] = tools
            body_head["tool_choice"] = "auto"
        if reasoning_effort:
            body_head["reasoning_effort"] = reasoning_effort
        if response_format:
            body_head["response_format"] = response_format
        extra_body = self._resolve_extra_body(deployment)
        for k, v in extra_body.items():
            body_head.setdefault(k, v)

        if self._llm_logging:
            self._log_request(model, messages, tools)

        t0 = time.monotonic()
        try:
            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                content=_stream_body(body_head, messages),
                timeout=self._request_timeout,
            ) as resp:
                if resp.status_code in _RETRYABLE_STATUS:
                    # Read the body for logs but don't raise the default
                    # ``HTTPStatusError`` — we want to signal retryable
                    # specifically, so the fallback loop engages.
                    text = await resp.aread()
                    raise _RetryableError(f"status {resp.status_code}: {text[:500]!r}")
                if resp.status_code == 400 and await _is_context_window_error(resp):
                    raise ContextWindowExceededError("Prompt exceeds model context window")
                resp.raise_for_status()

                response = await self._consume_sse_stream(resp)

        except httpx.ConnectError as e:
            raise _RetryableError(f"connect error: {e}") from e
        except httpx.ReadTimeout as e:
            raise _RetryableError(f"read timeout: {e}") from e
        except httpx.WriteTimeout as e:
            raise _RetryableError(f"write timeout: {e}") from e
        except asyncio.TimeoutError as e:
            raise _RetryableError(f"timeout: {e}") from e

        elapsed = time.monotonic() - t0
        if self._llm_logging:
            self._log_response(model, response, elapsed)
        return response

    def _build_headers(self, deployment: Deployment) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {deployment.api_key}",
            "Content-Type": "application/json",
        }
        for k, v in deployment.extra_headers.items():
            headers.setdefault(k, v)
        return headers

    def _resolve_extra_body(self, deployment: Deployment) -> dict[str, Any]:
        """Merge the deployment's ``extra_body`` with the session-affinity
        ``user`` hint (used by OpenRouter to keep prompt caches warm on
        the same upstream provider across turns)."""
        extra_body: dict[str, Any] = dict(deployment.extra_body)
        if "user" not in extra_body:
            ctx = structlog.contextvars.get_contextvars()
            session_key = ctx.get("session.key")
            if session_key:
                extra_body["user"] = str(session_key)
        return extra_body

    async def _consume_sse_stream(self, resp: httpx.Response) -> LLMResponse:
        """Accumulate SSE chunks into a single ``LLMResponse``.

        We need the full response anyway (the turn loop wants tool_calls
        and finish_reason materialized) — streaming is purely for the
        server-side TTFT and incremental-decode wins. The memory benefit
        of this provider lives in the request path, not the response
        path: response bodies are much smaller than prompts.

        Implements a TTFT budget: we demand the first SSE event inside
        ``stream_ttft_timeout`` seconds, after which the fallback chain
        engages.
        """
        # A real SSE response has ``content-type: text/event-stream``.
        # If an upstream misbehaves and returns JSON with status 200 (e.g.
        # an error body they forgot to set a 4xx for), the ``data:`` line
        # filter below would swallow every line and we'd silently return
        # an empty ``LLMResponse``. Surface a retryable error instead so
        # the fallback chain engages and the caller sees the failure.
        ct = (resp.headers.get("content-type") or "").lower()
        if "text/event-stream" not in ct:
            body = await resp.aread()
            raise _RetryableError(
                f"expected SSE, got content-type {ct!r}; body preview: {body[:500]!r}"
            )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Tool calls arrive streamed: first chunk carries ``index`` + name,
        # subsequent chunks for the same index carry ``arguments`` deltas.
        tool_call_parts: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        # Demand the first line inside the TTFT budget. Without this the
        # much larger ``request_timeout`` wins when a server accepts the
        # connection then never sends a byte, so fallback takes minutes
        # instead of seconds. The Copilot review on PR #60 flagged this.
        line_iter = resp.aiter_lines().__aiter__()
        try:
            first_line = await asyncio.wait_for(
                line_iter.__anext__(), timeout=self._stream_ttft_timeout
            )
        except asyncio.TimeoutError:
            raise _RetryableError(f"TTFT exceeded {self._stream_ttft_timeout}s") from None
        except StopAsyncIteration:
            raise _RetryableError("stream closed before any data") from None

        # Re-enter the normal loop with the first line pre-fetched.
        async def _lines_with_first() -> AsyncIterator[str]:
            yield first_line
            async for rest in line_iter:
                yield rest

        async for line in _lines_with_first():
            if not line:
                continue
            # SSE lines are "data: <json>" (plus occasional "event:" / comments).
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # Usage arrives in its own final chunk when ``stream_options
            # include_usage`` is set; choices is empty for that chunk.
            if chunk_usage := chunk.get("usage"):
                usage = {
                    "prompt_tokens": chunk_usage.get("prompt_tokens", 0) or 0,
                    "completion_tokens": chunk_usage.get("completion_tokens", 0) or 0,
                    "total_tokens": chunk_usage.get("total_tokens", 0) or 0,
                }
                if details := chunk_usage.get("prompt_tokens_details"):
                    usage["cached_tokens"] = details.get("cached_tokens", 0) or 0

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if piece := delta.get("content"):
                    content_parts.append(piece)
                if piece := delta.get("reasoning_content"):
                    reasoning_parts.append(piece)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_call_parts.setdefault(
                        idx, {"id": None, "name": None, "arguments": []}
                    )
                    if (tc_id := tc.get("id")) is not None:
                        slot["id"] = tc_id
                    fn = tc.get("function") or {}
                    if (name := fn.get("name")) is not None:
                        slot["name"] = name
                    if (args := fn.get("arguments")) is not None:
                        slot["arguments"].append(args)
                if fr := choice.get("finish_reason"):
                    finish_reason = fr

        tool_calls: list[ToolCallRequest] = []
        for idx in sorted(tool_call_parts):
            slot = tool_call_parts[idx]
            if not slot["name"]:
                continue
            raw_args = "".join(slot["arguments"])
            # json_repair can return non-dict shapes for malformed input;
            # coerce to an empty dict in that case so the tool call still
            # dispatches (the model will get a schema error back, which is
            # more useful than a hard provider-side failure).
            parsed = json_repair.loads(raw_args) if raw_args else {}
            args_obj: dict[str, object] = parsed if isinstance(parsed, dict) else {}
            tool_calls.append(
                ToolCallRequest(
                    id=slot["id"] or _short_tool_id(),
                    name=slot["name"],
                    arguments=args_obj,
                )
            )

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
            thinking_blocks=None,
        )

    def _log_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        logger.info(
            "llm_request",
            **{
                "llm.model": model,
                "llm.message.count": len(messages),
                "llm.tool.count": len(tools) if tools else 0,
            },
        )
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            text = str(content).replace("\n", "\\n")
            if self._llm_log_truncate >= 0:
                text = text[: self._llm_log_truncate]
            logger.info("llm_request_msg", **{"message.role": role, "message.text": text})

    def _log_response(self, model: str, response: LLMResponse, elapsed: float) -> None:
        logger.info(
            "llm_response",
            **{
                "llm.model": model,
                "llm.token.prompt": response.usage.get("prompt_tokens", "?"),
                "llm.token.completion": response.usage.get("completion_tokens", "?"),
                "llm.token.total": response.usage.get("total_tokens", "?"),
                "llm.token.cached": response.usage.get("cached_tokens", 0),
                "llm.duration_s": round(elapsed, 2),
                "llm.finish_reason": response.finish_reason,
                "llm.tools": [tc.name for tc in response.tool_calls],
            },
        )


class _RetryableError(Exception):
    """Marker for errors that should trigger the fallback chain.

    ``chat`` catches this and walks to the next model in the chain. If
    the whole chain exhausts and the cause chain doesn't surface a
    non-internal exception (e.g. a pure HTTP-status path with no
    chained httpx error), callers can see this type on the exhaustion
    path. Treat it as a generic "all fallbacks failed" signal.
    """


async def _is_context_window_error(resp: httpx.Response) -> bool:
    """Heuristic: OpenAI returns 400 with ``code: "context_length_exceeded"``;
    OpenRouter proxies that code. Treat the response body as authoritative.
    Only called on 400 responses so the cost of reading the body is paid
    exactly once and only in the error path.

    Async-read on the streaming response — ``resp.read()`` would be
    synchronous and fails against a real async httpx stream (only works
    against MockTransport because mocks buffer eagerly). See Copilot
    review on PR #60.
    """
    try:
        await resp.aread()
        body = resp.text
    except Exception:
        return False
    if not body:
        return False
    lower = body.lower()
    return "context_length_exceeded" in lower or "context window" in lower


async def _stream_body(
    head: dict[str, Any], messages: list[dict[str, Any]]
) -> AsyncIterator[bytes]:
    """Yield the JSON request body as a sequence of bytes chunks.

    The point is to avoid ever holding the full body as a contiguous
    string. ``messages`` is typically 90%+ of body size; we serialize
    each message individually and yield it as its own chunk so httpx
    can pipe it to the socket and the serialized bytes can be
    garbage-collected before the next message is processed.

    Everything except ``messages`` is stable and small — serialize it
    once, trim the closing brace, and reuse that prefix. The closing
    brace is emitted last, after the messages array is closed.
    """
    # Serialize the non-messages part once — these keys are fixed-size
    # scalars (model, temperature, etc.) plus ``tools`` which is stable
    # across a turn's LLM iterations, so this serialization is small
    # and doesn't repeat for each message.
    head_json = json.dumps(head, ensure_ascii=False)
    # Splice ``"messages":[...]`` in just before the closing ``}``. The
    # dict has no ``messages`` key (caller passes it separately), so
    # ``head_json`` ends with ``}``.
    assert head_json.endswith("}"), "head_json must end with closing brace"
    prefix = head_json[:-1]  # everything except final ``}``

    if prefix and prefix[-1] != "{":
        # ``head`` had at least one key; join with a comma.
        yield prefix.encode("utf-8") + b',"messages":['
    else:
        # ``head`` was empty (unexpected but handled) — start fresh.
        yield b'{"messages":['

    for i, msg in enumerate(messages):
        sep = b"," if i > 0 else b""
        yield sep + json.dumps(msg, ensure_ascii=False).encode("utf-8")

    yield b"]}"
