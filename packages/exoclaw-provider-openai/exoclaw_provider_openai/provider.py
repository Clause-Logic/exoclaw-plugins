"""OpenAI-compatible LLM provider for exoclaw, runs on CPython + MicroPython.

The provider speaks the OpenAI chat-completions protocol and streams
the request body via HTTP/1.1 chunked transfer encoding, so the full
JSON never materialises as one contiguous string. That's the
peak-memory reduction the ``docs/memory-model.md`` Step B plan is
aimed at.

Network plumbing is delegated to ``exoclaw.http.HTTPClient``:
CPython gets the ``httpx``-backed implementation (connection
pooling, battle-tested error handling); MicroPython gets the
hand-rolled ``asyncio.open_connection`` path. Same provider source
either way — no runtime gates here.

Per-model routing and fallback: each model name maps to exactly one
``Deployment`` (base URL + API key + optional extra headers), and
each model has an optional fallback chain. A retryable error on the
primary walks the chain; non-retryable errors (auth, 400-class)
bubble up.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

from exoclaw._compat import (
    IS_MICROPYTHON,
    aiter_compat,
    get_log_contextvars,
    get_logger,
    monotonic_diff_ms,
    monotonic_ms,
)

if not IS_MICROPYTHON:
    # ``json_repair`` is a CPython-only dep — handles malformed
    # tool-call JSON the model produces under load. MicroPython
    # falls back to plain ``json.loads`` below; on a chip the
    # extra resiliency isn't worth pulling in a 3rd-party
    # dependency that isn't packaged for ``mip``.
    import json_repair
from exoclaw.http import (
    HTTPClient,
    HTTPConnectError,
    HTTPError,
    HTTPReadTimeout,
    HTTPWriteTimeout,
)
from exoclaw.providers.types import (
    ContextWindowExceededError,
    LLMResponse,
    ResponseFormat,
    ToolCallRequest,
)

if TYPE_CHECKING:
    # ``collections.abc`` doesn't ship on MicroPython; pulled in
    # for type-checking only. ``from __future__ import annotations``
    # stringifies all annotations so the runtime never resolves
    # these names.
    from collections.abc import AsyncIterator

    from exoclaw.http import ClientProto, ResponseProto

logger = get_logger()

# 9-char alnum tool-call id alphabet. OpenAI/Anthropic accept arbitrary
# strings for ``tool_calls[].id``; some providers (Mistral) reject
# longer/punctuated ids, so keep to a safe subset.
_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# Response-retryable HTTP status codes. 408/425 join the 5xx/429 set
# for safety — some providers emit them on transient queue pressure.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _short_tool_id() -> str:
    """9-char alnum tool-call id."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        # ``secrets`` doesn't ship on MP; ``os.urandom`` is the
        # cross-runtime CSPRNG. Map each byte to one char in
        # ``_ALNUM`` (62-char alphabet — close enough to flat;
        # the bias is negligible for an id-shape value).
        raw = os.urandom(9)
        return "".join(_ALNUM[b % len(_ALNUM)] for b in raw)
    import secrets  # pragma: no cover (micropython)

    return "".join(secrets.choice(_ALNUM) for _ in range(9))  # pragma: no cover (micropython)


# Dataclass-or-plain-class dual pattern: MicroPython strips
# annotations at compile time so the runtime ``@dataclass`` decorator
# can't introspect ``base_url: str`` etc. and ends up with no fields.
# Build a hand-written class on MP; keep ``@dataclass`` on CPython
# for the standard ``__repr__``/``__eq__`` machinery.
if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class Deployment:
        """A single model → endpoint binding.

        ``extra_headers`` is merged into the request headers on every
        call; ``extra_body`` is merged into the JSON body at the top
        level (e.g. for OpenRouter's ``provider`` routing object, or
        a custom ``transforms`` flag). Both are read-only after
        construction.
        """

        base_url: str
        api_key: str
        extra_headers: dict[str, str] = field(default_factory=dict)
        extra_body: dict[str, Any] = field(default_factory=dict)

else:  # pragma: no cover (cpython)

    class Deployment:
        """MicroPython fallback — plain class with hand-written
        ``__init__``. Same shape as the CPython ``@dataclass`` branch
        above; MP can't introspect ``base_url: str`` annotations at
        runtime, so the ``@dataclass`` decorator would produce a
        no-field class."""

        def __init__(
            self,
            base_url: str,
            api_key: str,
            extra_headers: dict[str, str] | None = None,
            extra_body: dict[str, Any] | None = None,
        ) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.extra_headers = extra_headers if extra_headers is not None else {}
            self.extra_body = extra_body if extra_body is not None else {}


class OpenAIStreamingProvider:
    """Direct-HTTP provider speaking OpenAI chat-completions protocol.

    Implements the exoclaw ``LLMProvider`` protocol. The HTTP layer
    is ``exoclaw.http.HTTPClient``, so the same source runs on both
    CPython (httpx underneath) and MicroPython (hand-rolled
    HTTP/1.1)."""

    def __init__(
        self,
        default_model: str,
        deployments: dict[str, Deployment],
        fallbacks: dict[str, list[str]] | None = None,
        *,
        request_timeout: float = 120.0,
        stream_ttft_timeout: float = 15.0,
        client: "ClientProto | None" = None,
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

        # Allow dependency injection for tests. In production one
        # client is reused across all requests so the underlying
        # transport (httpx on CPython) can pool connections.
        self._owns_client = client is None
        self._client: ClientProto = client or HTTPClient(timeout=request_timeout)

        # ``os.getenv`` is the cross-runtime API — MicroPython
        # doesn't ship ``os.environ`` but does ship ``getenv``.
        self._llm_logging = (os.getenv("LLM_LOGGING") or "").lower() == "true"
        self._llm_log_truncate = int(os.getenv("LLM_LOG_TRUNCATE") or "500")

    def get_default_model(self) -> str:
        return self.default_model

    async def close(self) -> None:
        """Close the underlying HTTP client. Safe to call multiple times."""
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
                # Exception for the type checker and fall through to
                # the _RetryableError itself if the cause isn't a
                # plain Exception (it always is in practice).
                # MicroPython exceptions don't expose ``__cause__``
                # at all; ``getattr`` keeps the fallback identical
                # across runtimes (no ``raise from`` chain available
                # on MP, so the wrapper exception is the cause we
                # log on that path anyway).
                cause = getattr(e, "__cause__", None)
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

        # Request body is assembled as a dict of metadata + the
        # messages list, but we stream it out chunk-per-message so
        # the full JSON never lives in memory as one string. See
        # ``_stream_body``.
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

        # ``monotonic_ms`` is the cross-runtime monotonic clock —
        # ``time.monotonic()`` doesn't ship on MicroPython.
        t0 = monotonic_ms()
        try:
            async with self._client.stream_post(
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

        except HTTPConnectError as e:
            raise _RetryableError(f"connect error: {e}") from e
        except HTTPReadTimeout as e:
            raise _RetryableError(f"read timeout: {e}") from e
        except HTTPWriteTimeout as e:
            raise _RetryableError(f"write timeout: {e}") from e
        except asyncio.TimeoutError as e:
            raise _RetryableError(f"timeout: {e}") from e
        # ``HTTPStatusError`` from ``resp.raise_for_status()`` is NOT
        # caught here — non-retryable 4xx (auth, malformed request)
        # is caller-fault and should bubble up so the caller sees
        # the misconfiguration instead of silently walking the
        # fallback chain. Retryable 5xx/429 status codes are filtered
        # earlier in this function and re-raised as ``_RetryableError``
        # before ever reaching ``raise_for_status``.

        elapsed = monotonic_diff_ms(monotonic_ms(), t0) / 1000.0
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
            ctx = get_log_contextvars()
            session_key = ctx.get("session.key")
            if session_key:
                extra_body["user"] = str(session_key)
        return extra_body

    async def _consume_sse_stream(self, resp: "ResponseProto") -> LLMResponse:
        """Accumulate SSE chunks into a single ``LLMResponse``.

        We need the full response anyway (the turn loop wants
        tool_calls and finish_reason materialized) — streaming is
        purely for the server-side TTFT and incremental-decode wins.

        Implements a TTFT budget: we demand the first SSE event
        inside ``stream_ttft_timeout`` seconds, after which the
        fallback chain engages.
        """
        # A real SSE response has ``content-type: text/event-stream``.
        # If an upstream misbehaves and returns JSON with status 200
        # (e.g. an error body they forgot to set a 4xx for), the
        # ``data:`` line filter below would swallow every line and
        # we'd silently return an empty ``LLMResponse``. Surface a
        # retryable error instead so the fallback chain engages.
        ct = (resp.headers.get("content-type") or "").lower()
        if "text/event-stream" not in ct:
            body = await resp.aread()
            raise _RetryableError(
                f"expected SSE, got content-type {ct!r}; body preview: {body[:500]!r}"
            )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Tool calls arrive streamed: first chunk carries ``index`` +
        # name, subsequent chunks for the same index carry
        # ``arguments`` deltas.
        tool_call_parts: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        # Demand the first line inside the TTFT budget. Without this
        # the much larger ``request_timeout`` wins when a server
        # accepts the connection then never sends a byte.
        #
        # CPython uses ``asyncio.wait_for`` (clean cancellation
        # semantics, runs in a separate task). MicroPython skips
        # ``wait_for`` here: it puts the calling task on
        # ``_task_queue`` (sleep) while the wrapped IO inner task
        # registers in ``IOQueue``; combined with SSL streaming reads
        # this hits a pairheap-double-push assert in MP's asyncio.
        # On MP we just await the first line and rely on the
        # broader ``request_timeout`` to surface stalls.
        line_iter = resp.aiter_lines().__aiter__()
        try:
            if IS_MICROPYTHON:  # pragma: no cover (cpython)
                first_line = await line_iter.__anext__()
            else:  # pragma: no cover (micropython)
                first_line = await asyncio.wait_for(
                    line_iter.__anext__(), timeout=self._stream_ttft_timeout
                )
        except asyncio.TimeoutError:
            raise _RetryableError(f"TTFT exceeded {self._stream_ttft_timeout}s") from None
        except StopAsyncIteration:
            raise _RetryableError("stream closed before any data") from None

        # Iterate ``line_iter`` directly, injecting the pre-fetched
        # ``first_line`` on the first turn. We can't go through an
        # ``async def gen() -> AsyncIterator[str]: yield first_line;
        # async for rest in line_iter: yield rest`` adapter on MP:
        # ``async def`` + ``yield`` collapses to a sync generator
        # there, and a nested ``async for`` inside it silently
        # truncates iteration after a few yields (we saw the loop
        # exit cleanly after just the OpenRouter heartbeat line, no
        # ``data:`` chunk ever processed). ``_LineIter`` (MP) and
        # httpx (CPython) both implement ``__aiter__``/``__anext__``
        # natively, so no adapter is needed.
        _need_first = True
        while True:
            if _need_first:
                line = first_line
                _need_first = False
            else:
                try:
                    line = await line_iter.__anext__()
                except StopAsyncIteration:
                    break
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
            except ValueError:
                # MP's ``json`` has no ``JSONDecodeError`` — referencing
                # it in the ``except`` tuple raises at lookup time on
                # MP. CPython's ``JSONDecodeError`` is a ``ValueError``
                # subclass anyway, so ``ValueError`` covers both.
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
            # json_repair can return non-dict shapes for malformed
            # input; coerce to an empty dict so the tool call still
            # dispatches (the model will get a schema error back,
            # which is more useful than a hard provider-side failure).
            # On MicroPython, ``json_repair`` isn't available — fall
            # back to plain ``json.loads`` and let malformed input
            # raise. The except clause below catches it and yields
            # an empty dict.
            if not raw_args:
                parsed: object = {}
            elif IS_MICROPYTHON:
                try:
                    parsed = json.loads(raw_args)
                except ValueError:
                    # See note above: MP's ``json`` doesn't expose
                    # ``JSONDecodeError``; ``ValueError`` covers both
                    # runtimes.
                    parsed = {}
            else:
                parsed = json_repair.loads(raw_args)
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


class _RetryableError(HTTPError):
    """Marker for errors that should trigger the fallback chain.

    ``chat`` catches this and walks to the next model in the chain.
    Inherits from ``HTTPError`` so the catch block in ``_chat_once``
    that re-raises HTTP-layer errors as ``_RetryableError`` does the
    right thing if the underlying client raised an
    ``exoclaw.http`` exception we didn't enumerate explicitly."""


async def _is_context_window_error(resp: "ResponseProto") -> bool:
    """Heuristic: OpenAI returns 400 with ``code: "context_length_exceeded"``;
    OpenRouter proxies that code. Treat the response body as authoritative.
    Only called on 400 responses so the cost of reading the body is paid
    exactly once and only in the error path.
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


async def _emit_message(msg: dict[str, Any]) -> AsyncIterator[bytes]:
    """Serialize one message, streaming ``content`` from disk if the
    message carries a ``_content_file`` reference.

    Step D (memory-model.md): tools that opt into ``execute_streaming``
    drain their output to a per-turn scratch file. The agent loop
    attaches the path to the tool message via the ``_content_file``
    transport-metadata key. This helper detects that key and streams
    the file's bytes into the JSON ``content`` field as JSON-escaped
    chunks, so a multi-MB tool result never materialises as one
    contiguous Python string at the moment of the LLM call.

    Underscore-prefixed keys are stripped from the serialized output
    regardless — they're transport metadata, not part of the LLM
    message. (``loop.py`` already strips them on the persistence
    path; this is the symmetric strip on the wire path.)

    ``ensure_ascii=False`` is omitted — MicroPython's ``json.dumps``
    doesn't accept the kwarg, and the default (``True``) is fine on
    both runtimes for OpenAI-compatible servers.
    """
    content_file_str = msg.get("_content_file")
    if not content_file_str:
        clean = {k: v for k, v in msg.items() if not k.startswith("_")}
        yield json.dumps(clean).encode("utf-8")
        return

    # File-backed: assemble ``{<head>, "content": "<streamed escaped>"}``
    # without ever holding the full content in heap.
    head = {k: v for k, v in msg.items() if k != "content" and not k.startswith("_")}
    head_json = json.dumps(head)
    assert head_json.endswith("}"), "head_json must end with closing brace"
    prefix = head_json[:-1]  # everything except final ``}``
    # ``head`` always has at least ``role``, so ``prefix`` is never
    # just ``{`` after stripping the closing brace — comma-separator
    # before ``"content":`` is always correct. Defensive check kept
    # in case a future caller passes a content-only message.
    sep = b"," if prefix.rstrip() != "{" else b""
    yield prefix.encode("utf-8") + sep + b'"content":"'

    # Read the scratch file in fixed-size character chunks. Text
    # mode handles UTF-8 codepoint-boundary alignment for us, so a
    # chunk is always whole codepoints. ``json.dumps`` of each
    # chunk gives correct JSON-string escaping; slicing off the
    # surrounding quotes gives just the escaped body bytes.
    try:
        # ``newline=""`` matches the writer's open mode in
        # ``DirectExecutor.execute_tool_with_handle`` so universal-
        # newline translation doesn't turn the writer's exact byte
        # sequence (``\r\n`` on Windows when a tool emits CRLF) into
        # ``\n`` here, drifting wire content from on-disk content.
        # MicroPython's ``open`` doesn't accept ``encoding`` /
        # ``newline``; gate on runtime so the same source works.
        if IS_MICROPYTHON:  # pragma: no cover (cpython)
            fh = open(content_file_str)
        else:  # pragma: no cover (micropython)
            fh = open(content_file_str, encoding="utf-8", newline="")
        try:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                escaped = json.dumps(chunk)[1:-1]
                yield escaped.encode("utf-8")
        finally:
            fh.close()
    except (OSError, UnicodeError):
        # Scratch file disappeared between tool execution and provider
        # send (manual cleanup, OS tmpwatch, race with post_turn) or
        # a transient read error / encoding glitch. Fall back to the
        # inline ``content`` preview that the executor already
        # populated when it returned the ``ToolResult``. The LLM sees
        # the head + footer line (``[streamed N bytes ...]``) rather
        # than a 400 from a malformed JSON payload, and the request
        # streaming continues for the rest of the messages.
        fallback = msg.get("content")
        if isinstance(fallback, str) and fallback:
            escaped = json.dumps(fallback)[1:-1]
            yield escaped.encode("utf-8")

    yield b'"}'


async def _stream_body(
    head: dict[str, Any], messages: list[dict[str, Any]]
) -> AsyncIterator[bytes]:
    """Yield the JSON request body as a sequence of bytes chunks.

    The point is to avoid ever holding the full body as a contiguous
    string. ``messages`` is typically 90%+ of body size; we serialize
    each message individually and yield it as its own chunk so the
    HTTP client can pipe it to the socket and the serialized bytes
    can be garbage-collected before the next message is processed.

    Per-message serialization is delegated to ``_emit_message`` which
    handles both inline messages (the common case) and file-backed
    Step-D streaming-tool-result messages.
    """
    # Serialize the non-messages part once — these keys are
    # fixed-size scalars (model, temperature, etc.) plus ``tools``
    # which is stable across a turn's LLM iterations, so this
    # serialization is small and doesn't repeat for each message.
    head_json = json.dumps(head)
    # Splice ``"messages":[...]`` in just before the closing ``}``.
    assert head_json.endswith("}"), "head_json must end with closing brace"
    prefix = head_json[:-1]  # everything except final ``}``

    if prefix and prefix[-1] != "{":
        # ``head`` had at least one key; join with a comma.
        yield prefix.encode("utf-8") + b',"messages":['
    else:
        # ``head`` was empty (unexpected but handled) — start fresh.
        yield b'{"messages":['

    for i, msg in enumerate(messages):
        if i > 0:
            yield b","
        # ``_emit_message`` is ``async def`` + ``yield`` which on
        # CPython produces an async generator and on MicroPython
        # 1.27 a plain generator. ``aiter_compat`` adapts either.
        async for chunk in aiter_compat(_emit_message(msg)):
            yield chunk

    yield b"]}"
