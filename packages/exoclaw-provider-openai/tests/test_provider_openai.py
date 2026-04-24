"""Tests for ``OpenAIStreamingProvider``.

Covers the two properties that matter for correctness:

1. **Streaming request body.** ``_stream_body`` must emit the full body
   as a correctly-formed JSON document where ``messages`` is a JSON
   array, assembled chunk-per-message. Reassembling the chunks must
   parse back to the original dict.
2. **Per-model routing + fallback.** ``chat`` must POST to the
   deployment associated with the requested model, and must walk the
   fallback chain on retryable errors.

Uses a mock httpx transport so we don't reach the network. The
streaming path is exercised via a real ``httpx.AsyncClient`` wrapping
``httpx.MockTransport`` — that's what catches "my chunks don't
compose into valid JSON" bugs the quickest.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from exoclaw.providers.types import ContextWindowExceededError
from exoclaw_provider_openai import Deployment, OpenAIStreamingProvider
from exoclaw_provider_openai.provider import _stream_body


async def _collect_bytes(gen: Any) -> bytes:
    buf = bytearray()
    async for chunk in gen:
        buf.extend(chunk)
    return bytes(buf)


class TestStreamBody:
    async def test_round_trip_single_message(self) -> None:
        """One message, minimal head — assembled chunks must parse as
        the full intended JSON with messages spliced in."""
        head = {"model": "m1", "temperature": 0.5}
        messages = [{"role": "user", "content": "hi"}]

        body = await _collect_bytes(_stream_body(head, messages))
        parsed = json.loads(body)

        assert parsed == {
            "model": "m1",
            "temperature": 0.5,
            "messages": [{"role": "user", "content": "hi"}],
        }

    async def test_round_trip_many_messages(self) -> None:
        """Separators between chunks must be correct — forgetting the
        ``, `` before message #2 is a common bug. Verify with a larger
        payload."""
        head = {"model": "m1"}
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]

        body = await _collect_bytes(_stream_body(head, messages))
        parsed = json.loads(body)

        assert parsed["model"] == "m1"
        assert parsed["messages"] == messages

    async def test_head_with_tools_and_nested_types(self) -> None:
        """``tools`` is a list of dicts with nested schemas — must
        round-trip without mangling quotes/escaping."""
        head = {
            "model": "m1",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stream": True,
        }
        messages = [{"role": "assistant", "content": 'he said "hi"\nthen left'}]

        body = await _collect_bytes(_stream_body(head, messages))
        parsed = json.loads(body)

        assert parsed["tools"] == head["tools"]
        assert parsed["messages"] == messages

    async def test_empty_messages_still_valid_json(self) -> None:
        """Zero-message edge case — body is still a valid JSON object
        with an empty messages array."""
        head = {"model": "m1"}
        messages: list[dict[str, Any]] = []

        body = await _collect_bytes(_stream_body(head, messages))
        parsed = json.loads(body)

        assert parsed == {"model": "m1", "messages": []}


# SSE helpers ---------------------------------------------------------------


def _sse_completion(
    content: str = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> bytes:
    """Build an SSE response that a real OpenAI-compatible server would
    send. Content arrives in a single chunk (real servers split more, but
    our parser is chunk-aware so this is sufficient to validate assembly)."""
    events: list[str] = []
    delta: dict[str, Any] = {}
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    events.append(
        "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
    )
    events.append(
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )
    )
    events.append("data: [DONE]")
    return ("\n\n".join(events) + "\n\n").encode("utf-8")


# Provider routing ----------------------------------------------------------


def _provider(
    transport: httpx.MockTransport,
    deployments: dict[str, Deployment] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
    default: str = "primary",
) -> OpenAIStreamingProvider:
    deployments = deployments or {
        "primary": Deployment(base_url="https://a.example/v1", api_key="k-a"),
        "backup": Deployment(base_url="https://b.example/v1", api_key="k-b"),
    }
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return OpenAIStreamingProvider(
        default_model=default,
        deployments=deployments,
        fallbacks=fallbacks,
        client=client,
    )


class TestProviderRouting:
    async def test_routes_to_deployment_base_url_and_key(self) -> None:
        """Primary model's base_url + api_key must be used on the
        outgoing request. Other models' deployments must not leak into
        it."""
        seen: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            return httpx.Response(
                200,
                content=_sse_completion("hello"),
                headers={"content-type": "text/event-stream"},
            )

        provider = _provider(httpx.MockTransport(handler))

        resp = await provider.chat(messages=[{"role": "user", "content": "hi"}])

        assert resp.content == "hello"
        assert len(seen) == 1
        req = seen[0]
        assert str(req.url) == "https://a.example/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer k-a"

    async def test_body_is_streamed_not_preserialized(self) -> None:
        """Sanity: the POST must carry the messages we sent, and the
        content-type must be application/json (not multipart or form)."""
        captured: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["body"] = req.read()
            captured["content_type"] = req.headers.get("content-type")
            return httpx.Response(
                200,
                content=_sse_completion(),
                headers={"content-type": "text/event-stream"},
            )

        provider = _provider(httpx.MockTransport(handler))
        messages = [{"role": "user", "content": "hello"}]
        await provider.chat(messages=messages)

        assert captured["content_type"] == "application/json"
        parsed = json.loads(captured["body"])
        assert parsed["messages"] == messages
        assert parsed["stream"] is True

    async def test_fallback_on_503(self) -> None:
        """Retryable status on primary → fallback handles the call and
        the caller gets the fallback's response without seeing the
        primary's error."""
        calls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)
            calls.append(url)
            if url.startswith("https://a.example"):
                return httpx.Response(503, content=b'{"error":"busy"}')
            return httpx.Response(
                200,
                content=_sse_completion("from-backup"),
                headers={"content-type": "text/event-stream"},
            )

        provider = _provider(
            httpx.MockTransport(handler),
            fallbacks={"primary": ["backup"]},
        )

        resp = await provider.chat(messages=[{"role": "user", "content": "x"}])

        assert resp.content == "from-backup"
        assert len(calls) == 2
        assert calls[0].startswith("https://a.example")
        assert calls[1].startswith("https://b.example")

    async def test_no_fallback_on_401(self) -> None:
        """Auth errors are caller-fault, not a transient failure. The
        provider must surface the error instead of silently walking the
        fallback chain (which would hide a misconfiguration)."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(401, content=b'{"error":"bad key"}')

        provider = _provider(
            httpx.MockTransport(handler),
            fallbacks={"primary": ["backup"]},
        )

        with pytest.raises(httpx.HTTPStatusError):
            await provider.chat(messages=[{"role": "user", "content": "x"}])

    async def test_context_window_error_does_not_fallback(self) -> None:
        """A context-window-exceeded on the primary won't succeed on
        the fallback (usually smaller context) — surface the specific
        error so the caller can compact."""
        calls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(str(req.url))
            return httpx.Response(
                400,
                content=b'{"error":{"code":"context_length_exceeded"}}',
            )

        provider = _provider(
            httpx.MockTransport(handler),
            fallbacks={"primary": ["backup"]},
        )

        with pytest.raises(ContextWindowExceededError):
            await provider.chat(messages=[{"role": "user", "content": "x"}])

        assert len(calls) == 1  # fallback NOT attempted

    async def test_fallback_exhausted_raises_last_error(self) -> None:
        """When every model in the chain fails, the last error bubbles
        up instead of being silently swallowed."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(503, content=b'{"error":"busy"}')

        provider = _provider(
            httpx.MockTransport(handler),
            fallbacks={"primary": ["backup"]},
        )

        with pytest.raises(Exception, match="503"):
            await provider.chat(messages=[{"role": "user", "content": "x"}])

    async def test_response_parses_tool_calls(self) -> None:
        """Streamed tool-call chunks must reassemble into a
        ``ToolCallRequest`` with the function name and parsed
        arguments."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse_completion(
                    content="",
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_abc",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"weather"}',
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                headers={"content-type": "text/event-stream"},
            )

        provider = _provider(httpx.MockTransport(handler))
        resp = await provider.chat(messages=[{"role": "user", "content": "x"}])

        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == "lookup"
        assert tc.arguments == {"q": "weather"}
        assert tc.id == "call_abc"

    async def test_extra_body_and_headers_applied(self) -> None:
        """Per-deployment ``extra_headers`` land in the request; per-
        deployment ``extra_body`` lands in the JSON payload."""
        captured: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(req.headers)
            captured["body"] = json.loads(req.read())
            return httpx.Response(
                200,
                content=_sse_completion(),
                headers={"content-type": "text/event-stream"},
            )

        deployments = {
            "primary": Deployment(
                base_url="https://a.example/v1",
                api_key="k",
                extra_headers={"HTTP-Referer": "https://openclaw"},
                extra_body={"provider": {"order": ["deepinfra"]}},
            ),
        }
        provider = _provider(httpx.MockTransport(handler), deployments=deployments)

        await provider.chat(messages=[{"role": "user", "content": "x"}])

        assert captured["headers"].get("http-referer") == "https://openclaw"
        assert captured["body"]["provider"] == {"order": ["deepinfra"]}

    async def test_unknown_model_rejected_at_init(self) -> None:
        """A fallback referring to an undeclared deployment is a
        configuration bug. Catch it at construction time, not on the
        first failure."""
        with pytest.raises(ValueError, match="fallback"):
            OpenAIStreamingProvider(
                default_model="primary",
                deployments={
                    "primary": Deployment(base_url="https://a.example/v1", api_key="k"),
                },
                fallbacks={"primary": ["nonexistent"]},
            )

    async def test_default_model_must_be_declared(self) -> None:
        """Same contract on the ``default_model`` — fail at init."""
        with pytest.raises(ValueError, match="default_model"):
            OpenAIStreamingProvider(
                default_model="not-there",
                deployments={
                    "primary": Deployment(base_url="https://a.example/v1", api_key="k"),
                },
            )
