"""Stateless, per-request adapter for the official OpenAI Responses API.

This module deliberately contains no credentials or global client.  The caller
owns the shared async client and the lifetime of each request-scoped adapter.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.models.base import ModelDelta, ModelDone
from app.models.tool_calls import DecodedFunctionCall, HermesDecodeResult

OPENAI_MODELS = frozenset({"gpt-6-luna", "gpt-6.1-terra"})
REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})


@dataclass(frozen=True)
class OpenAIOptions:
    model: str = "gpt-6-luna"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 25000

    def __post_init__(self):
        if self.model not in OPENAI_MODELS:
            raise ValueError("Unsupported OpenAI model")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("Unsupported OpenAI reasoning effort")
        if not 1 <= self.max_output_tokens <= 128000:
            raise ValueError("Invalid OpenAI output token limit")


class OpenAIProviderError(RuntimeError):
    """Sanitized provider error safe to return to an API client."""
    def __init__(self, message: str, category: str = "upstream_error", *, metrics=None):
        super().__init__(message)
        self.category = category
        self.metrics = metrics or {}


def _get(obj: Any, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _error(exc: Exception) -> OpenAIProviderError:
    name = type(exc).__name__
    if name in {"APITimeoutError", "TimeoutError"}:
        return OpenAIProviderError("OpenAI request timed out. Please retry.", "timeout")
    if name == "RateLimitError":
        return OpenAIProviderError("OpenAI rate limit reached. Please retry later.", "rate_limit")
    if name in {"NotFoundError", "PermissionDeniedError"}:
        return OpenAIProviderError("The selected OpenAI model is unavailable for this account.", "model_unavailable")
    return OpenAIProviderError("OpenAI request failed. Please retry or choose another provider.")


def _tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Existing planner schemas use the Chat Completions/Hermes envelope.
    return [{"type": "function", **schema["function"], "strict": False} for schema in schemas]


class OpenAIResponsesRuntime:
    inference_backend = "openai"
    resolved_revision = None
    adapter_path = None
    adapter_fingerprint = None
    adapter_attached = False
    model_variant = "remote"
    is_loaded = True

    def __init__(self, client: Any, options: OpenAIOptions):
        self.client, self.options = client, options
        self.model_id = options.model

    @property
    def generation_settings(self):
        return {"reasoning_effort": self.options.reasoning_effort,
                "max_output_tokens": self.options.max_output_tokens}

    @property
    def engine_metadata(self):
        return {"inference_backend": "openai", "provider": "openai"}

    async def aload(self):
        pass

    async def aclose(self):
        # The application lifespan owns the shared client.
        pass

    async def _events(self, messages, cancel, tools=None):
        stream = None
        pending = None
        try:
            kwargs = dict(model=self.model_id, input=messages, stream=True, store=False,
                          reasoning={"effort": self.options.reasoning_effort},
                          max_output_tokens=self.options.max_output_tokens)
            if tools:
                kwargs["tools"] = _tools(tools)
            pending = asyncio.create_task(self.client.responses.create(**kwargs))
            while not pending.done() and not cancel.is_set():
                await asyncio.wait({pending}, timeout=0.1)
            if cancel.is_set():
                if pending.done() and not pending.cancelled() and pending.exception() is None:
                    stream = pending.result()
                return
            stream = pending.result()
            pending = None
            iterator = stream.__aiter__()
            while not cancel.is_set():
                pending = asyncio.create_task(iterator.__anext__())
                while not pending.done() and not cancel.is_set():
                    await asyncio.wait({pending}, timeout=0.1)
                if cancel.is_set():
                    return
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    return
                pending = None
                yield event
        except asyncio.CancelledError:
            raise
        except OpenAIProviderError:
            raise
        except Exception as exc:
            raise _error(exc) from None
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            if stream is not None:
                try:
                    await stream.close()
                except Exception:
                    # Upstream cleanup must not leak exception text or credentials.
                    pass

    async def _run(self, messages, cancel, tools=None):
        started = time.perf_counter_ns()
        first_token = None
        response = None
        async with aclosing(self._events(messages, cancel, tools)) as events:
            async for event in events:
                kind = _get(event, "type")
                if kind == "response.output_text.delta" and not tools:
                    delta = _get(event, "delta", "")
                    if delta:
                        now = time.perf_counter_ns()
                        first_token = first_token or now
                        yield ModelDelta(delta, now)
                elif kind == "response.completed":
                    response = _get(event, "response")
                elif kind == "response.incomplete":
                    response = _get(event, "response")
                    reason = _get(_get(response, "incomplete_details"), "reason")
                    usage = _get(response, "usage")
                    def count(value):
                        return value if type(value) is int and value >= 0 else None
                    metrics = {
                        "generation_ms": (time.perf_counter_ns() - started) / 1e6,
                        "input_tokens": count(_get(usage, "input_tokens")),
                        "output_tokens": count(_get(usage, "output_tokens")),
                        "reasoning_tokens": count(_get(_get(usage, "output_tokens_details"), "reasoning_tokens")),
                        "cached_input_tokens": count(_get(_get(usage, "input_tokens_details"), "cached_tokens")),
                        "finish_reason": "incomplete",
                    }
                    message = ("OpenAI output exceeded the budget; increase it or lower reasoning effort."
                               if reason == "max_output_tokens" else "OpenAI output was incomplete. Please retry.")
                    raise OpenAIProviderError(message, "incomplete", metrics=metrics)
                elif kind in {"response.failed", "error"}:
                    raise OpenAIProviderError("OpenAI request failed. Please retry or choose another provider.")
        if cancel.is_set():
            return
        if response is None or _get(response, "status") != "completed":
            raise OpenAIProviderError("OpenAI stream ended before completion. Please retry.", "interrupted")
        usage = _get(response, "usage")
        output_details = _get(usage, "output_tokens_details")
        input_details = _get(usage, "input_tokens_details")
        finished = time.perf_counter_ns()
        done = ModelDone(self.model_id, None, started, first_token, finished,
                         _get(usage, "input_tokens"), _get(usage, "output_tokens"),
                         max_new_tokens=self.options.max_output_tokens,
                         finish_reason="completed", hit_max_new_tokens=False, truncated=False,
                         timing_observer="openai_response_stream",
                         reasoning_tokens=_get(output_details, "reasoning_tokens"),
                         cached_input_tokens=_get(input_details, "cached_tokens"))
        yield response, done

    async def stream(self, messages, *, max_new_tokens, cancel, tools=None) -> AsyncIterator[ModelDelta | ModelDone]:
        # Local token caps are not meaningful for reasoning models.
        if tools:
            raise ValueError("Use plan() for native OpenAI function calls")
        async with aclosing(self._run(messages, cancel)) as events:
            async for item in events:
                yield item if isinstance(item, ModelDelta) else item[1]

    async def plan(self, messages, *, tools, cancel):
        """Return native calls and replayable output, including encrypted reasoning."""
        result = None
        async with aclosing(self._run(messages, cancel, tools=tools)) as events:
            async for item in events:
                if not isinstance(item, ModelDelta):
                    result = item
        if result is None:
            raise OpenAIProviderError("OpenAI request was cancelled.")
        response, done = result
        calls, failures, replay = [], 0, []
        for item in _get(response, "output", []):
            replay.append(item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item)
            if _get(item, "type") != "function_call":
                continue
            try:
                args = json.loads(_get(item, "arguments", ""))
                if not isinstance(args, dict) or not _get(item, "call_id") or not _get(item, "name"):
                    raise ValueError()
                calls.append(DecodedFunctionCall(_get(item, "call_id"), _get(item, "name"), args))
            except (ValueError, TypeError):
                failures += 1
        return HermesDecodeResult("", tuple(calls), failures), replay, done
