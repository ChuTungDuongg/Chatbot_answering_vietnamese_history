from __future__ import annotations

import asyncio
import inspect
import logging
import time
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel
from app.rag.backends import DenseBackendError
from app.rag.progress import StageProgress, ProgressCallback
from app.mcp.schemas import MCPToolError


logger = logging.getLogger(__name__)


class AgentTool(Protocol):
    name: str
    description: str
    input_schema: type[BaseModel]

    def run(self, arguments: BaseModel) -> Any:
        ...


@dataclass(frozen=True)
class ToolExecutionPolicy:
    read_only: bool = False
    parallel_safe: bool = False
    exclusive_resource: str | None = None


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result_count: int | None = None
    error: str | None = None
    provider: str = "builtin"
    server: str | None = None


@dataclass(frozen=True)
class ToolExecutionContext:
    owner_id: str | None = None
    conversation_id: str | None = None
    session_id: str = "default"
    request_id: str | None = None
    attachment_ids: tuple[str, ...] | None = None
    retrieval_backend: str | None = None
    progress: ProgressCallback | None = None
    retrieval_metrics: dict[str, Any] = field(default_factory=dict)
    allowed_tools: frozenset[str] | None = None
    cancel: threading.Event | None = None
    mcp_failure_policy: str = "continue"
    mcp_metrics: list[dict[str, Any]] = field(default_factory=list)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        self._sync_locks: dict[str, threading.Lock] = {}

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        descriptor = getattr(tool, "descriptor", None) or {
            "name": tool.name, "description": tool.description,
            "input_schema": tool.input_schema.model_json_schema(), "provider": "builtin",
        }
        self._tools[tool.name] = tool
        self._descriptors[tool.name] = descriptor

    def execution_policy(self, name: str) -> ToolExecutionPolicy:
        tool = self.get(name)
        # Remote readOnlyHint is not evidence of session concurrency safety.
        if self._descriptors[name].get("provider") == "mcp":
            return ToolExecutionPolicy()
        policy = getattr(tool, "execution_policy", None)
        if not isinstance(policy, ToolExecutionPolicy):
            return ToolExecutionPolicy(exclusive_resource=f"tool:{name}")
        return policy

    def validate_arguments(self, name: str, arguments: dict[str, Any]):
        tool = self.get(name)
        validator = getattr(tool, "validate_arguments", None)
        return validator(arguments) if callable(validator) else tool.input_schema.model_validate(arguments)

    def get(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        # Schemas are computed once, outside the request hot path.
        return list(self._descriptors.values())

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None = None,
    ) -> tuple[Any, ToolCallRecord]:
        started = time.perf_counter()
        provider, server = "builtin", None
        try:
            if context and context.allowed_tools is not None and name not in context.allowed_tools:
                return None, ToolCallRecord(name=name, arguments={}, error="tool_not_allowed")
            tool = self.get(name)
            provider = self._descriptors[name].get("provider", "builtin")
            server = getattr(tool, "server", None)
            parsed = self.validate_arguments(name, arguments)
            if context is None:
                runner = tool.run
                call_args = (parsed,)
            else:
                run_with_context = getattr(tool, "run_with_context", None)
                runner = run_with_context if callable(run_with_context) else tool.run
                call_args = (parsed, context) if callable(run_with_context) else (parsed,)
            stages = StageProgress(context.progress if context else None,
                                   context.retrieval_backend if context else None)
            resource = self.execution_policy(name).exclusive_resource
            with (nullcontext() if provider == "mcp" else stages.track(f"tool:{name}")):
                if inspect.iscoroutinefunction(runner):
                    if resource:
                        # Share the same gate with synchronous workers; never
                        # release a worker's gate merely because its awaiter stops.
                        lock = self._sync_locks.setdefault(resource, threading.Lock())
                        while not lock.acquire(blocking=False):
                            if context and context.cancel and context.cancel.is_set():
                                raise asyncio.CancelledError()
                            await asyncio.sleep(.01)
                        try:
                            result = await runner(*call_args)
                        finally:
                            lock.release()
                    else:
                        result = await runner(*call_args)
                else:
                    lock = self._sync_locks.setdefault(resource, threading.Lock()) if resource else None
                    def invoke():
                        acquired = False
                        try:
                            while lock and not acquired:
                                if context and context.cancel and context.cancel.is_set():
                                    raise asyncio.CancelledError()
                                acquired = lock.acquire(timeout=.05)
                            if context and context.cancel and context.cancel.is_set():
                                raise asyncio.CancelledError()
                            return runner(*call_args)
                        finally:
                            if acquired:
                                lock.release()
                    result = await asyncio.to_thread(invoke)
                    if inspect.isawaitable(result):
                        result = await result
            count = len(result) if hasattr(result, "__len__") else None
            logger.info(
                "agent_tool_call",
                extra={
                    "request_id": context.request_id if context is not None else None,
                    "tool_name": name,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "result_count": count,
                },
            )
            safe_arguments = {} if provider == "mcp" else parsed.model_dump()
            return result, ToolCallRecord(name=name, arguments=safe_arguments, result_count=count,
                                          provider=provider, server=server)
        except DenseBackendError:
            # History retrieval must fail honestly, never generate from another lane.
            raise
        except Exception as exc:
            if provider == "mcp" and context and context.mcp_failure_policy == "fail":
                raise MCPToolError("Công cụ MCP không hoàn tất; request yêu cầu dừng khi lỗi.") from None
            logger.warning(
                "agent_tool_error",
                extra={
                    "request_id": context.request_id if context is not None else None,
                    "tool_name": name,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error_type": type(exc).__name__,
                },
            )
            return None, ToolCallRecord(name=name, arguments={} if provider == "mcp" else dict(arguments),
                error="MCP tool failed" if provider == "mcp" else "tool_failed", provider=provider, server=server)
