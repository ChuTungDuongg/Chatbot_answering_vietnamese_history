from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class AgentTool(Protocol):
    name: str
    description: str
    input_schema: type[BaseModel]

    def run(self, arguments: BaseModel) -> Any:
        ...


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result_count: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class ToolExecutionContext:
    owner_id: str | None = None
    conversation_id: str | None = None
    session_id: str = "default"
    request_id: str | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None = None,
    ) -> tuple[Any, ToolCallRecord]:
        started = time.perf_counter()
        try:
            tool = self.get(name)
            parsed = tool.input_schema.model_validate(arguments)
            if context is None:
                runner = tool.run
                call_args = (parsed,)
            else:
                run_with_context = getattr(tool, "run_with_context", None)
                runner = run_with_context if callable(run_with_context) else tool.run
                call_args = (parsed, context) if callable(run_with_context) else (parsed,)
            if inspect.iscoroutinefunction(runner):
                result = await runner(*call_args)
            else:
                result = await asyncio.to_thread(runner, *call_args)
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
            return result, ToolCallRecord(name=name, arguments=parsed.model_dump(), result_count=count)
        except Exception as exc:
            logger.warning(
                "agent_tool_error",
                extra={
                    "request_id": context.request_id if context is not None else None,
                    "tool_name": name,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error_type": type(exc).__name__,
                },
            )
            return None, ToolCallRecord(name=name, arguments=dict(arguments), error=str(exc))
