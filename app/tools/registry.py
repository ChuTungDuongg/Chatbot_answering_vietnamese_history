from __future__ import annotations

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

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[Any, ToolCallRecord]:
        tool = self.get(name)
        parsed = tool.input_schema.model_validate(arguments)
        started = time.perf_counter()
        try:
            result = tool.run(parsed)
            if inspect.isawaitable(result):
                result = await result
            count = len(result) if hasattr(result, "__len__") else None
            logger.info(
                "agent_tool_call",
                extra={"tool_name": name, "latency_ms": (time.perf_counter() - started) * 1000},
            )
            return result, ToolCallRecord(name=name, arguments=parsed.model_dump(), result_count=count)
        except Exception as exc:
            logger.warning(
                "agent_tool_error",
                extra={"tool_name": name, "latency_ms": (time.perf_counter() - started) * 1000},
            )
            return None, ToolCallRecord(name=name, arguments=parsed.model_dump(), error=str(exc))
