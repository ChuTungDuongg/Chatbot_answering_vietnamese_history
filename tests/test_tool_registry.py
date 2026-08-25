from __future__ import annotations

import asyncio
from pydantic import BaseModel

from app.tools.registry import ToolRegistry


class EchoInput(BaseModel):
    text: str


class EchoTool:
    name = "echo"
    description = "Echo test tool"
    input_schema = EchoInput

    def run(self, arguments: EchoInput):
        return [{"text": arguments.text}]


def test_tool_registry_calls_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())
    result, record = asyncio.run(registry.call("echo", {"text": "hello"}))
    assert result == [{"text": "hello"}]
    assert record.name == "echo"
    assert record.result_count == 1
