"""MCP tools implement the same registry interface as configured built-ins."""

import hashlib
import json
import re

from jsonschema import Draft202012Validator
from app.mcp.schemas import MCPToolError


def tool_name(server, remote_name):
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", remote_name)[:20] or "tool"
    digest = hashlib.sha256(f"{server}\0{remote_name}".encode()).hexdigest()[:8]
    return f"mcp__{server[:16]}__{slug}__{digest}"


def validate_schema(schema):
    if len(json.dumps(schema).encode()) > 8192: raise MCPToolError("MCP schema vượt giới hạn.")
    def walk(node, depth=0):
        if depth > 20: raise MCPToolError("MCP schema quá sâu.")
        if isinstance(node, dict):
            if any(key in node and not str(node[key]).startswith("#")
                   for key in ("$ref", "$dynamicRef", "$recursiveRef")):
                raise MCPToolError("MCP schema không được tham chiếu URL bên ngoài.")
            for value in node.values(): walk(value, depth+1)
        elif isinstance(node, list):
            for value in node: walk(value, depth+1)
    walk(schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


class MCPToolAdapter:
    provider = "mcp"
    def __init__(self, manager, server, remote_name, description, schema):
        self.manager, self.server, self.remote_name = manager, server, remote_name
        self.name = tool_name(server, remote_name)
        self.description, self.schema = description, schema
        self.validator = validate_schema(schema)
        self.descriptor = {"name": self.name, "description": description, "input_schema": schema,
                           "provider": "mcp", "server": server, "label": remote_name[:64]}

    def validate_arguments(self, arguments):
        if len(json.dumps(arguments).encode()) > 16384 or not self.validator.is_valid(arguments):
            raise MCPToolError("Tham số MCP tool không hợp lệ.")
        return arguments

    async def run_with_context(self, arguments, context):
        # Defence in depth: the registry view also checks the same permission set.
        if context is None or context.allowed_tools is None or self.name not in context.allowed_tools:
            raise MCPToolError("MCP tool không được phép trong request này.")
        return await self.manager.execute(self.server, self.remote_name, arguments, context=context)

    async def run(self, arguments):
        raise MCPToolError("MCP tool yêu cầu request permission context.")
