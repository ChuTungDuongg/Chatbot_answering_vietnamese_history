"""Immutable request-local tool views. Natural-language preferences only restrict."""

import json
import re
from dataclasses import dataclass

from app.mcp.schemas import ToolSteering


class ToolPolicyError(ValueError):
    """A safe client-facing policy/configuration mismatch."""


@dataclass(frozen=True)
class ToolView:
    allowed_names: frozenset[str]
    schemas: tuple[dict, ...]
    steering: dict
    failure_policy: str = "continue"


def tool_view(registry, *, steering: ToolSteering | None = None, question="",
              mcp_capabilities=None, max_mcp_tools=8, schema_budget=16384):
    steering = steering or ToolSteering()
    descriptors = registry.describe()
    builtins = {item["name"] for item in descriptors if item.get("provider", "builtin") == "builtin"}
    allowed = set(builtins)
    capabilities = mcp_capabilities or {"enabled": False, "servers": []}
    servers = {server["id"]: server for server in capabilities.get("servers", [])}
    if steering.mcp_enabled:
        if not capabilities.get("enabled"):
            raise ToolPolicyError("MCP chưa được bật trên server.")
        if any(server not in servers or not servers[server]["available"] for server in steering.allowed_mcp_servers):
            raise ToolPolicyError("MCP server được chọn không khả dụng.")
        allowed.update(item["name"] for item in descriptors if item.get("provider") == "mcp"
                       and item.get("server") in steering.allowed_mcp_servers)
    if steering.allowed_tools is not None:
        requested = set(steering.allowed_tools)
        if not requested.issubset(allowed):
            raise ToolPolicyError("Công cụ được chọn chưa được server/request cho phép.")
        allowed.intersection_update(requested)
    normalized = question.casefold()
    if re.search(r"chỉ dùng (?:kho sử liệu|nguồn|dữ liệu) (?:local|nội bộ)|local only", normalized):
        allowed.intersection_update({"search_history"})
    if re.search(r"không (?:dùng|sử dụng) wikipedia|no wikipedia", normalized):
        allowed.difference_update({"search_wikipedia", "fetch_wikipedia_page"})
    if re.search(r"không (?:dùng|sử dụng) mcp|no mcp", normalized):
        allowed.intersection_update(builtins)
    selected = [item for item in descriptors if item["name"] in allowed]
    external = [item for item in selected if item.get("provider") == "mcp"]
    if len(external) > max_mcp_tools or sum(len(json.dumps(item, ensure_ascii=False).encode()) for item in external) > schema_budget:
        raise ToolPolicyError("Quá nhiều công cụ MCP; chọn ít công cụ hơn cho request này.")
    schemas = tuple({"type": "function", "function": {"name": item["name"],
                     "description": item["description"], "parameters": item["input_schema"]}} for item in selected)
    return ToolView(frozenset(allowed), schemas,
        {**steering.model_dump(), "effective_tools": sorted(allowed)}, steering.mcp_failure_policy)


def builtin_capabilities(registry):
    labels = {"search_history": "Kho sử liệu", "search_uploaded_documents": "Tài liệu của bạn",
              "search_wikipedia": "Wikipedia", "fetch_wikipedia_page": "Đọc Wikipedia",
              "search_web": "Tìm trên web", "fetch_page": "Đọc trang web"}
    return [{"id": item["name"], "label": labels.get(item["name"], item["name"]), "available": True}
            for item in registry.describe() if item.get("provider", "builtin") == "builtin"]
