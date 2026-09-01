from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


SCHEMA_VERSION = "trajectory-v1"
DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý trung tâm về lịch sử Việt Nam. Hãy dùng công cụ khi cần, "
    "chỉ kết luận từ bằng chứng quan sát được và trích dẫn ID nguồn."
)
CENTRAL_V2_SYSTEM_PROMPT = (
    "Bạn là Central V2. Dùng function call có cấu trúc khi công cụ được cung cấp, "
    "đọc kết quả công cụ, chỉ kết luận từ bằng chứng và không tiết lộ suy luận ẩn."
)
QWEN3_TOOL_TEMPLATE_CONTRACT = {
    "family": "qwen3",
    "tool_format": "semantic_tool_calls",
    "tool_result_role": "tool",
    "enable_thinking": False,
}

SEARCH_HISTORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_history",
        "description": "Search the local Vietnamese-history corpus and return ranked chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

RETRIEVE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Retrieve information relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": True,
        },
    },
}

SEARCH_WIKIPEDIA_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": "Search Vietnamese or English Wikipedia and return stable evidence snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "language": {"type": "string", "enum": ["vi", "en"], "default": "vi"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

SEARCH_WEB_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search external web evidence when configured.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_id(source_dataset: str, source_key: Any, *, prefix: str = "traj") -> str:
    payload = f"{source_dataset}\n{stable_json(source_key)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    safe_source = "".join(ch if ch.isalnum() else "-" for ch in source_dataset.lower()).strip("-")
    return f"{prefix}-{safe_source[:30]}-{digest}"


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def tool_names(tools: Iterable[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
    return names


def make_trajectory(
    *,
    trajectory_id: str,
    source_dataset: str,
    task_type: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    difficulty: str = "medium",
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_list = list(tools or [])
    return {
        "id": trajectory_id,
        "schema_version": SCHEMA_VERSION,
        "source_dataset": source_dataset,
        "task_type": task_type,
        "uses_tools": any(message.get("tool_calls") for message in messages),
        "difficulty": difficulty,
        "tools": tool_list,
        "messages": messages,
        "provenance": dict(provenance or {}),
    }
