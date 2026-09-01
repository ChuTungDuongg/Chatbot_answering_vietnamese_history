from __future__ import annotations

import copy
import json
import re
from typing import Any

from ..schema import canonical_id, tool_call


class AdapterError(ValueError):
    pass


def source_key(row: dict[str, Any], index: int) -> Any:
    for key in ("id", "sample_id", "uuid", "index"):
        if row.get(key) not in {None, ""}:
            return {"field": key, "value": row[key]}
    return {"index": index, "row": row}


def clean_reasoning(text: str, *, include_reasoning: bool) -> str:
    value = str(text)
    if include_reasoning:
        return value.strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"^\s*(thought|reasoning)\s*:.*?(?=\n\s*(answer|action)\s*:)", "", value, flags=re.DOTALL | re.IGNORECASE)
    return value.strip()


def normalize_tool_definition(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
        name = str(function.get("name") or "").strip()
        parameters = function.get("parameters")
        if not name or not isinstance(parameters, dict):
            raise AdapterError("function tool definition requires name and parameters")
        normalized = copy.deepcopy(tool)
        normalized["function"]["name"] = name
        normalized["function"].setdefault("description", "")
        return normalized
    name = tool.get("name")
    if not name:
        raise AdapterError("tool definition is missing name")
    parameters = tool.get("parameters") or tool.get("input_schema")
    if not isinstance(parameters, dict):
        raise AdapterError(f"tool {name!r} is missing a parameters/input_schema object")
    return {
        "type": "function",
        "function": {
            "name": str(name),
            "description": str(tool.get("description") or ""),
            "parameters": copy.deepcopy(parameters),
        },
    }


def get_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or row.get("conversations") or row.get("conversation")
    if not isinstance(messages, list) or not messages:
        raise AdapterError("row has no compatible messages/conversations/conversation list")
    return messages


ROLE_MAP = {"human": "user", "gpt": "assistant", "function": "tool", "observation": "tool"}


def semantic_messages(
    raw_messages: list[dict[str, Any]],
    *,
    include_reasoning: bool,
    legacy_function_name: str | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []
    call_number = 0
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise AdapterError(f"message {index} is not an object")
        role = ROLE_MAP.get(str(raw.get("role") or raw.get("from") or "").lower(), str(raw.get("role") or "").lower())
        if role not in {"system", "user", "assistant", "tool"}:
            raise AdapterError(f"message {index} has unsupported role {role!r}")
        content = raw.get("content", raw.get("value"))
        if role == "assistant":
            function_call = raw.get("function_call")
            calls = raw.get("tool_calls")
            if function_call:
                calls = [{"function": function_call}]
            if calls:
                normalized_calls: list[dict[str, Any]] = []
                for raw_call in calls:
                    function = raw_call.get("function") or raw_call
                    name = str(function.get("name") or legacy_function_name or "")
                    if not name:
                        raise AdapterError(f"message {index} tool call is missing a name")
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as exc:
                            raise AdapterError(f"message {index} tool arguments are invalid JSON") from exc
                    if not isinstance(arguments, dict):
                        raise AdapterError(f"message {index} tool arguments must be an object")
                    call_number += 1
                    call_id = str(raw_call.get("id") or f"call_{call_number:04d}")
                    normalized_calls.append(tool_call(call_id, name, arguments))
                    pending.append((call_id, name))
                output.append({"role": "assistant", "content": None, "tool_calls": normalized_calls})
            else:
                cleaned = clean_reasoning(str(content or ""), include_reasoning=include_reasoning)
                if not cleaned:
                    raise AdapterError(f"message {index} has an empty assistant response after reasoning removal")
                output.append({"role": "assistant", "content": cleaned})
        elif role == "tool":
            if not pending:
                raise AdapterError(f"message {index} is a tool observation without a preceding call")
            call_id, expected_name = pending.pop(0)
            output.append(
                {
                    "role": "tool",
                    "name": str(raw.get("name") or expected_name),
                    "tool_call_id": str(raw.get("tool_call_id") or call_id),
                    "content": str(content or ""),
                }
            )
        else:
            cleaned = str(content or "").strip()
            if not cleaned:
                raise AdapterError(f"message {index} has empty {role} content")
            output.append({"role": role, "content": cleaned})
    if pending:
        raise AdapterError("assistant tool call has no following tool observation")
    return output


def provenance(
    *,
    dataset_id: str,
    split: str,
    row: dict[str, Any],
    index: int,
    license_name: str | None,
    transformations: list[str],
) -> dict[str, Any]:
    metadata = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in {"messages", "conversations", "conversation"}
    }
    return {
        "dataset_id": dataset_id,
        "original_split": split,
        "original_row_key": source_key(row, index),
        "license": license_name,
        "source_metadata": metadata,
        "transformations": transformations,
        "requires_final_answer": True,
    }


def trajectory_id(dataset_id: str, row: dict[str, Any], index: int) -> str:
    return canonical_id(dataset_id, source_key(row, index))
