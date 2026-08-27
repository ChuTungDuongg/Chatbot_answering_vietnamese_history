from __future__ import annotations

import hashlib
import json
from typing import Any

from app.agents.policy_schema import PolicyLimits, ResearchPolicyState, policy_messages


def _json_value(value: Any, *, field: str, expected: type) -> Any:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"xLAM {field} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, expected):
        raise ValueError(f"xLAM {field} must decode to {expected.__name__}")
    return parsed


def convert_xlam(row: dict[str, Any], *, row_index: int = 0) -> dict[str, Any]:
    """Convert the official query/tools/answers xLAM schema without dropping calls."""
    query_value = row.get("query")
    if isinstance(query_value, str):
        try:
            decoded = json.loads(query_value)
            query = decoded if isinstance(decoded, str) else query_value
        except json.JSONDecodeError:
            query = query_value
    else:
        raise ValueError("xLAM query must be a string")
    query = query.strip()
    if not query:
        raise ValueError("xLAM query is empty")
    tools = _json_value(row.get("tools"), field="tools", expected=list)
    answers = _json_value(row.get("answers"), field="answers", expected=list)
    if not tools or not answers:
        raise ValueError("xLAM tools and answers must be non-empty")

    normalized_tools = []
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or not str(tool.get("name") or "").strip()
            or not isinstance(tool.get("parameters"), dict)
        ):
            raise ValueError("xLAM contains a malformed tool definition")
        normalized_tools.append({
            "name": str(tool["name"]),
            "description": str(tool.get("description") or ""),
            "input_schema": dict(tool.get("parameters") or {}),
        })
    tool_names = {tool["name"] for tool in normalized_tools}
    calls = []
    for answer in answers:
        if not isinstance(answer, dict) or not isinstance(answer.get("arguments"), dict):
            raise ValueError("xLAM answer must contain name and object arguments")
        name = str(answer.get("name") or "")
        if name not in tool_names:
            raise LookupError(f"xLAM answer calls undefined tool: {name}")
        calls.append({"tool_name": name, "arguments": answer["arguments"]})

    state = ResearchPolicyState(
        question=query,
        step=1,
        limits=PolicyLimits(max_steps=1),
        tools=normalized_tools,
        observations=[],
        evidence_ids=[],
        trajectory_class="generic_tool_use",
    )
    target: dict[str, Any]
    if len(calls) == 1:
        target = {"action": "tool", **calls[0]}
    else:
        target = {"action": "tool_batch", "tool_calls": calls}
    original_id = str(row.get("id") or f"xlam-{row_index:06d}")
    group_id = f"xlam-{hashlib.sha256(query.casefold().encode()).hexdigest()[:20]}"
    return {
        "id": f"{original_id}-step-001",
        "source_dataset": "xlam",
        "source": "generic_tool_use",
        "stage": "generic_tool_use",
        "original_sample_id": original_id,
        "group_id": group_id,
        "trajectory_id": original_id,
        "trajectory_class": "generic_tool_use",
        "step": 1,
        "grounded": True,
        "synthetic": False,
        "messages": policy_messages(state, target, generic=True),
        "training_prompt": state.model_dump(exclude_none=True),
        "training_target": target,
    }
