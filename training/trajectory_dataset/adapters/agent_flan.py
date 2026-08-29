from __future__ import annotations

from typing import Any

from ..schema import make_trajectory
from .common import (
    AdapterError,
    get_messages,
    normalize_tool_definition,
    provenance,
    semantic_messages,
    trajectory_id,
)


DATASET_ID = "internlm/Agent-FLAN"


def normalize_agent_flan(
    row: dict[str, Any],
    *,
    index: int = 0,
    split: str = "train",
    include_reasoning: bool = False,
) -> dict[str, Any]:
    raw_tools = row.get("tools") or row.get("functions") or []
    if raw_tools and not isinstance(raw_tools, list):
        raise AdapterError("Agent-FLAN tools/functions must be a list")
    tools = [normalize_tool_definition(tool) for tool in raw_tools]
    messages = semantic_messages(get_messages(row), include_reasoning=include_reasoning)
    called_names = {
        call["function"]["name"]
        for message in messages
        for call in message.get("tool_calls", [])
    }
    defined_names = {tool["function"]["name"] for tool in tools}
    if called_names - defined_names:
        raise AdapterError(
            "Agent-FLAN row contains semantic tool calls but no matching tool definitions: "
            f"{sorted(called_names - defined_names)}"
        )
    return make_trajectory(
        trajectory_id=trajectory_id(DATASET_ID, row, index),
        source_dataset="agent_flan",
        task_type="generic_agent_tool_behavior" if called_names else "generic_no_tool_behavior",
        messages=messages,
        tools=tools,
        difficulty=str(row.get("difficulty") or "medium"),
        provenance=provenance(
            dataset_id=DATASET_ID,
            split=split,
            row=row,
            index=index,
            license_name="Apache-2.0",
            transformations=["roles_to_canonical", "reasoning_preserved" if include_reasoning else "reasoning_removed"],
        ),
    )
