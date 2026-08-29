from __future__ import annotations

from typing import Any

from ..schema import RETRIEVE_TOOL, make_trajectory
from .common import get_messages, provenance, semantic_messages, trajectory_id


DATASET_ID = "khaimaitien/multi-hop-qa-function-calling-format-V1.0"


def normalize_multihop(
    row: dict[str, Any],
    *,
    index: int = 0,
    split: str = "train",
    include_reasoning: bool = False,
) -> dict[str, Any]:
    # `retrieve` is the dataset's documented generic function. It must remain
    # generic and is intentionally not renamed to project-specific search_history.
    messages = semantic_messages(
        get_messages(row),
        include_reasoning=include_reasoning,
        legacy_function_name="retrieve",
    )
    call_count = sum(len(message.get("tool_calls") or []) for message in messages)
    if not call_count:
        raise ValueError("multi-hop adapter row contains no retrieval function call")
    return make_trajectory(
        trajectory_id=trajectory_id(DATASET_ID, row, index),
        source_dataset="multi_hop_function_calling",
        task_type="multi_hop_retrieval" if call_count > 1 else "single_step_retrieval",
        messages=messages,
        tools=[RETRIEVE_TOOL],
        difficulty="hard" if call_count > 1 else "medium",
        provenance=provenance(
            dataset_id=DATASET_ID,
            split=split,
            row=row,
            index=index,
            license_name=row.get("license"),
            transformations=["legacy_function_call_to_tool_calls", "preserved_generic_retrieve_tool"],
        ),
    )
