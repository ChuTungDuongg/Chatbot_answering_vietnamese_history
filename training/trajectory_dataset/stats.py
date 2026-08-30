from __future__ import annotations

import json
import math
from collections import Counter
from statistics import mean, median
from typing import Any, Iterable


def _describe(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(mean(ordered), 3),
        "median": round(median(ordered), 3),
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
    }


def dataset_stats(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    sources = Counter(str(row.get("source_dataset") or "unknown") for row in materialized)
    tasks = Counter(str(row.get("task_type") or "unknown") for row in materialized)
    tool_counts: list[int] = []
    answer_lengths: list[int] = []
    turn_counts: list[int] = []
    observation_chars: list[int] = []
    observation_result_counts: list[int] = []
    for row in materialized:
        messages = row.get("messages") or []
        tool_counts.append(sum(len(message.get("tool_calls") or []) for message in messages))
        answers = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "assistant" and str(message.get("content") or "").strip()
        ]
        answer_lengths.append(len(answers[-1].split()) if answers else 0)
        turn_counts.append(len(messages))
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            observation_chars.append(len(content))
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = []
            observation_result_counts.append(len(payload) if isinstance(payload, list) else 0)
    total = len(materialized)
    with_tools = sum(count > 0 for count in tool_counts)
    multi_step = sum(count > 1 for count in tool_counts)
    return {
        "total_rows": total,
        "rows_per_source": dict(sorted(sources.items())),
        "rows_per_task_type": dict(sorted(tasks.items())),
        "tool_usage": {
            "with_tools": with_tools,
            "without_tools": total - with_tools,
            "tool_ratio": round(with_tools / total, 4) if total else 0.0,
            "single_tool": sum(count == 1 for count in tool_counts),
            "multi_tool": multi_step,
            "multi_step_ratio": round(multi_step / total, 4) if total else 0.0,
        },
        "answer_length_words": _describe(answer_lengths),
        "trajectory_turn_count": _describe(turn_counts),
        "observation_chars": _describe(observation_chars),
        "observation_result_counts": _describe(observation_result_counts),
    }
