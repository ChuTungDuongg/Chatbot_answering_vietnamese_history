"""Descriptive statistics for saved HTTP latency records."""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


HEADLINE_METRICS = (
    "ttfb_ms", "first_status_event_ms", "answer_ttft_ms", "model_ttft_ms",
    "retrieval_ms", "prompt_build_ms", "generation_ms", "e2e_ms",
    "tpot_ms", "tokens_per_second", "decode_tokens_per_second",
    "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
    "model_calls", "tool_calls", "tool_execution_ms", "tool_parse_failures",
    "action_rounds", "time_until_final_generation_ms", "final_answer_ttft_ms",
    "planning_model_ms", "planning_model_ttft_ms", "planning_wall_ms",
    "planning_input_tokens", "planning_output_tokens", "dense_search_ms", "output_tokens",
)


def distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    data = sorted(float(value) for value in values
                  if isinstance(value, (float, int)) and not isinstance(value, bool)
                  and math.isfinite(value))
    if not data:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None, "p99": None}

    def percentile(p: float) -> float:
        index = (len(data) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return data[lo] + (data[hi] - data[lo]) * (index - lo)

    return {"count": len(data), "mean": statistics.mean(data),
            "std": statistics.pstdev(data), "p50": percentile(.5),
            "p95": percentile(.95), "p99": percentile(.99)}


def summarize(records: list[dict[str, Any]], *, phase_durations_s: dict[str, float] | None = None) -> dict[str, Any]:
    """Cold, warm, and warmup observations are always separate."""
    groups: dict[str, Any] = {}
    for phase in ("cold", "warm", "warmup"):
        rows = [row for row in records if row.get("phase") == phase]
        successful = [row for row in rows if row.get("success") is True]
        count = len(rows)
        wall = (phase_durations_s or {}).get(phase)
        groups[phase] = {
            "count": count,
            "success_count": len(successful),
            "error_count": count - len(successful),
            "success_rate": len(successful) / count if count else None,
            "error_rate": (count - len(successful)) / count if count else None,
            "wall_seconds": wall,
            "successful_requests_per_second": len(successful) / wall if wall and wall > 0 else None,
            "output_tokens_per_second": sum(row.get("output_tokens") or 0 for row in successful) / wall
                if wall and wall > 0 and all(row.get("output_tokens") is not None for row in successful) else None,
            "metrics": {name: distribution(row.get(name) for row in successful)
                        for name in HEADLINE_METRICS},
        }
    return {"schema_version": 1, "groups": groups}
