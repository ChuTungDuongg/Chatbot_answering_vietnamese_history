from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any


DEFAULT_MIX_RATIOS = {
    "custom_history": 0.55,
    "multi_hop_function_calling": 0.17,
    "agent_flan": 0.12,
    "vietnam_history_200k": 0.16,
}
AGENT_FLAN_POOL_TARGET = 700


def normalize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if not ratios or any(value < 0 for value in ratios.values()):
        raise ValueError("mix ratios must be non-negative and contain at least one source")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("at least one mix ratio must be positive")
    return {key: value / total for key, value in ratios.items() if value > 0}


def mix_sources(
    sources: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    *,
    seed: int = 42,
    max_total: int | None = None,
) -> list[dict[str, Any]]:
    weights = normalize_ratios(ratios)
    missing = [name for name in weights if not sources.get(name)]
    if missing:
        raise ValueError(f"mix sources have no rows: {missing}")
    # Avoid losing an exactly feasible row to binary floating-point underflow
    # (for example, 2200 / 0.55 can evaluate just below 4000).
    feasible_total = min(
        math.floor((len(sources[name]) / weight) + 1e-9)
        for name, weight in weights.items()
    )
    target_total = feasible_total if max_total is None else min(feasible_total, max(max_total, 0))
    if target_total <= 0:
        return []
    counts = {name: int(target_total * weight) for name, weight in weights.items()}
    remaining = target_total - sum(counts.values())
    order = sorted(weights, key=lambda name: (-(target_total * weights[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    randomizer = random.Random(seed)
    mixed: list[dict[str, Any]] = []
    for name in sorted(counts):
        candidates = list(sources[name])
        randomizer.shuffle(candidates)
        mixed.extend(candidates[: counts[name]])
    randomizer.shuffle(mixed)
    return mixed


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source_dataset") or "unknown") for row in rows))


def required_source_rows(
    source: str, *, final_max_samples: int, ratios: dict[str, float] = DEFAULT_MIX_RATIOS,
) -> int:
    """Return a conservative source capacity gate for a requested final mix."""
    if final_max_samples < 0:
        raise ValueError("final_max_samples must be non-negative")
    weights = normalize_ratios(ratios)
    if source not in weights:
        raise ValueError(f"source {source!r} is absent from mix ratios")
    return math.ceil(final_max_samples * weights[source])


def agent_flan_pool_gate(
    available_rows: int,
    *,
    final_max_samples: int,
    pool_target: int = AGENT_FLAN_POOL_TARGET,
    ratios: dict[str, float] = DEFAULT_MIX_RATIOS,
) -> dict[str, Any]:
    required = required_source_rows(
        "agent_flan", final_max_samples=final_max_samples, ratios=ratios,
    )
    preferred_reached = available_rows >= pool_target
    quota_satisfied = available_rows >= required
    dedup_margin = max(20, math.ceil(required * 0.10)) if required else 0
    degraded_minimum = required + dedup_margin
    degraded_ready = available_rows >= degraded_minimum
    return {
        "valid": preferred_reached or degraded_ready,
        "available_rows": available_rows,
        "pool_target": pool_target,
        "preferred_target_reached": preferred_reached,
        "degraded_pool": degraded_ready and not preferred_reached,
        "final_max_samples": final_max_samples,
        "final_required_rows": required,
        "final_mix_quota_satisfied": quota_satisfied,
        "dedup_margin_rows": dedup_margin,
        "degraded_minimum_rows": degraded_minimum,
    }


def mix_capacity_report(
    sources: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    *,
    requested_total: int | None,
) -> dict[str, Any]:
    weights = normalize_ratios(ratios)
    required = {
        name: int(requested_total * weight) if requested_total is not None else None
        for name, weight in weights.items()
    }
    available = {name: len(sources.get(name) or []) for name in weights}
    insufficient = [
        name for name in weights
        if required[name] is not None and available[name] < int(required[name] or 0)
    ]
    return {
        "requested_total": requested_total,
        "ratios": weights,
        "required_rows": required,
        "available_safe_rows": available,
        "insufficient_sources": insufficient,
        "agent_flan_safe_pool_insufficient": "agent_flan" in insufficient,
        "duplicates_fabricated": False,
    }
