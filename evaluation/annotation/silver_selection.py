"""Deterministic target planning, diversity checks, and deep/audit sampling."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from typing import Any

from evaluation.annotation.generate_candidates import scaled_targets
from evaluation.annotation.policy import fold, similar


FAMOUS = re.compile(r"dien bien phu|ho chi minh|cach mang thang tam|chien tranh viet nam|nha tran")


def usable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["annotation_status"] == "auto_reviewed"
            and row["confidence"] in {"high", "medium"}]


def next_category(rows: list[dict[str, Any]], config: dict[str, Any], target: int) -> str:
    targets = scaled_targets(config, target)
    counts = Counter(row["generated"]["category"] for row in usable(rows))
    needed = [key for key, value in targets.items() if counts[key] < value]
    if not needed:
        raise ValueError("SILVER category target already satisfied")
    return min(needed, key=lambda key: (counts[key] / max(1, targets[key]), key))


def next_difficulty(rows: list[dict[str, Any]], config: dict[str, Any], target: int,
                    category: str) -> str:
    if category in {"comparison", "multi_hop"}:
        return "hard"
    raw = config["difficulty_targets"]
    counts = Counter(row["generated"]["difficulty"] for row in usable(rows))
    return min(raw, key=lambda key: (counts[key] / max(1, raw[key] * target / config["base_target"]), key))


def select_seed(category: str, pools: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]],
                config: dict[str, Any]) -> dict[str, Any] | None:
    if category == "out_of_domain":
        return None
    sources = Counter(row["seed"].get("source_id") for row in rows if row["seed"].get("source_id"))
    titles = Counter(fold(row["seed"].get("title") or "") for row in rows if row["seed"].get("title"))
    used_chunks = {row["seed"].get("chunk_id") for row in rows}
    famous = sum(bool(FAMOUS.search(fold(row["seed"].get("title") or ""))) for row in rows)
    choices = [*pools.get(category, []), *pools.get("_general", [])]
    for seed in choices:
        title = fold(seed["title"])
        if seed["chunk_id"] in used_chunks or sources[seed["source_id"]] >= config["max_candidates_per_source"]:
            continue
        if titles[title] >= config["max_candidates_per_title"]:
            continue
        if FAMOUS.search(title) and famous >= config["max_famous_topics"]:
            continue
        return seed
    return None


def duplicate_reason(question: str, rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        other = (row.get("generated") or {}).get("question")
        if other and similar(question, other):
            return "duplicate_or_near_duplicate"
    return None


def balanced_subset(rows: list[dict[str, Any]], size: int, seed: int = 2026) -> list[dict[str, Any]]:
    """Greedy coverage of category, difficulty, question type, failure and backend behavior."""
    pool = list(rows)
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    while pool and len(chosen) < size:
        def features(row: dict[str, Any]) -> list[str]:
            generated, labels = row["generated"], row["labels"]
            diag = labels["diagnostics"]
            return [f"category:{generated['category']}", f"difficulty:{generated['difficulty']}",
                    f"type:{generated.get('question_type')}", f"answerable:{labels['answerable']}",
                    f"miss:{diag['retrieval_miss']}",
                    f"disagreement:{diag['backend_disagreement']}"]
        pick = max(pool, key=lambda row: (sum(1 / (1 + counts[item]) for item in features(row)),
                                          hashlib.sha256(row["candidate_id"].encode()).hexdigest()))
        pool.remove(pick)
        chosen.append(pick)
        counts.update(features(pick))
    return chosen


def audit_queue(rows: list[dict[str, Any]], size: int, risky_size: int,
                seed: int = 2026) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("labels")]
    def risk(row: dict[str, Any]) -> int:
        labels, generated = row["labels"], row["generated"]
        diag = labels["diagnostics"]
        deep = row.get("deep") or {}
        return (20 * (row["confidence"] == "low") +
                8 * diag["backend_disagreement"] + 8 * diag["retrieval_miss"] +
                10 * labels["contradiction"]["contradiction_detected"] +
                6 * (generated["category"] in {"multi_hop", "comparison", "false_premise", "ambiguous"}) +
                6 * (deep.get("all_supported") is False) +
                4 * (len(labels["relevant_source_ids"]) > 1))
    risky = sorted(candidates, key=lambda row: (-risk(row), row["candidate_id"]))[:min(risky_size, size)]
    selected = {row["candidate_id"] for row in risky}
    safe = [row for row in candidates if row["candidate_id"] not in selected and
            row["confidence"] in {"high", "medium"}]
    random.Random(seed).shuffle(safe)
    selected_rows = risky + safe[:max(0, size - len(risky))]
    if len(selected_rows) < size:
        selected |= {row["candidate_id"] for row in selected_rows}
        selected_rows += [row for row in candidates if row["candidate_id"] not in selected][:size-len(selected_rows)]
    return selected_rows
