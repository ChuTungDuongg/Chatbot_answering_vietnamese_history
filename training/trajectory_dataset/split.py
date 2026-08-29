from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .dedup import first_user_question, normalized_question


@dataclass(frozen=True)
class TrajectorySplits:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]


def source_group(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") or {}
    for value in (
        provenance.get("source_group"),
        provenance.get("source_document_id"),
        provenance.get("article_id"),
        row.get("group_id"),
    ):
        if value not in {None, ""}:
            return str(value)
    question = normalized_question(first_user_question(row))
    payload = question or str(row.get("id") or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_trajectories(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.9,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    preserve_official_splits: bool = True,
) -> TrajectorySplits:
    if min(train_ratio, validation_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/validation/test ratios must sum to 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[source_group(row)].append(row)
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    assigned: dict[str, str] = {}
    if preserve_official_splits:
        for group in groups:
            official = {
                str((row.get("provenance") or {}).get("original_split") or "").lower()
                for row in grouped[group]
            }
            official.discard("")
            if official and official <= {"test"}:
                assigned[group] = "test"
            elif official and official <= {"validation", "valid", "eval", "dev"}:
                assigned[group] = "validation"

    unassigned = [group for group in groups if group not in assigned]
    total_unassigned = len(unassigned)
    n_train = int(total_unassigned * train_ratio)
    n_validation = int(total_unassigned * validation_ratio)
    if total_unassigned >= 3 and all(ratio > 0 for ratio in (train_ratio, validation_ratio, test_ratio)):
        n_train = min(max(n_train, 1), total_unassigned - 2)
        n_validation = min(max(n_validation, 1), total_unassigned - n_train - 1)
    for group in unassigned[:n_train]:
        assigned[group] = "train"
    for group in unassigned[n_train : n_train + n_validation]:
        assigned[group] = "validation"
    for group in unassigned[n_train + n_validation :]:
        assigned[group] = "test"

    result = {"train": [], "validation": [], "test": []}
    for group in groups:
        result[assigned[group]].extend(grouped[group])
    return TrajectorySplits(**result)
