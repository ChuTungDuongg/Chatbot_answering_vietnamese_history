from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from .dedup import first_user_question, normalized_question


@dataclass(frozen=True)
class TrajectorySplits:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]


def source_groups(row: dict[str, Any]) -> list[str]:
    provenance = row.get("provenance") or {}
    values = provenance.get("source_groups")
    if isinstance(values, list):
        groups = [str(value) for value in values if value not in {None, ""}]
        if groups:
            return list(dict.fromkeys(groups))
    for value in (
        provenance.get("source_group"),
        provenance.get("source_document_id"),
        provenance.get("article_id"),
        row.get("group_id"),
    ):
        if value not in {None, ""}:
            return [str(value)]
    question = normalized_question(first_user_question(row))
    payload = question or str(row.get("id") or "")
    return [hashlib.sha256(payload.encode("utf-8")).hexdigest()]


def source_group(row: dict[str, Any]) -> str:
    """Backward-compatible primary source group accessor."""
    return source_groups(row)[0]


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


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
    union_find = _UnionFind(len(rows))
    first_row_for_group: dict[str, int] = {}
    for index, row in enumerate(rows):
        for group in source_groups(row):
            if group in first_row_for_group:
                union_find.union(index, first_row_for_group[group])
            else:
                first_row_for_group[group] = index

    components: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        components.setdefault(union_find.find(index), []).append(row)
    component_ids = sorted(
        components,
        key=lambda root: min(str(row.get("id") or "") for row in components[root]),
    )
    random.Random(seed).shuffle(component_ids)
    assigned: dict[int, str] = {}
    if preserve_official_splits:
        validation_names = {"validation", "valid", "eval", "dev"}
        for root in component_ids:
            official = {
                str((row.get("provenance") or {}).get("original_split") or "").lower()
                for row in components[root]
            } - {"", "train"}
            has_test = "test" in official
            has_validation = bool(official & validation_names)
            if has_test and has_validation:
                raise ValueError("shared source groups connect conflicting official test and validation rows")
            if has_test:
                assigned[root] = "test"
            elif has_validation:
                assigned[root] = "validation"

    unassigned = [root for root in component_ids if root not in assigned]
    count = len(unassigned)
    n_train = int(count * train_ratio)
    n_validation = int(count * validation_ratio)
    if count >= 3 and all(ratio > 0 for ratio in (train_ratio, validation_ratio, test_ratio)):
        n_train = min(max(n_train, 1), count - 2)
        n_validation = min(max(n_validation, 1), count - n_train - 1)
    for root in unassigned[:n_train]:
        assigned[root] = "train"
    for root in unassigned[n_train:n_train + n_validation]:
        assigned[root] = "validation"
    for root in unassigned[n_train + n_validation:]:
        assigned[root] = "test"

    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for root in component_ids:
        result[assigned[root]].extend(components[root])
    return TrajectorySplits(**result)
