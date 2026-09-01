from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .dedup import first_user_question, normalized_question


IMPORTANT_CUSTOM_TASK_TYPES = (
    "factual", "cause", "significance", "compare", "summary", "multihop",
    "verification", "hard_negative", "insufficient_evidence",
)


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
    first_row_for_question: dict[str, int] = {}
    for index, row in enumerate(rows):
        for group in source_groups(row):
            if group in first_row_for_group:
                union_find.union(index, first_row_for_group[group])
            else:
                first_row_for_group[group] = index
        question_key = normalized_question(first_user_question(row))
        if question_key:
            if question_key in first_row_for_question:
                union_find.union(index, first_row_for_question[question_key])
            else:
                first_row_for_question[question_key] = index

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
    capacities = {
        "train": n_train,
        "validation": n_validation,
        "test": count - n_train - n_validation,
    }

    def component_features(root: int) -> tuple[set[str], set[str]]:
        component = components[root]
        tasks = {
            str(row.get("task_type") or "")
            for row in component
            if str(row.get("source_dataset") or "").startswith("custom_history")
        }
        sources = {str(row.get("source_dataset") or "unknown") for row in component}
        return tasks, sources

    covered_tasks = {name: set() for name in capacities}
    covered_sources = {name: set() for name in capacities}
    for root, name in assigned.items():
        tasks, sources = component_features(root)
        covered_tasks[name].update(tasks)
        covered_sources[name].update(sources)

    # Deterministic group-aware stratification.  It never splits a connected
    # source-group component; it only selects which whole component fills each
    # split slot, preferring missing custom behaviors and then missing sources.
    remaining = set(unassigned)
    rank = {root: index for index, root in enumerate(unassigned)}
    for name in ("validation", "test", "train"):
        while capacities[name] > 0 and remaining:
            root = max(
                remaining,
                key=lambda candidate: (
                    len(component_features(candidate)[0] - covered_tasks[name]),
                    len(component_features(candidate)[1] - covered_sources[name]),
                    -rank[candidate],
                ),
            )
            assigned[root] = name
            tasks, sources = component_features(root)
            covered_tasks[name].update(tasks)
            covered_sources[name].update(sources)
            capacities[name] -= 1
            remaining.remove(root)

    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for root in component_ids:
        result[assigned[root]].extend(components[root])
    return TrajectorySplits(**result)


def split_coverage_report(splits: TrajectorySplits) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}}
    groups_by_split: dict[str, set[str]] = {}
    questions_by_split: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        rows = getattr(splits, name)
        source_counts = Counter(str(row.get("source_dataset") or "unknown") for row in rows)
        task_counts = Counter(str(row.get("task_type") or "unknown") for row in rows)
        custom_counts = Counter(
            str(row.get("task_type") or "unknown")
            for row in rows
            if str(row.get("source_dataset") or "").startswith("custom_history")
        )
        groups_by_split[name] = {group for row in rows for group in source_groups(row)}
        questions_by_split[name] = {
            key for row in rows if (key := normalized_question(first_user_question(row)))
        }
        report["splits"][name] = {
            "rows": len(rows),
            "source_dataset_counts": dict(sorted(source_counts.items())),
            "task_type_counts": dict(sorted(task_counts.items())),
            "custom_history_task_type_counts": dict(sorted(custom_counts.items())),
            "missing_custom_task_types": [
                task for task in IMPORTANT_CUSTOM_TASK_TYPES if custom_counts[task] == 0
            ],
        }
    overlaps: dict[str, list[str]] = {}
    names = ("train", "validation", "test")
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = sorted(groups_by_split[left] & groups_by_split[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared[:20]
    report["source_group_leakage_count"] = sum(len(values) for values in overlaps.values())
    report["source_group_leakage"] = overlaps
    question_overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = sorted(questions_by_split[left] & questions_by_split[right])
            if shared:
                question_overlaps[f"{left}__{right}"] = shared[:20]
    report["normalized_question_leakage_count"] = sum(len(values) for values in question_overlaps.values())
    report["normalized_question_leakage"] = question_overlaps
    return report
