from __future__ import annotations

import random
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.common.jsonl import read_jsonl


@dataclass(frozen=True)
class DatasetSplits:
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]]
    test: list[dict[str, Any]]


def load_messages(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for idx, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Row {idx} is missing a chat-style messages list")
    return rows


def split_rows(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.90,
    eval_ratio: float = 0.05,
    seed: int = 42,
    max_samples: int | None = None,
    group_key: str | None = None,
    stratify_key: str | None = None,
) -> DatasetSplits:
    if not 0 <= train_ratio <= 1 or not 0 <= eval_ratio <= 1 or train_ratio + eval_ratio > 1:
        raise ValueError("invalid split ratios")

    def canonical_group(row: dict[str, Any]) -> str:
        for key in ([group_key] if group_key else []) + ["group_id", "original_sample_id", "trajectory_id"]:
            value = row.get(key) if key else None
            if value:
                return str(value)
        try:
            question, _ = first_user_assistant(row)
            question = question.split("Tài liệu tham khảo:", 1)[0]
            if question.lstrip().lower().startswith("câu hỏi:"):
                question = question.split(":", 1)[1]
            payload = " ".join(question.lower().split())
        except ValueError:
            payload = str(row.get("id") or json.dumps(row, ensure_ascii=False, sort_keys=True))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(canonical_group(row), []).append(row)
    group_ids = list(grouped)
    random.Random(seed).shuffle(group_ids)

    if max_samples is not None and max_samples <= 0:
        group_ids = []
    elif max_samples is not None:
        selected: list[str] = []
        selected_rows = 0
        minimum_groups = min(3, len(group_ids))
        for group_id in group_ids:
            size = len(grouped[group_id])
            selected.append(group_id)
            selected_rows += size
            # Whole groups are indivisible. Permit a deterministic overshoot so a
            # smoke split still has train/eval/test groups without leakage.
            if selected_rows >= max_samples and len(selected) >= minimum_groups:
                break
        group_ids = selected

    n_groups = len(group_ids)
    n_train = int(n_groups * train_ratio)
    n_eval = int(n_groups * eval_ratio)
    if n_groups >= 3:
        n_train = min(max(n_train, 1), n_groups - 2)
        n_eval = min(max(n_eval, 1), n_groups - n_train - 1)
    n_test = n_groups - n_train - n_eval
    if stratify_key and n_eval and n_test:
        eval_ids, test_ids = _stratified_group_holdouts(
            group_ids,
            grouped,
            eval_groups=n_eval,
            test_groups=n_test,
            stratify_key=stratify_key,
            seed=seed,
        )
        held_out = set(eval_ids) | set(test_ids)
        train_ids = [group_id for group_id in group_ids if group_id not in held_out]
    else:
        train_ids = group_ids[:n_train]
        eval_ids = group_ids[n_train : n_train + n_eval]
        test_ids = group_ids[n_train + n_eval :]

    def flatten(ids: list[str]) -> list[dict[str, Any]]:
        return [row for group_id in ids for row in grouped[group_id]]

    return DatasetSplits(train=flatten(train_ids), eval=flatten(eval_ids), test=flatten(test_ids))


def _stratified_group_holdouts(
    group_ids: list[str],
    grouped: dict[str, list[dict[str, Any]]],
    *,
    eval_groups: int,
    test_groups: int,
    stratify_key: str,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Choose whole-group holdouts with deterministic class coverage scoring."""
    classes = sorted({str(row.get(stratify_key) or "unknown") for group_id in group_ids for row in grouped[group_id]})
    global_counts = {
        cls: sum(str(row.get(stratify_key) or "unknown") == cls for group_id in group_ids for row in grouped[group_id])
        for cls in classes
    }
    total_rows = sum(global_counts.values())
    global_shares = {cls: count / max(total_rows, 1) for cls, count in global_counts.items()}
    groups_per_class = {
        cls: sum(any(str(row.get(stratify_key) or "unknown") == cls for row in grouped[group_id]) for group_id in group_ids)
        for cls in classes
    }

    def subset_score(ids: tuple[str, ...]) -> tuple[float, ...]:
        rows = [row for group_id in ids for row in grouped[group_id]]
        counts = {
            cls: sum(str(row.get(stratify_key) or "unknown") == cls for row in rows)
            for cls in classes
        }
        covered = [cls for cls, count in counts.items() if count]
        rare_coverage = sum(1.0 / max(groups_per_class[cls], 1) for cls in covered)
        shares = {cls: counts[cls] / max(len(rows), 1) for cls in classes}
        distribution_error = sum(abs(shares[cls] - global_shares[cls]) for cls in classes)
        target_rows = total_rows * len(ids) / max(len(group_ids), 1)
        row_error = abs(len(rows) - target_rows) / max(target_rows, 1.0)
        tie = int(hashlib.sha256(f"{seed}:{'|'.join(sorted(ids))}".encode()).hexdigest()[:12], 16)
        return (len(covered), rare_coverage, -distribution_error, -row_error, -tie)

    def candidates(count: int) -> list[tuple[tuple[str, ...], tuple[float, ...]]]:
        combination_count = 1
        for index in range(count):
            combination_count = combination_count * (len(group_ids) - index) // (index + 1)
        if count <= 4 and combination_count <= 100_000:
            values = list(itertools.combinations(group_ids, count))
        else:
            # Large holdouts use a bounded deterministic greedy pool.
            chosen: list[str] = []
            remaining = list(group_ids)
            while len(chosen) < count:
                best = max(remaining, key=lambda value: subset_score(tuple(chosen + [value])))
                chosen.append(best)
                remaining.remove(best)
            values = [tuple(chosen)]
        ranked = sorted(((value, subset_score(value)) for value in values), key=lambda item: item[1], reverse=True)
        return ranked[:500]

    eval_candidates = candidates(eval_groups)
    test_candidates = candidates(test_groups)
    best_pair: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    best_score: tuple[float, ...] | None = None
    for eval_ids, eval_score in eval_candidates:
        eval_set = set(eval_ids)
        for test_ids, test_score in test_candidates:
            if eval_set.intersection(test_ids):
                continue
            pair_score = (
                min(eval_score[0], test_score[0]),
                eval_score[0] + test_score[0],
                min(eval_score[1], test_score[1]),
                eval_score[1] + test_score[1],
                eval_score[2] + test_score[2],
                eval_score[3] + test_score[3],
                eval_score[4] + test_score[4],
            )
            if best_score is None or pair_score > best_score:
                best_score = pair_score
                best_pair = (eval_ids, test_ids)
    if best_pair is None:
        raise ValueError("could not allocate disjoint stratified group holdouts")
    return list(best_pair[0]), list(best_pair[1])


def split_statistics(splits: DatasetSplits) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("train", "eval", "test"):
        rows = getattr(splits, name)
        groups = {str(row.get("group_id") or row.get("trajectory_id") or row.get("id")) for row in rows}
        classes: dict[str, int] = {}
        sources: dict[str, int] = {}
        for row in rows:
            cls = str(
                row.get("trajectory_class")
                or row.get("behavior")
                or row.get("trajectory", {}).get("trajectory_class")
                or "unknown"
            )
            source = str(row.get("source_dataset") or row.get("source") or "unknown")
            classes[cls] = classes.get(cls, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        result[name] = {"rows": len(rows), "unique_groups": len(groups), "classes": classes, "sources": sources}
    return result


def first_user_assistant(row: dict[str, Any]) -> tuple[str, str]:
    user_text = ""
    assistant_text = ""
    for message in row.get("messages", []):
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", ""))
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant" and not assistant_text:
            assistant_text = content
    if not user_text or not assistant_text:
        raise ValueError(f"Row {row.get('id', '<unknown>')} must contain user and assistant messages")
    return user_text, assistant_text



