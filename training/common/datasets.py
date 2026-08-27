from __future__ import annotations

import random
import hashlib
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
    train_ids = group_ids[:n_train]
    eval_ids = group_ids[n_train : n_train + n_eval]
    test_ids = group_ids[n_train + n_eval :]

    def flatten(ids: list[str]) -> list[dict[str, Any]]:
        return [row for group_id in ids for row in grouped[group_id]]

    return DatasetSplits(train=flatten(train_ids), eval=flatten(eval_ids), test=flatten(test_ids))


def split_statistics(splits: DatasetSplits) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("train", "eval", "test"):
        rows = getattr(splits, name)
        groups = {str(row.get("group_id") or row.get("trajectory_id") or row.get("id")) for row in rows}
        classes: dict[str, int] = {}
        sources: dict[str, int] = {}
        for row in rows:
            cls = str(row.get("trajectory_class") or row.get("trajectory", {}).get("trajectory_class") or "unknown")
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



