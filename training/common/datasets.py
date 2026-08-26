from __future__ import annotations

import random
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
) -> DatasetSplits:
    if max_samples is not None:
        rows = rows[: max(0, max_samples)]

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_eval = int(n_total * eval_ratio)

    return DatasetSplits(
        train=shuffled[:n_train],
        eval=shuffled[n_train : n_train + n_eval],
        test=shuffled[n_train + n_eval :],
    )


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



