from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.agents.history_contract import (
    build_history_answerer_messages,
    parse_history_answer_output,
    parse_history_training_user_text,
)


DATASET_PATH = Path("datasets/history_answerer/train.jsonl")
REPLAY_TYPES = (
    "grounded_qa",
    "noisy_context",
    "insufficient_context",
    "false_premise",
)


def _golden_rows():
    grouped = defaultdict(list)
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_type = str(row.get("type"))
            if row_type in REPLAY_TYPES and len(grouped[row_type]) < 10:
                grouped[row_type].append(row)
    assert all(len(grouped[row_type]) == 10 for row_type in REPLAY_TYPES)
    return grouped


def test_canonical_history_builder_exactly_replays_40_training_prompts():
    for row_type, rows in _golden_rows().items():
        for row in rows:
            assert [message["role"] for message in row["messages"]] == [
                "user",
                "assistant",
            ]
            user_text = row["messages"][0]["content"]
            question, evidence = parse_history_training_user_text(user_text)
            runtime_messages = build_history_answerer_messages(question, evidence)

            assert runtime_messages == [{"role": "user", "content": user_text}], row_type
            assert [item["chunk_id"] for item in evidence]


def test_canonical_history_outputs_preserve_the_runtime_citation_contract():
    for rows in _golden_rows().values():
        for row in rows:
            question, evidence = parse_history_training_user_text(
                row["messages"][0]["content"]
            )
            del question
            allowed = [item["chunk_id"] for item in evidence]
            parsed = parse_history_answer_output(
                row["messages"][1]["content"],
                allowed_source_ids=allowed,
            )
            assert set(parsed.source_ids) <= set(allowed)
            assert parsed.answer
