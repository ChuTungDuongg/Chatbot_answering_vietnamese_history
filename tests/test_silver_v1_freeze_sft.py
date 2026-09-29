"""Tiny, model-free tests for the frozen SILVER split and RAG-SFT derivation."""

from __future__ import annotations

from collections import Counter

import pytest

from app.rag.response_modes import MODE_INSTRUCTIONS, mode_instruction
from app.rag.prompting import build_messages
from evaluation.annotation.codex_batch import CodexBatchStore
from evaluation.silver_v1 import (
    assign_modes, build_sft, extra_modes, group_aware_split, sft_answer, sft_row,
    validate_sft, validate_split, write_frozen,
)


class TinyCorpus:
    def __init__(self):
        self.rows = {
            f"chk_{i}": {"chunk_id": f"chk_{i}", "source_id": f"src_{i}",
                         "document_id": f"doc_{i}", "title": f"Nguồn {i}",
                         "text": f"Dữ kiện thực từ nguồn {i}."}
            for i in range(1, 16)
        }
        self.sha256 = "tiny-sha"
        self.count = len(self.rows)

    def chunk(self, chunk_id):
        return self.rows.get(chunk_id)

    def has_source(self, source_id):
        return any(row["source_id"] == source_id for row in self.rows.values())


def record(number: int, *, source: int | None = None) -> dict:
    source = source or number
    return {"id": f"q{number:02d}", "question": f"Câu hỏi {number}?",
            "gold_answer": f"Dữ kiện thực từ nguồn {source}.",
            "required_facts": [f"Dữ kiện thực từ nguồn {source}."],
            "relevant_chunk_ids": [f"chk_{source}"],
            "relevant_source_ids": [f"src_{source}"],
            "difficulty": ("easy", "medium", "hard")[(number-1) % 3],
            "category": "chronology" if number % 2 else "comparison",
            "question_type": "false_premise" if number == 9 else "chronology",
            "answerable": number != 9, "in_domain": True}


def test_source_group_split_is_deterministic_and_disjoint():
    rows = [record(i, source=(1 if i in {1, 2, 3} else i)) for i in range(1, 13)]
    first = group_aware_split(rows, seed=42)
    assert first == group_aware_split(list(reversed(rows)), seed=42)
    validate_split(rows, first)
    owners = {name for name, items in first.items() if any(row["id"] == "q01" for row in items)}
    assert len(owners) == 1
    owner = owners.pop()
    assert {"q01", "q02", "q03"} <= {row["id"] for row in first[owner]}
    assert set(row["id"] for items in first.values() for row in items) == set(row["id"] for row in rows)


def test_split_validator_rejects_source_or_id_leakage():
    rows = [record(1), record(2, source=1), record(3)]
    with pytest.raises(ValueError, match="Primary source"):
        validate_split(rows, {"train": [rows[0]], "validation": [rows[1]], "test": [rows[2]]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_split(rows, {"train": [rows[0]], "validation": [rows[0], rows[1]],
                              "test": [rows[2]]})


def test_mode_assignments_and_variant_limits_are_reproducible():
    ids = [f"q{i:02d}" for i in range(1, 13)]
    base = assign_modes(ids)
    assert base == assign_modes(list(reversed(ids)))
    assert Counter(base.values()) == {"concise": 4, "standard": 4, "detailed": 4}
    second = extra_modes(base, count=6)
    assert second == extra_modes(base, count=6)
    assert len(second) == 6 and all(base[identifier] != mode for identifier, mode in second.items())
    assert Counter(base.values()) + Counter(second.values()) == {
        "concise": 6, "standard": 6, "detailed": 6}
    assert set(MODE_INSTRUCTIONS) == {"concise", "standard", "detailed"}
    with pytest.raises(ValueError):
        mode_instruction("compact")
    prompt = build_messages("Một câu hỏi", [], response_mode="detailed")
    assert MODE_INSTRUCTIONS["detailed"] in prompt[0]["content"]


def test_sft_uses_real_chunk_text_and_keeps_test_held_out():
    corpus = TinyCorpus()
    assignments = {"train": [record(1), record(2), record(3)],
                   "validation": [record(4)], "test": [record(5)]}
    train, validation = build_sft(assignments, corpus, extra_count=0)
    assert len(train) == 3 and len(validation) == 1
    assert {row["canonical_id"] for row in train} == {"q01", "q02", "q03"}
    assert "Dữ kiện thực từ nguồn" in train[0]["messages"][1]["content"]
    assert train[0]["messages"][0]["role"] == "system"
    assert train[0]["messages"][1]["role"] == "user"
    assert train[0]["messages"][2]["role"] == "assistant"
    assert MODE_INSTRUCTIONS[train[0]["response_mode"]] in train[0]["messages"][0]["content"]
    assert validate_sft(assignments, train, validation, corpus)["test_ids_in_sft"] == 0
    with pytest.raises(ValueError, match="leakage"):
        validate_sft(assignments, train + [sft_row(assignments["test"][0], "concise", corpus)],
                     validation, corpus)
    with pytest.raises(ValueError, match="leakage"):
        validate_sft(assignments, train + [sft_row(assignments["validation"][0], "concise", corpus)],
                     validation, corpus)
    changed = [dict(item) for item in train]
    changed[0]["messages"] = [dict(message) for message in changed[0]["messages"]]
    changed[0]["messages"][1]["content"] = "Nguồn giả."
    with pytest.raises(ValueError, match="Ungrounded"):
        validate_sft(assignments, changed, validation, corpus)


def test_unanswerable_target_never_gains_invented_answer():
    item = record(9)
    item["gold_answer"] = "Nguồn hiện có chưa đủ để xác nhận tiền đề."
    for mode in MODE_INSTRUCTIONS:
        assert sft_answer(item, mode) == item["gold_answer"]


def test_max_two_variants_and_frozen_file(tmp_path):
    corpus = TinyCorpus()
    assignments = {"train": [record(1)], "validation": [record(2)], "test": [record(3)]}
    train = [sft_row(assignments["train"][0], mode, corpus) for mode in MODE_INSTRUCTIONS]
    validation = [sft_row(assignments["validation"][0], "standard", corpus)]
    with pytest.raises(ValueError, match="More than two"):
        validate_sft(assignments, train, validation, corpus)
    path = tmp_path / "frozen.jsonl"
    write_frozen(path, b"same\n")
    write_frozen(path, b"same\n")
    with pytest.raises(RuntimeError, match="Frozen output differs"):
        write_frozen(path, b"changed\n")


def test_frozen_workspace_refuses_batch_61(tmp_path):
    corpus = TinyCorpus()
    workspace = tmp_path / "workspace"
    store = CodexBatchStore(workspace, corpus)
    try:
        (workspace / "frozen_v1.json").write_text("{}", encoding="utf-8")
        assert store.status()["frozen_v1"] is True
        assert store.status()["next_batch"] is None
        with pytest.raises(RuntimeError, match="no Batch 61"):
            store.next_batch()
    finally:
        store.close()
