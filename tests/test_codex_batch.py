"""Tiny fixtures for manually curated Codex SILVER batch safety."""

import pytest

from evaluation.annotation import codex_batch
from evaluation.schema import load_questions


class TinyCorpus:
    sha256 = "tiny-corpus-sha"
    count = 2

    def __init__(self):
        self.rows = {
            "chk_1": {"chunk_id": "chk_1", "source_id": "src_1", "text": "Văn Miếu được xây dựng năm 1070."},
            "chk_2": {"chunk_id": "chk_2", "source_id": "src_2", "text": "Quốc Tử Giám được lập năm 1076."},
        }

    def chunk(self, chunk_id):
        return self.rows.get(chunk_id)

    def has_source(self, source_id):
        return any(row["source_id"] == source_id for row in self.rows.values())


def entry(number, chunk_id):
    row = TinyCorpus().chunk(chunk_id)
    question = ("Văn Miếu được xây dựng vào năm nào?" if number == 1
                else "Quốc Tử Giám được lập năm mấy?")
    record = {
        "id": f"vn_hist_silver_{number:04d}", "question": question,
        "category": "chronology", "difficulty": "easy", "question_type": "chronology",
        "gold_answer": row["text"], "required_facts": [row["text"]],
        "relevant_chunk_ids": [chunk_id], "relevant_source_ids": [row["source_id"]],
        "gold_citation_source_ids": [row["source_id"]], "factual_paragraph_indices": [0],
        "answerable": True, "in_domain": True,
        "notes": "Codex SILVER; no human review.", "annotation_origin": "automatic",
        "annotation_status": "auto_reviewed", "confidence": "high",
    }
    audit = {"evidence": [{"chunk_id": chunk_id, "decision": "relevant", "quote": row["text"],
                            "supports_fact_indices": [0]}],
             "pass_b": "Independently checked the single claim against the quoted chunk."}
    return record, audit


def test_batch_persists_and_merges_with_silver_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_batch, "BATCH_SIZE", 2)
    monkeypatch.setattr(codex_batch, "DIFFICULTY_QUOTA", {"easy": 2})
    store = codex_batch.CodexBatchStore(tmp_path / "workspace", TinyCorpus())
    master = tmp_path / "questions_1000.jsonl"
    legacy = tmp_path / "questions_500.jsonl"
    legacy.write_text("legacy snapshot\n", encoding="utf-8")
    try:
        first = entry(1, "chk_1")
        store.add(*first)
        assert store.status()["current_saved"] == 1
        store.close()
        store = codex_batch.CodexBatchStore(tmp_path / "workspace", TinyCorpus())
        assert store.status()["current_saved"] == 1
        store.add(*entry(2, "chk_2"))
        assert store.complete(master) == 2
        assert store.complete(master) == 2  # retry after publish/merge interruption
        rows = load_questions(master)
        assert [row.id for row in rows] == ["vn_hist_silver_0001", "vn_hist_silver_0002"]
        assert all(row.annotation_origin == "automatic" and row.review_status is None for row in rows)
        assert store.status()["completed_batches"] == 1
        assert legacy.read_text(encoding="utf-8") == "legacy snapshot\n"
    finally:
        store.close()


def test_thousand_target_keeps_batch_eleven_available_and_stops_after_twenty(tmp_path):
    assert codex_batch.TARGET_QUESTIONS == 1000
    assert codex_batch.MAX_BATCHES == 20
    assert codex_batch.MASTER.name == "questions_1000.jsonl"
    assert f"{codex_batch.ID_PREFIX}{1000:04d}" == "vn_hist_silver_1000"
    store = codex_batch.CodexBatchStore(tmp_path / "workspace", TinyCorpus())
    try:
        batch_dir = store.workspace / "batches"
        batch_dir.mkdir()
        for number in range(1, 11):
            store._batch_path(number).touch()
        assert store.completed() == 10
        assert store.next_batch() == 11  # 500 is halfway, not terminal.
        for number in range(11, 21):
            store._batch_path(number).touch()
        assert store.completed() == 20
        assert store.status()["next_batch"] is None
        with pytest.raises(RuntimeError, match="complete"):
            store.next_batch()
        store._batch_path(21).touch()
        with pytest.raises(RuntimeError, match="exceeds"):
            store.completed()
    finally:
        store.close()


def test_batch_rejects_wrong_difficulty_mix(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_batch, "BATCH_SIZE", 2)
    monkeypatch.setattr(codex_batch, "DIFFICULTY_QUOTA", {"easy": 2})
    store = codex_batch.CodexBatchStore(tmp_path / "workspace", TinyCorpus())
    try:
        store.add(*entry(1, "chk_1"))
        record, audit = entry(2, "chk_2")
        record["difficulty"] = "medium"
        store.add(record, audit)
        with pytest.raises(ValueError, match="difficulty counts"):
            store.complete(tmp_path / "master.jsonl")
        assert not (tmp_path / "master.jsonl").exists()
    finally:
        store.close()


def test_rejects_human_provenance_unknown_chunk_and_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_batch, "BATCH_SIZE", 2)
    monkeypatch.setattr(codex_batch, "DIFFICULTY_QUOTA", {"easy": 2})
    store = codex_batch.CodexBatchStore(tmp_path / "workspace", TinyCorpus())
    try:
        record, audit = entry(1, "chk_1")
        record["annotation_origin"] = "human"
        with pytest.raises(ValueError, match="automatic SILVER"):
            store.add(record, audit)
        record["annotation_origin"] = "automatic"
        record["relevant_chunk_ids"] = ["missing"]
        with pytest.raises(ValueError, match="Relevant chunk absent"):
            store.add(record, audit)
        store.add(*entry(1, "chk_1"))
        record, audit = entry(2, "chk_2")
        record["question"] = entry(1, "chk_1")[0]["question"]
        with pytest.raises(ValueError, match="Duplicate"):
            store.add(record, audit)
        assert store.status()["current_saved"] == 1
    finally:
        store.close()
