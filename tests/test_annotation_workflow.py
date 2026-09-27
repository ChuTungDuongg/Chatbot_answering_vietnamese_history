"""Tiny corpus only: no Qwen, Qdrant, full V1 scan, or external LLM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.export import export
from evaluation.annotation.generate_candidates import (
    Candidate, draft_candidates, evidence_from_result, hydrate_workspace, scaled_targets,
)
from evaluation.annotation.policy import duplicate_pairs
from evaluation.annotation.report import coverage
from evaluation.annotation.select_deep_subset import choose, retrieval_behavior, select
from evaluation.annotation.ui import create_app
from evaluation.annotation.validate_dataset import read_records, validate_records, workspace_records
from evaluation.annotation.workspace import Workspace
from evaluation.schema import load_questions


def fixture(tmp_path: Path):
    rows = [
        {"chunk_id": "c1", "source_id": "s1", "document_id": "d1", "chunk_index": 0,
         "title": "Trận Bạch Đằng", "text": "Trận Bạch Đằng năm 938 có vai trò quan trọng.",
         "url": "https://example.invalid/bach-dang", "years": ["938"], "section": "Trận đánh"},
        {"chunk_id": "c2", "source_id": "s1", "document_id": "d1", "chunk_index": 1,
         "title": "Trận Bạch Đằng", "text": "Nguồn này mô tả thêm diễn biến.",
         "url": "https://example.invalid/bach-dang", "years": [], "section": "Diễn biến"},
        {"chunk_id": "c3", "source_id": "s2", "document_id": "d2", "chunk_index": 0,
         "title": "Nhà Trần", "text": "Nhà Trần tồn tại qua nhiều giai đoạn.",
         "url": "https://example.invalid/nha-tran", "years": [], "section": "Triều đại"},
        {"chunk_id": "c4", "source_id": "s3", "document_id": "d3", "chunk_index": 0,
         "title": "Hiệp định Genève", "text": "Hiệp định Genève được ký vào năm 1954.",
         "url": "https://example.invalid/geneve", "years": ["1954"], "section": "Hiệp định"},
    ]
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    lookup = CorpusLookup.build(corpus, tmp_path / "workspace/corpus_lookup.sqlite3",
                                expected_sha=hashlib.sha256(corpus.read_bytes()).hexdigest(), expected_count=4)
    return corpus, lookup, rows


def candidate(identifier="x1", question="Trận Bạch Đằng có ý nghĩa gì?", category="battles",
              origin="c1", source="s1", title="Trận Bạch Đằng", difficulty="medium"):
    return Candidate.model_validate({"candidate_id": identifier, "question": question,
        "draft_question": question, "category": category, "difficulty": difficulty,
        "origin_chunk_id": origin, "origin_source_id": source, "origin_title": title,
        "draft_answer": "Bản nháp cần kiểm tra.", "candidate_required_facts": ["Bản nháp"],
        "generation_metadata": {"provider": "deterministic-template", "revision": "test"},
        "review_status": "pending"}).model_dump()


def accepted(workspace: Workspace, lookup: CorpusLookup, identifier="x1"):
    workspace.set_evidence(identifier, [{"stage": "dense", "rank": 1, "chunk_id": "c1",
        "source_id": "s1", "document_id": "d1", "score": .9, "details": {}}])
    return workspace.apply_review(identifier, {"answerable": True, "in_domain": True,
        "chunk_decisions": {"c1": "relevant", "c2": "not_relevant"},
        "source_decisions": {"s1": "relevant"}, "notes": "Reviewed against article."}, "accept", lookup)


def test_corpus_lookup_uses_offsets_and_nearby_without_text_copy(tmp_path):
    corpus, lookup, _ = fixture(tmp_path)
    assert lookup.count == 4 and lookup.has_chunk("c1") and lookup.has_source("s1")
    assert [row["chunk_id"] for row in lookup.nearby("c1")] == ["c1", "c2"]
    assert lookup.get_chunk("c2")["text"].startswith("Nguồn")
    assert not lookup.has_chunk("missing")
    lookup.close()
    import sqlite3
    with sqlite3.connect(tmp_path / "workspace/corpus_lookup.sqlite3") as connection:
        assert "text" not in [row[1] for row in connection.execute("PRAGMA table_info(chunks)")]


def test_candidate_generation_is_pending_and_balanced_on_tiny_corpus(tmp_path):
    corpus, lookup, _ = fixture(tmp_path)
    config = {"seed": 5, "base_target": 3, "max_candidates_per_source": 1,
              "max_candidates_per_title": 1, "max_famous_topics": 1,
              "category_targets": {"battles": 1, "dynasties": 1, "treaties": 1}, "edge_targets": {}}
    rows = draft_candidates(corpus, config, limit=3)
    assert len(rows) == 3 and {row["category"] for row in rows} == set(config["category_targets"])
    assert all(row["review_status"] == "pending" and row["draft_question"] == row["question"] for row in rows)
    assert all("relevant_chunk_ids" not in row and "gold_answer" not in row for row in rows)
    assert len({row["origin_source_id"] for row in rows}) == 3
    lookup.close()


def test_expansion_uses_new_edge_prompts_and_keeps_drafts_pending(tmp_path):
    corpus, lookup, _ = fixture(tmp_path)
    config = {"seed": 5, "base_target": 2, "max_candidates_per_source": 1,
              "max_candidates_per_title": 1, "max_famous_topics": 1,
              "category_targets": {}, "edge_targets": {"out_of_domain": 2}}
    pilot = draft_candidates(corpus, config, limit=2)
    expanded = draft_candidates(corpus, config, limit=4, existing=pilot)
    assert scaled_targets(config, 4) == {"out_of_domain": 4}
    assert len(expanded) == 2
    assert len({row["question"] for row in pilot + expanded}) == 4
    assert all(row["review_status"] == "pending" for row in expanded)
    lookup.close()


def test_review_transitions_persistence_and_draft_separation(tmp_path):
    _, lookup, _ = fixture(tmp_path)
    workspace = Workspace(tmp_path / "workspace")
    workspace.insert_candidates([candidate()])
    with pytest.raises(ValueError, match="Retrieve candidate evidence"):
        workspace.apply_review("x1", {"answerable": True, "in_domain": True}, "accept", lookup)
    row = accepted(workspace, lookup)
    assert row["review_status"] == "accepted" and row["review_version"] == 1
    assert row["gold_answer"] is None and row["draft_answer"] == "Bản nháp cần kiểm tra."
    assert row["original_draft_answer"] == "Bản nháp cần kiểm tra."
    assert row["reviewed_at"]
    with pytest.raises(RuntimeError, match="another session"):
        workspace.apply_review("x1", {}, "save", lookup, expected_revision=0)
    workspace.close()
    workspace = Workspace(tmp_path / "workspace")
    assert workspace.get("x1")["review_status"] == "accepted"
    with pytest.raises(ValueError, match="rerun candidate evidence"):
        workspace.apply_review("x1", {"question": "Câu hỏi đã chỉnh sửa?"}, "accept", lookup)
    updated = workspace.apply_review("x1", {"question": "Câu hỏi đã chỉnh sửa?"}, "save", lookup)
    assert updated["review_status"] == "needs_review"
    assert not updated["evidence_complete"] and not updated["evidence"]
    assert not updated["chunk_decisions"] and not updated["source_decisions"]
    with pytest.raises(ValueError, match="Retrieve candidate evidence"):
        workspace.apply_review("x1", {}, "accept", lookup)
    workspace.set_evidence("x1", [{"stage": "dense", "rank": 1, "chunk_id": "c1",
        "source_id": "s1", "document_id": "d1", "score": .9, "details": {}}])
    reaccepted = workspace.apply_review("x1", {"chunk_decisions": {"c1": "relevant"},
        "source_decisions": {"s1": "relevant"}}, "accept", lookup)
    assert reaccepted["review_version"] == 2
    assert workspace.apply_review("x1", {}, "reject", lookup)["review_status"] == "rejected"
    assert workspace.apply_review("x1", {}, "skip", lookup)["review_status"] == "rejected"
    workspace.close(); lookup.close()


def test_evidence_mapping_and_resume_only_incomplete(tmp_path):
    _, lookup, rows = fixture(tmp_path)
    result = {"annotation_stages": {"dense": [{"row_id": 0, "rank": 1, "score": .8,
        "query_rank": 1}], "bm25": [{"row_id": 1, "rank": 1, "score": 2.0}],
        "rrf": [{"row_id": 0, "rank": 1, "score": .04}],
        "reranked": [{"row_id": 0, "rank": 1, "score": .9}]}}
    mapped = evidence_from_result(result, rows)
    assert {item["stage"] for item in mapped} == {"dense", "bm25", "rrf", "reranked"}
    assert mapped[0]["chunk_id"] == "c1" and mapped[1]["chunk_id"] == "c2"
    workspace = Workspace(tmp_path / "workspace")
    workspace.insert_candidates([candidate(), candidate("x2", "Nhà Trần là gì?", "dynasties", "c3", "s2", "Nhà Trần")])
    class FakeRetriever:
        calls = []
        def retrieve(self, question, *, include_stage_diagnostics):
            assert include_stage_diagnostics is True
            self.calls.append(question)
            if len(self.calls) == 2:
                raise RuntimeError("offline fixture failure")
            return result
    fake = FakeRetriever()
    assert hydrate_workspace(workspace, fake, rows) == 1
    assert hydrate_workspace(workspace, fake, rows) == 1
    assert len(fake.calls) == 3
    assert all(row["evidence_complete"] for row in workspace.all_candidates())
    workspace.close(); lookup.close()


def test_validator_export_and_schema_exclude_pending(tmp_path):
    corpus, lookup, _ = fixture(tmp_path)
    workspace = Workspace(tmp_path / "workspace")
    workspace.insert_candidates([candidate(), candidate("x2", "Nhà Trần là gì?", "dynasties", "c3", "s2", "Nhà Trần")])
    accepted(workspace, lookup)
    path = tmp_path / "retrieval.jsonl"
    assert export(workspace, lookup, path, tier="retrieval") == 1
    questions = load_questions(path)
    assert len(questions) == 1 and questions[0].id == "x1"
    assert questions[0].relevant_chunk_ids == ["c1"] and questions[0].relevant_source_ids == ["s1"]
    assert questions[0].gold_answer is None and questions[0].review_status == "accepted"
    with pytest.raises(FileExistsError):
        export(workspace, lookup, path, tier="retrieval")
    bad = workspace_records(workspace, "retrieval")
    bad[0]["relevant_chunk_ids"] = ["unknown"]
    assert any("unknown chunk" in item for item in validate_records(bad, lookup)["errors"])
    bad[0]["relevant_chunk_ids"] = ["c1"]
    bad[0]["relevant_source_ids"] = ["s2"]
    assert any("source is absent" in item for item in validate_records(bad, lookup)["errors"])
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{bad json}\n' + path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    raw, errors = read_records(invalid)
    assert len(raw) == 2 and "invalid JSON" in errors[0]
    assert any("duplicate ID" in item for item in validate_records(raw, lookup)["errors"])
    workspace.close(); lookup.close()


def test_deep_queue_selection_acceptance_and_export(tmp_path):
    _, lookup, _ = fixture(tmp_path)
    workspace = Workspace(tmp_path / "workspace")
    workspace.insert_candidates([candidate(), candidate("x2", "Nhà Trần là gì?", "dynasties", "c3", "s2", "Nhà Trần", "hard")])
    accepted(workspace, lookup)
    workspace.set_evidence("x2", [{"stage": "reranked", "rank": 1, "chunk_id": "c3",
        "source_id": "s2", "document_id": "d2", "score": .7, "details": {}}])
    workspace.apply_review("x2", {"answerable": True, "in_domain": True,
        "chunk_decisions": {"c3": "relevant"}, "source_decisions": {"s2": "relevant"}}, "accept", lookup)
    rows = [workspace.get(item["candidate_id"]) for item in workspace.all_candidates()]
    assert retrieval_behavior(rows[0]) in ("final_miss", "dense_only_hit", "shared_hit")
    assert {item["candidate_id"] for item in choose(rows, 2)} == {"x1", "x2"}
    with pytest.raises(ValueError, match="Only 2 accepted"):
        select(workspace, tmp_path / "too_large.jsonl", limit=3)
    queue = tmp_path / "deep_queue.jsonl"
    assert select(workspace, queue, limit=2) == 2
    assert all(row["deep_status"] == "pending" for row in workspace.all_candidates())
    with pytest.raises(ValueError, match="gold answer"):
        workspace.apply_review("x1", {}, "accept_deep", lookup)
    reviewed = workspace.apply_review("x1", {"gold_answer": "Chiến thắng có ý nghĩa lịch sử.",
        "required_facts": ["ý nghĩa lịch sử"], "gold_citation_source_ids": ["s1"],
        "factual_paragraph_indices": [0]}, "accept_deep", lookup)
    assert reviewed["answer_reviewed"] and reviewed["citation_reviewed"]
    path = tmp_path / "deep.jsonl"
    assert export(workspace, lookup, path, tier="deep") == 1
    deep = load_questions(path)[0]
    assert deep.gold_answer and deep.required_facts == ["ý nghĩa lịch sử"]
    assert deep.gold_citation_source_ids == ["s1"]
    workspace.close(); lookup.close()


def test_duplicates_report_backup_and_local_ui_mapping(tmp_path):
    corpus, lookup, _ = fixture(tmp_path)
    workspace = Workspace(tmp_path / "workspace")
    workspace.insert_candidates([candidate(), candidate("x2", "Trận Bạch Đằng có ý nghĩa gì?", "battles", "c3", "s2", "Nhà Trần")])
    exact, _ = duplicate_pairs(workspace.all_candidates())
    assert exact
    assert coverage(workspace.all_candidates())["review_status"]["pending"] == 2
    workspace.set_evidence("x1", [{"stage": "dense", "rank": 1, "chunk_id": "c1",
        "source_id": "s1", "document_id": "d1", "score": .9, "details": {"query": "Bạch Đằng"}}])
    backup = tmp_path / "reports/annotation/backup.jsonl"
    workspace.backup(backup)
    assert backup.is_file() and not backup.with_name("backup.jsonl.partial").exists()
    restored = Workspace(tmp_path / "restored")
    restored.restore(backup)
    assert restored.count() == 2 and restored.evidence("x1")[0]["stage"] == "dense"
    restored.close(); workspace.close(); lookup.close()
    app = create_app(tmp_path / "workspace", corpus)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Accept retrieval" in page.text
        item = client.get("/api/candidates/x1").json()
        assert item["evidence"][0]["text"].startswith("Trận Bạch Đằng")
        assert [row["chunk_id"] for row in client.get("/api/chunks/c1/nearby").json()] == ["c1", "c2"]
        response = client.post("/api/candidates/x1/review", json={"action": "accept",
            "patch": {"answerable": True, "in_domain": True,
                      "chunk_decisions": {"c1": "relevant"}, "source_decisions": {"s1": "relevant"}},
            "expected_revision": item["revision"]})
        assert response.status_code == 200 and response.json()["review_status"] == "accepted"
