"""Tiny, offline checks for the isolated automatic benchmark workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.annotation import auto_annotate
from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.export import export_silver, silver_record
from evaluation.annotation.records import gold_record
from evaluation.annotation.silver_logic import (adjudicate_retrieval, candidate_evidence,
                                                adjudicate_deep,
                                                check_deep_answer, check_deep_verification,
                                                score_confidence, semantic_call)
from evaluation.annotation.silver_report import aggregate
from evaluation.annotation.silver_audit import prepare_audit
from evaluation.annotation.silver_selection import (audit_queue, balanced_subset,
                                                    duplicate_reason, next_category, select_seed)
from evaluation.annotation.silver_store import SilverStore
from evaluation.annotation.silver_validate import validate
from evaluation.annotation.workspace import Workspace
from evaluation.annotation.ui import create_app


@pytest.fixture
def tiny(tmp_path: Path):
    corpus = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "c1", "source_id": "s1", "document_id": "d1", "chunk_index": 0,
         "title": "Bạch Đằng", "text": "Ngô Quyền chỉ huy trận Bạch Đằng năm 938."},
        {"chunk_id": "c2", "source_id": "s2", "document_id": "d2", "chunk_index": 0,
         "title": "Nam Hán", "text": "Quân Nam Hán bị đánh bại trên sông Bạch Đằng."},
        {"chunk_id": "c3", "source_id": "s3", "document_id": "d3", "chunk_index": 0,
         "title": "Nhà Lý", "text": "Nhà Lý dời đô về Thăng Long."},
    ]
    corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    lookup = CorpusLookup.build(corpus, tmp_path / "corpus_lookup.sqlite3")
    store = SilverStore(tmp_path / "silver")
    yield store, lookup, rows
    store.close()
    lookup.close()


def seed(store: SilverStore, seq: int = 1, category: str = "battles") -> dict:
    item = store.insert_seed(seq, {"chunk_id": "c1", "source_id": "s1", "title": "Bạch Đằng",
                                   "category": category, "difficulty": "easy"})
    store.save(item["candidate_id"], generated={"question": "Ai lãnh đạo trận Bạch Đằng năm 938?",
                                                 "category": category, "difficulty": "easy",
                                                 "question_type": category})
    return store.get(item["candidate_id"])


class Semantic:
    def __init__(self, *, disagree: bool = False, contradict: bool = False):
        self.calls = []
        self.disagree, self.contradict = disagree, contradict

    def complete(self, stage, payload):
        self.calls.append(stage)
        if stage == "relevance":
            return {"decisions": [{"chunk_id": card["chunk_id"], "decision": "relevant",
                                   "confidence": "high", "reason": "Text directly answers the question"}
                                  for card in payload["evidence"]],
                    "answerable": True, "in_domain": True}
        if stage == "verify":
            return {"answerable": not self.disagree, "in_domain": True,
                    "sufficient": not self.disagree,
                    "supported_chunk_ids": [card["chunk_id"] for card in payload["evidence"]],
                    "reason": "Independent evidence check"}
        if stage == "contradiction":
            return {"contradiction_detected": self.contradict,
                    "contradicting_chunk_ids": ["c1"] if self.contradict else [],
                    "contradiction_notes": "Conflict" if self.contradict else ""}
        raise AssertionError(stage)


CONFIG = {"evidence_text_chars": 400, "semantic_batch_size": 6}


def test_two_pass_labels_and_backend_disagreement(tiny):
    store, lookup, _ = tiny
    row = seed(store)
    evidence = candidate_evidence({"faiss": [{"chunk_id": "c1", "score": .9}],
                                   "qdrant": [{"chunk_id": "c2", "score": .8}],
                                   "bm25": [{"chunk_id": "c1"}],
                                   "reranked": [{"chunk_id": "c1"}]}, "c1", limit=4)
    provider = Semantic()
    labels = adjudicate_retrieval(store, provider, row["candidate_id"], row["generated"],
                                  evidence, lookup, CONFIG)
    assert provider.calls == ["relevance", "verify", "contradiction"]
    assert labels["diagnostics"]["backend_disagreement"] is True
    assert labels["relevant_chunk_ids"] == ["c1", "c2"]
    assert labels["relevant_source_ids"] == ["s1", "s2"]
    assert labels["annotation_origin"] == "automatic"
    assert store.call_count() == 3


def test_semantic_disagreement_and_contradiction_require_human(tiny):
    store, lookup, _ = tiny
    row = seed(store)
    generated = {**row["generated"], "difficulty": "hard"}
    evidence = [{"chunk_id": "c1", "stages": ["origin", "faiss"], "scores": {"faiss": .9}}]
    labels = adjudicate_retrieval(store, Semantic(disagree=True, contradict=True),
                                  row["candidate_id"], generated, evidence, lookup, CONFIG)
    assert labels["confidence"]["level"] == "low"
    assert labels["annotation_status"] == "needs_human_review"
    assert labels["semantic_pass_agreement"] is False
    assert labels["contradiction"]["contradiction_detected"] is True


def test_empty_evidence_still_has_two_semantic_passes(tiny):
    store, lookup, _ = tiny
    row = seed(store)
    provider = Semantic()
    adjudicate_retrieval(store, provider, row["candidate_id"], row["generated"], [], lookup, CONFIG)
    assert provider.calls == ["relevance", "verify"]


def test_cache_resume_and_provenance_mismatch(tiny):
    store, _, _ = tiny
    row = seed(store)
    provider = SimpleNamespace(complete=lambda stage, payload: {"value": 1})
    result = semantic_call(store, provider, row["candidate_id"], "example", {"question": "a"},
                           lambda value: None)
    assert result == {"value": 1}
    assert semantic_call(store, SimpleNamespace(complete=lambda *a: pytest.fail("called")),
                         row["candidate_id"], "example", {"question": "a"}, lambda value: None) == result
    with pytest.raises(RuntimeError, match="input changed"):
        semantic_call(store, provider, row["candidate_id"], "example", {"question": "b"}, lambda value: None)
    store.set_metadata("run_identity", {"corpus": "one"})
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        store.set_metadata("run_identity", {"corpus": "two"})
    reopened = SilverStore(store.directory)
    assert reopened.get(row["candidate_id"])["generated"] == row["generated"]
    reopened.close()


def test_silver_gold_isolation_and_export(tiny, tmp_path):
    store, lookup, _ = tiny
    row = seed(store)
    labels = {"annotation_origin": "automatic", "annotation_status": "auto_reviewed",
              "answerable": True, "in_domain": True, "semantic_pass_agreement": True,
              "contradiction": {"contradiction_detected": False},
              "relevant_chunk_ids": ["c1"], "relevant_source_ids": ["s1"]}
    store.save(row["candidate_id"], labels=labels, annotation_status="auto_reviewed", confidence="high")
    output = tmp_path / "silver.jsonl"
    assert export_silver(store, lookup, output, tier="silver-retrieval") == 1
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["annotation_origin"] == "automatic"
    assert "review_status" not in exported
    with pytest.raises(ValueError, match="Automatic"):
        gold_record({"annotation_origin": "automatic", "review_status": "accepted",
                     "retrieval_reviewed": True}, "retrieval")
    low = seed(store, 2)
    store.save(low["candidate_id"], labels=labels, annotation_status="needs_human_review", confidence="low")
    assert len((tmp_path / "silver.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ValueError):
        silver_record(store.get(low["candidate_id"]), "silver-retrieval")


def test_duplicate_replacement_and_target_planning():
    config = {"base_target": 2, "category_targets": {"battles": 2}, "edge_targets": {},
              "difficulty_targets": {"easy": 1, "medium": 1, "hard": 0},
              "max_candidates_per_source": 1, "max_candidates_per_title": 1, "max_famous_topics": 0}
    assert duplicate_reason("Ai lãnh đạo trận Bạch Đằng năm 938?", [
        {"generated": {"question": "Ai lãnh đạo trận Bạch Đằng năm 938?"}}])
    assert next_category([], config, 2) == "battles"
    pools = {"battles": [{"chunk_id": "c1", "source_id": "s1", "title": "Bạch Đằng"},
                         {"chunk_id": "c2", "source_id": "s2", "title": "Nam Hán"}], "_general": []}
    first = select_seed("battles", pools, [], config)
    assert first["chunk_id"] == "c1"
    second = select_seed("battles", pools, [{"seed": first}], config)
    assert second["chunk_id"] == "c2"


def test_deep_fact_citation_support_validation():
    answer = {"silver_answer": "Ngô Quyền chỉ huy trận Bạch Đằng.",
              "required_facts": [{"fact": "Ngô Quyền chỉ huy trận Bạch Đằng.",
                                  "supporting_chunk_ids": ["c1"], "supporting_source_ids": ["s1"],
                                  "confidence": "high"}],
              "candidate_citation_source_ids": ["s1"], "factual_paragraph_indices": [0]}
    check_deep_answer(answer, {"c1": "s1"})
    check_deep_verification({"answer_evidence_coverage": True, "atomic_facts": True,
                             "fact_support": [{"fact": answer["required_facts"][0]["fact"],
                                               "supported": True, "supporting_chunk_ids": ["c1"]}],
                             "citation_support": [{"source_id": "s1", "supported": True,
                                                   "supporting_fact_indices": [0]}]},
                            answer["required_facts"], ["s1"], {"c1": "s1"})
    with pytest.raises(ValueError):
        check_deep_answer({**answer, "candidate_citation_source_ids": ["s9"]}, {"c1": "s1"})
    with pytest.raises(ValueError):
        check_deep_answer({**answer, "required_facts": [{**answer["required_facts"][0],
                                                        "supporting_source_ids": ["s2"]}]}, {"c1": "s1"})


def test_balanced_deep_and_random_safe_audit():
    rows = []
    for index in range(12):
        rows.append({"candidate_id": f"id{index}", "annotation_status": "auto_reviewed"
                     if index < 10 else "needs_human_review", "confidence": "high" if index < 8 else "low",
                     "generated": {"question": f"Câu hỏi lịch sử {index}?", "category": "battles"
                                   if index < 6 else "comparison", "difficulty": "easy"
                                   if index < 6 else "hard", "question_type": "single"
                                   if index < 6 else "multi"},
                     "labels": {"answerable": True, "in_domain": True,
                                "relevant_chunk_ids": ["c1"], "relevant_source_ids": ["s1"],
                                "diagnostics": {"retrieval_miss": index % 3 == 0,
                                                "backend_disagreement": index % 2 == 0,
                                                "faiss_qdrant_overlap": .5},
                                "contradiction": {"contradiction_detected": False}}})
    chosen = balanced_subset(rows[:8], 4)
    assert len(chosen) == 4
    assert len({row["generated"]["category"] for row in chosen}) == 2
    queue = audit_queue(rows, 6, 4, 2026)
    assert len(queue) == 6
    assert any(row["confidence"] == "high" for row in queue[4:])
    report = aggregate(rows, target=10, deep_target=4)
    assert report["auto_reviewed"] == 8
    assert report["backend_disagreements"] == 6


def test_call_estimate_and_dry_run_no_model(tmp_path, capsys):
    config = {"max_evidence_chunks": 12, "semantic_batch_size": 6}
    assert auto_annotate.call_estimate(500, 150, 1500, config)["total_nominal_upper"] == 2800
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert auto_annotate.main(["--config", str(path), "--workspace", str(tmp_path / "silver")]) == 0
    assert not (tmp_path / "silver").exists()
    assert "dry run complete" in capsys.readouterr().out


def test_target_means_usable_and_replaces_duplicate(tiny, monkeypatch):
    store, lookup, _ = tiny
    config = {"base_target": 2, "category_targets": {"battles": 2}, "edge_targets": {},
              "difficulty_targets": {"easy": 1, "medium": 1, "hard": 0}, "seed": 1,
              "max_candidates_per_source": 2, "max_candidates_per_title": 2,
              "max_famous_topics": 0}
    pool = {"battles": [{"chunk_id": f"c{i}", "source_id": f"s{i}",
                         "title": f"Chủ đề {i}", "excerpt": "Dữ liệu lịch sử"} for i in range(1, 4)],
            "_general": []}
    questions = iter(["Ai chỉ huy trận đánh lịch sử đầu tiên?",
                      "Ai chỉ huy trận đánh lịch sử đầu tiên?",
                      "Triều đại nào dời đô về Thăng Long?"])
    provider = SimpleNamespace(complete=lambda stage, payload: {
        "question": next(questions), "difficulty": payload["difficulty"],
        "candidate_required_facts": [], "question_type": "single"})
    def fake_retrieval(row, silver, *_):
        silver.save(row["candidate_id"], evidence={"stages": {}, "inspection": []},
                    labels={"annotation_origin": "automatic", "annotation_status": "auto_reviewed",
                            "answerable": True, "in_domain": True, "relevant_chunk_ids": ["c1"],
                            "relevant_source_ids": ["s1"], "diagnostics": {"retrieval_miss": False,
                            "backend_disagreement": False, "faiss_qdrant_overlap": None},
                            "contradiction": {"contradiction_detected": False}},
                    annotation_status="auto_reviewed", confidence="high")
    monkeypatch.setattr(auto_annotate, "process_retrieval", fake_retrieval)
    args = SimpleNamespace(target=2, deep_target=0, max_attempts=3, max_questions=None,
                           concurrency=1, start_after=0, phase="retrieval")
    auto_annotate.run(args, config, provider, None, None, None, lookup, store, pool)
    assert store.count() == 3
    assert [r["annotation_status"] for r in store.rows()] == ["auto_reviewed", "rejected", "auto_reviewed"]
    assert len([r for r in store.rows() if r["annotation_status"] == "auto_reviewed"]) == 2
    # Resume cannot add another 500; the absolute usable target is already met.
    auto_annotate.run(args, config, provider, None, None, None, lookup, store, pool)
    assert store.count() == 3


def test_audit_queue_is_pending_until_explicit_human_acceptance(tiny, tmp_path):
    store, lookup, _ = tiny
    row = seed(store)
    store.set_metadata("run_identity", {"corpus_sha256": lookup.corpus_sha256})
    labels = {"annotation_origin": "automatic", "annotation_status": "auto_reviewed",
              "confidence": {"level": "high"}, "answerable": True, "in_domain": True,
              "relevant_chunk_ids": ["c1"], "relevant_source_ids": ["s1"],
              "diagnostics": {"backend_disagreement": False, "retrieval_miss": False},
              "contradiction": {"contradiction_detected": False}}
    store.save(row["candidate_id"], labels=labels, annotation_status="auto_reviewed", confidence="high",
               evidence={"stages": {"faiss": [{"chunk_id": "c1", "rank": 1, "score": .9}]}})
    human = Workspace(tmp_path / "audit")
    try:
        ids = prepare_audit(store, human, lookup, size=1, risky=1)
        assert ids == [row["candidate_id"]]
        pending = human.get(ids[0])
        assert pending["review_status"] == "pending"
        assert pending["retrieval_reviewed"] is False
        assert pending["chunk_decisions"] == {}
        assert pending["evidence"][0]["source_id"] == "s1"
        from fastapi.testclient import TestClient
        with TestClient(create_app(human.directory, lookup.corpus,
                                   lookup_workspace=lookup.database.parent)) as client:
            response = client.get(f"/api/candidates/{ids[0]}")
            assert response.status_code == 200
            assert response.json()["evidence"][0]["text"]
        accepted = human.apply_review(ids[0], {"answerable": True, "in_domain": True,
                                               "chunk_decisions": {"c1": "relevant"},
                                               "source_decisions": {"s1": "relevant"}}, "accept", lookup)
        assert accepted["review_status"] == "accepted"
        gold = gold_record(accepted, "retrieval")
        assert gold["annotation_origin"] == "human"
        assert gold["annotation_status"] == "human_accepted"
    finally:
        human.close()


def test_confidence_is_heuristic_and_hard_stops_on_conflict():
    diagnostics = {"retrieval_miss": False, "backend_disagreement": False,
                   "bm25_support": True, "reranker_support": True}
    high = score_confidence(category="battles", difficulty="easy", answerable=True,
                            in_domain=True, relevant={"c1"}, sources={"s1"},
                            pass_agreement=True, sufficient=True, contradiction=False,
                            diagnostics=diagnostics, uncertain=False)
    assert high["level"] == "high"
    assert high["calibrated_probability"] is False
    low = score_confidence(category="battles", difficulty="easy", answerable=True,
                           in_domain=True, relevant={"c1"}, sources={"s1"},
                           pass_agreement=True, sufficient=True, contradiction=True,
                           diagnostics=diagnostics, uncertain=False)
    assert low["level"] == "low"
    assert "unresolved_contradiction" in low["penalties"]


def test_deep_citation_failure_requires_human_review(tiny):
    store, lookup, _ = tiny
    row = seed(store)
    store.save(row["candidate_id"], labels={"answerable": True, "in_domain": True,
                                             "relevant_chunk_ids": ["c1"], "relevant_source_ids": ["s1"],
                                             "confidence": {"level": "high"}})
    row = store.get(row["candidate_id"])
    class DeepProvider:
        def complete(self, stage, payload):
            if stage == "deep_answer":
                return {"silver_answer": "Ngô Quyền chỉ huy Bạch Đằng năm 938.",
                        "required_facts": [{"fact": "Ngô Quyền chỉ huy Bạch Đằng năm 938.",
                                            "supporting_chunk_ids": ["c1"],
                                            "supporting_source_ids": ["s1"], "confidence": "high"}],
                        "candidate_citation_source_ids": ["s1"], "factual_paragraph_indices": [0]}
            return {"fact_support": [{"fact": "Ngô Quyền chỉ huy Bạch Đằng năm 938.",
                                      "supported": True, "supporting_chunk_ids": ["c1"]}],
                    "citation_support": [{"source_id": "s1", "supported": False,
                                          "supporting_fact_indices": []}],
                    "answer_evidence_coverage": True, "atomic_facts": True,
                    "reason": "Citation does not support the claim"}
    result = adjudicate_deep(store, DeepProvider(), row, lookup, CONFIG)
    assert result["all_supported"] is False
    assert result["confidence"] == "low"
    assert result["annotation_status"] == "needs_human_review"


def test_validator_detects_unknown_corpus_ids(tiny):
    store, lookup, _ = tiny
    row = seed(store)
    store.save(row["candidate_id"], labels={"annotation_origin": "automatic",
                                             "annotation_status": "auto_reviewed", "answerable": True,
                                             "in_domain": True, "semantic_pass_agreement": True,
                                             "contradiction": {"contradiction_detected": False},
                                             "relevant_chunk_ids": ["unknown"],
                                             "relevant_source_ids": ["s1"]},
               annotation_status="auto_reviewed", confidence="high")
    errors = validate(store.rows(), lookup)["errors"]
    assert any("unknown chunk" in error for error in errors)
