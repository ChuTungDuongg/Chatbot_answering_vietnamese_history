"""Project explicitly reviewed workspace rows into the existing Question contract."""

from __future__ import annotations

from typing import Any

from evaluation.schema import Question


def gold_record(candidate: dict[str, Any], tier: str) -> dict[str, Any]:
    if tier not in ("retrieval", "deep"):
        raise ValueError("Tier must be retrieval or deep")
    if candidate.get("review_status") != "accepted" or not candidate.get("retrieval_reviewed"):
        raise ValueError("Only explicitly accepted retrieval reviews may be exported")
    if candidate.get("annotation_origin") == "automatic":
        raise ValueError("Automatic annotations cannot enter human gold export")
    if tier == "deep" and (candidate.get("deep_status") != "accepted" or
                           not candidate.get("answer_reviewed") or
                           not candidate.get("citation_reviewed")):
        raise ValueError("Deep export requires explicit answer and citation acceptance")
    chunk_decisions = candidate.get("chunk_decisions") or {}
    source_decisions = candidate.get("source_decisions") or {}
    record = {"schema_version": 1, "id": candidate["candidate_id"],
              "question": candidate["question"], "category": candidate["category"],
              "difficulty": candidate.get("difficulty"),
              "question_type": candidate.get("question_type"),
              "benchmark_split": candidate.get("benchmark_split"),
              "relevant_chunk_ids": sorted(k for k, v in chunk_decisions.items() if v == "relevant"),
              "relevant_source_ids": sorted(k for k, v in source_decisions.items() if v == "relevant"),
              "answerable": candidate.get("answerable"), "in_domain": candidate.get("in_domain"),
              "notes": candidate.get("notes") or "", "review_status": "accepted",
              "retrieval_reviewed": True, "reviewed_at": candidate.get("reviewed_at"),
              "review_version": candidate.get("review_version"),
              "review_policy_version": candidate.get("review_policy_version"),
              "reviewer_id": candidate.get("reviewer_id"),
              "annotation_origin": "human", "annotation_status": "human_accepted"}
    if tier == "deep":
        record.update(gold_answer=candidate.get("gold_answer"),
                      required_facts=candidate.get("required_facts"),
                      gold_citation_source_ids=candidate.get("gold_citation_source_ids"),
                      factual_paragraph_indices=candidate.get("factual_paragraph_indices"),
                      answer_reviewed=True, citation_reviewed=True,
                      deep_reviewed_at=candidate.get("deep_reviewed_at"))
    return Question.model_validate(record).model_dump(mode="json", exclude_none=True)
