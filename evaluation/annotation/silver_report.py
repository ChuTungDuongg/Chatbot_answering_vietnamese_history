"""Summarize SILVER coverage, evidence support, and reviewed human-audit agreement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.annotation.policy import duplicate_pairs
from evaluation.annotation.silver_selection import usable
from evaluation.annotation.silver_store import SilverStore
from evaluation.annotation.workspace import Workspace


def _rate(matches: list[bool]) -> dict[str, Any]:
    return {"matching": sum(matches), "reviewed": len(matches),
            "agreement_rate": sum(matches) / len(matches) if matches else None}


def aggregate(rows: list[dict[str, Any]], *, target: int, deep_target: int,
              audit: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    reviewed = usable(rows)
    labeled = [row for row in rows if row.get("labels")]
    deep = [row for row in rows if row.get("deep_status") == "auto_reviewed"]
    def distribution(field: str) -> dict[str, int]:
        if field in {"category", "difficulty", "question_type"}:
            return dict(Counter((row.get("generated") or {}).get(field) for row in labeled))
        return dict(Counter((row.get("labels") or {}).get(field) for row in labeled))
    def average(values: list[int]) -> float | None:
        return sum(values) / len(values) if values else None
    exact, near = duplicate_pairs([{"candidate_id": row["candidate_id"],
                                    "question": row["generated"]["question"]}
                                   for row in rows if row.get("generated")])
    source_counts = Counter((row.get("seed") or {}).get("source_id") for row in reviewed
                            if (row.get("seed") or {}).get("source_id"))
    title_counts = Counter((row.get("seed") or {}).get("title") for row in reviewed
                           if (row.get("seed") or {}).get("title"))
    metrics = {"target_questions": target, "deep_target": deep_target,
               "raw_generated": sum(row.get("generated") is not None for row in rows),
               "auto_reviewed": len(reviewed), "usable_silver": len(reviewed),
               "high_confidence": sum(row.get("confidence") == "high" for row in labeled),
               "medium_confidence": sum(row.get("confidence") == "medium" for row in labeled),
               "low_confidence": sum(row.get("confidence") == "low" for row in labeled),
               "needs_human_review": sum(row["annotation_status"] == "needs_human_review" for row in rows),
               "rejected": sum(row["annotation_status"] == "rejected" for row in rows),
               "duplicate_rejections": sum(row.get("rejection_reason") == "duplicate_or_near_duplicate" for row in rows),
               "duplicate_warnings": {"exact": exact, "near": near},
               "top_origin_sources": source_counts.most_common(10),
               "top_origin_titles": title_counts.most_common(10),
               "category_distribution": distribution("category"),
               "difficulty_distribution": distribution("difficulty"),
               "question_type_distribution": distribution("question_type"),
               "answerable_distribution": distribution("answerable"),
               "in_domain_distribution": distribution("in_domain"),
               "faiss_qdrant_overlap_mean": average([row["labels"]["diagnostics"]["faiss_qdrant_overlap"]
                                                      for row in labeled if row["labels"]["diagnostics"]["faiss_qdrant_overlap"] is not None]),
               "backend_disagreements": sum(row["labels"]["diagnostics"]["backend_disagreement"] for row in labeled),
               "retrieval_misses": sum(row["labels"]["diagnostics"]["retrieval_miss"] for row in labeled),
               "avg_supporting_chunks": average([len(row["labels"]["relevant_chunk_ids"]) for row in labeled]),
               "avg_supporting_sources": average([len(row["labels"]["relevant_source_ids"]) for row in labeled]),
               "contradictions": sum(row["labels"]["contradiction"]["contradiction_detected"] for row in labeled),
               "deep_auto_reviewed": len(deep),
               "deep_needs_human_review": sum(row.get("deep_status") == "needs_human_review" for row in rows),
               "deep_answer_evidence_coverage": _rate([bool(row["deep"]["answer_evidence_coverage"])
                                                       for row in rows if row.get("deep")]),
               "atomic_fact_checks": _rate([bool(row["deep"].get("atomic_facts"))
                                             for row in rows if row.get("deep")]),
               "required_fact_support": _rate([bool(fact["supported"]) for row in rows if row.get("deep")
                                               for fact in row["deep"]["fact_support"]]),
               "citation_support": _rate([bool(citation["supported"]) for row in rows if row.get("deep")
                                          for citation in row["deep"]["citation_support"]]),
               "audit_queue_size": len(audit or [])}
    if audit:
        silver_by_id = {row["candidate_id"]: row for row in labeled}
        matched = [(silver_by_id[row["candidate_id"]], row) for row in audit
                   if row["review_status"] == "accepted" and row["candidate_id"] in silver_by_id]
        def comparison(pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
            return {"chunk_relevance": _rate([set(s["labels"]["relevant_chunk_ids"]) ==
                                               {key for key, value in h["chunk_decisions"].items() if value == "relevant"}
                                               for s, h in pairs]),
                    "source_relevance": _rate([set(s["labels"]["relevant_source_ids"]) ==
                                                {key for key, value in h["source_decisions"].items() if value == "relevant"}
                                                for s, h in pairs]),
                    "answerable": _rate([s["labels"]["answerable"] == h["answerable"] for s, h in pairs]),
                    "in_domain": _rate([s["labels"]["in_domain"] == h["in_domain"] for s, h in pairs])}
        metrics["human_audit_agreement"] = comparison(matched)
        metrics["human_audit_by_confidence"] = {level: comparison([(s, h) for s, h in matched
                                                                   if s["confidence"] == level])
                                                for level in ("high", "medium", "low")}
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace_silver_v1"))
    parser.add_argument("--audit-workspace", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/annotation/silver_v1_summary.json"))
    args = parser.parse_args(argv)
    silver = SilverStore(args.workspace)
    audit_path = args.audit_workspace or args.workspace / "audit_gold"
    audit = Workspace(audit_path) if (audit_path / "annotation.sqlite3").is_file() else None
    try:
        identity = silver.metadata("run_identity") or {}
        report = aggregate(silver.rows(), target=identity.get("target", 500),
                           deep_target=identity.get("deep_target", 150),
                           audit=audit.all_candidates() if audit else None)
        report["run_provenance"] = {**identity, "started_at": silver.metadata("started_at"),
                                    "git_commit": silver.metadata("git_commit")}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        silver.close()
        if audit:
            audit.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
