"""Copy a selected SILVER audit queue into an isolated, pending human GOLD workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.annotation.silver_selection import audit_queue
from evaluation.annotation.silver_store import SilverStore
from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.workspace import Workspace


def prepare_audit(silver: SilverStore, human: Workspace, lookup: CorpusLookup, *, size: int, risky: int,
                  seed: int = 2026) -> list[str]:
    identity = silver.metadata("run_identity")
    if not identity:
        raise RuntimeError("SILVER run provenance is missing")
    human.set_metadata("silver_run_identity", identity)
    selected = audit_queue(silver.rows(), size, risky, seed)
    ids = [row["candidate_id"] for row in selected]
    human.set_metadata("audit_queue_ids", ids)
    for row in selected:
        generated, spec = row["generated"], row["seed"]
        draft = {"candidate_id": row["candidate_id"], "question": generated["question"],
                 "draft_question": generated["question"], "category": generated["category"],
                 "difficulty": generated["difficulty"],
                 "question_type": generated.get("question_type") or generated["category"],
                 "origin_chunk_id": spec.get("chunk_id"), "origin_source_id": spec.get("source_id"),
                 "origin_title": spec.get("title"), "draft_answer": generated.get("draft_answer"),
                 "candidate_required_facts": generated.get("candidate_required_facts") or [],
                 "generation_metadata": {"annotation_origin": "automatic", "audit_source": "silver",
                                         "silver_confidence": row["confidence"]},
                 "review_status": "pending"}
        inserted = human.insert_candidates([draft])
        if not inserted:
            # Never replace a human edit or restore evidence for an edited question.
            continue
        stages = (row.get("evidence") or {}).get("stages") or {}
        entries = []
        for stage in ("faiss", "qdrant", "bm25", "rrf", "reranked"):
            human_stage = "dense" if stage in {"faiss", "qdrant"} else stage
            for item in stages.get(stage, []):
                corpus_row = lookup.get_chunk(item["chunk_id"])
                if corpus_row is None:
                    raise RuntimeError("Audit evidence chunk is absent from corpus")
                entries.append({"stage": human_stage,
                                "rank": 1 + sum(value["stage"] == human_stage for value in entries),
                                "chunk_id": item["chunk_id"], "source_id": corpus_row.get("source_id"),
                                "document_id": corpus_row.get("document_id"), "score": item.get("score"),
                                "details": {"backend": stage, "original_rank": item.get("rank")}})
        for origin in (spec, spec.get("paired") or {}):
            chunk_id = origin.get("chunk_id")
            if chunk_id and all(item["chunk_id"] != chunk_id for item in entries):
                corpus_row = lookup.get_chunk(chunk_id)
                entries.append({"stage": "dense", "rank": 1 + sum(item["stage"] == "dense" for item in entries),
                                "chunk_id": chunk_id, "source_id": corpus_row.get("source_id"),
                                "document_id": corpus_row.get("document_id"),
                                "details": {"backend": "origin"}})
        human.set_evidence(row["candidate_id"], entries)
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace_silver_v1"))
    parser.add_argument("--audit-workspace", type=Path)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--risky", type=int, default=35)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("artifacts/corpus_v1/retrieval"))
    parser.add_argument("--refresh-evidence", action="store_true")
    args = parser.parse_args(argv)
    if args.size < 1 or not 0 <= args.risky <= args.size:
        parser.error("Invalid audit size or risky allocation")
    path = args.audit_workspace or args.workspace / "audit_gold"
    silver, human = SilverStore(args.workspace), Workspace(path)
    lookup = CorpusLookup(args.corpus, args.workspace / "corpus_lookup.sqlite3")
    try:
        if args.refresh_evidence:
            from evaluation.annotation.auto_annotate import load_runtime
            from evaluation.annotation.generate_candidates import hydrate_workspace
            from types import SimpleNamespace
            runtime_args = SimpleNamespace(corpus=args.corpus.resolve(),
                                           retrieval_root=args.retrieval_root.resolve(),
                                           dense_mode="faiss-only")
            service, retriever, _ = load_runtime(runtime_args)
            try:
                refreshed = hydrate_workspace(human, retriever, service.chunks)
            finally:
                service.shutdown()
            print(f"[silver-audit] refreshed evidence for {refreshed} edited questions")
        else:
            ids = prepare_audit(silver, human, lookup, size=args.size, risky=args.risky)
            print(f"[silver-audit] {len(ids)} pending human reviews at {path}")
    finally:
        silver.close()
        human.close()
        lookup.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
