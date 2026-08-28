from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def candidate_summary(item: dict[str, Any], *, rank: int, selected_ids: set[str]) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": str(item.get("chunk_id") or ""),
        "title": item.get("title"),
        "text_preview": str(item.get("text") or "")[:220],
        "best_dense_score": item.get("best_dense_score"),
        "best_bm25_score": item.get("best_bm25_score"),
        "rrf_score": item.get("rrf_score"),
        "reranker_score": item.get("reranker_score"),
        "final_retrieval_score": item.get("final_retrieval_score"),
        "selected": str(item.get("chunk_id") or "") in selected_ids,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only diagnostics without loading Qwen or any LoRA."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--artifact-root", default="artifacts/vn_history_deployment")
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from app.config import settings

    settings.app_mode = "retrieval-only"
    settings.artifact_root = Path(args.artifact_root)
    settings.device = args.device

    from app.rag.retrieval import HybridRetriever
    from app.services.rag_service import RAGService

    service = RAGService()
    try:
        service.load()
        result = HybridRetriever(service).retrieve(args.question, final_k=max(1, args.final_k))
        selected_ids = {
            str(item.get("chunk_id") or "")
            for item in result.get("final_context", [])
        }
        report = {
            "question": args.question,
            "query_variants": result.get("query_variants", []),
            "is_ood": result.get("is_ood", False),
            "ood_reason": result.get("ood_reason", ""),
            "max_dense": result.get("max_dense"),
            "selected_ids": sorted(selected_ids),
            "candidates": [
                candidate_summary(item, rank=index + 1, selected_ids=selected_ids)
                for index, item in enumerate(
                    result.get("candidates20", [])[: max(1, args.candidate_k)]
                )
            ],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        service.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
