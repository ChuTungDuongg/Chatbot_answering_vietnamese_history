from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.rag.retrieval import (
    balance_comparison_candidates,
    build_comparison_target_queries,
)


class SearchHistoryInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)


class SearchHistoryTool:
    name = "search_history"
    description = "Search the local Vietnamese-history corpus and return ranked chunks."
    input_schema = SearchHistoryInput

    def __init__(self, retriever: Any):
        self.retriever = retriever

    def run(self, arguments: SearchHistoryInput) -> list[dict[str, Any]]:
        analysis_fn = getattr(self.retriever, "analyze_question", None)
        analysis = analysis_fn(arguments.query) if callable(analysis_fn) else {}
        comparison_plan = build_comparison_target_queries(arguments.query, analysis)
        if comparison_plan:
            merged: dict[str, dict[str, Any]] = {}
            for role in ("target_a", "target_b"):
                result = self.retriever.retrieve(
                    comparison_plan[f"{role}_query"],
                    final_k=max(arguments.top_k, 6),
                )
                for chunk in result.get("final_context") or []:
                    chunk_id = str(chunk.get("chunk_id") or "")
                    if not chunk_id:
                        continue
                    existing = merged.setdefault(chunk_id, dict(chunk))
                    existing["retrieval_query_roles"] = list(dict.fromkeys([
                        *existing.get("retrieval_query_roles", []),
                        role,
                    ]))
            chunks, balance = balance_comparison_candidates(
                arguments.query,
                list(merged.values()),
                arguments.top_k,
            )
            for chunk in chunks:
                chunk.setdefault("source_kind", "history")
                chunk["target_specific_queries"] = comparison_plan
                chunk["comparison_balance"] = balance
            return chunks

        result = self.retriever.retrieve(arguments.query, final_k=arguments.top_k)
        chunks = result.get("final_context") or []
        for chunk in chunks:
            chunk.setdefault("source_kind", "history")
        return chunks
