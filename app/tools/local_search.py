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
    candidate_pool: bool = Field(default=False, description="Return ranked candidates before final diversity selection for host-side analytical selection.")


class SearchHistoryTool:
    name = "search_history"
    description = "Search the local Vietnamese-history corpus and return ranked chunks."
    input_schema = SearchHistoryInput

    def __init__(self, retriever: Any):
        self.retriever = retriever
        self._entity_title_index = None
        self._indexed_chunks = None

    def resolve_entity_title(self, target: str, expected_type: str | None = None) -> str | None:
        # Use the already loaded corpus. Never initialize embeddings/models here.
        from app.agents.central.targets import EntityTitleIndex
        chunks = getattr(getattr(self.retriever, "service", None), "chunks", None)
        if chunks is None:
            return None
        if self._indexed_chunks is not chunks:
            titles = (str(row.get("title") or row.get("page_title") or row.get("source_title")
                          or (row.get("metadata") or {}).get("title") or "") for row in chunks)
            self._entity_title_index = EntityTitleIndex(titles)
            self._indexed_chunks = chunks
        return self._entity_title_index.resolve(target, expected_type)

    def can_overlap_model_load_and_retrieval(self) -> bool:
        # RAGService loads both models onto settings.device. Inspect the already
        # loaded objects without invoking loaders; unknown/same-CUDA stays serial.
        service = getattr(self.retriever, "service", None)
        devices = [str(getattr(getattr(service, name, None), "device", "unknown"))
                   for name in ("embedder", "reranker")]
        return all(device == "cpu" for device in devices)

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
        chunks = (result.get("candidates20") or result.get("final_context") or [])[:arguments.top_k] if arguments.candidate_pool else result.get("final_context") or []
        chunks = [dict(chunk) for chunk in chunks]
        for chunk in chunks:
            chunk.setdefault("source_kind", "history")
        return chunks
