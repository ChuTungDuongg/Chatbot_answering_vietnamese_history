from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
        result = self.retriever.retrieve(arguments.query, final_k=arguments.top_k)
        chunks = result.get("final_context") or []
        for chunk in chunks:
            chunk.setdefault("source_kind", "history")
        return chunks

