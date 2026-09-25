"""Central's history tool delegates to the exact Hybrid retriever."""

from typing import Any

from pydantic import BaseModel, Field

from app.rag.retriever import Retriever


class SearchHistoryInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)


class SearchHistoryTool:
    name = "search_history"
    description = "Search the local Vietnamese history corpus for grounded source chunks."
    input_schema = SearchHistoryInput

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def run(self, arguments: SearchHistoryInput) -> list[dict[str, Any]]:
        result = self.retriever.retrieve(arguments.query, arguments.top_k)
        return [{**chunk, "source_kind": "history"}
                for chunk in result.get("final_context") or []]
