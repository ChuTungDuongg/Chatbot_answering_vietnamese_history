"""Central's history tool delegates to the exact Hybrid retriever."""

from typing import Any

from pydantic import BaseModel, Field

from app.rag.retriever import Retriever
from app.rag.backends import DenseBackendError
from app.tools.registry import ToolExecutionContext


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

    def run_with_context(self, arguments: SearchHistoryInput, context: ToolExecutionContext):
        options = {**({"dense_backend": context.retrieval_backend} if context.retrieval_backend else {}),
                   **({"progress": context.progress} if context.progress else {})}
        try:
            result = self.retriever.retrieve(arguments.query, arguments.top_k, **options)
        except DenseBackendError:
            raise
        except Exception:
            raise DenseBackendError("Tìm tư liệu thất bại. Hãy thử lại hoặc chọn nguồn truy xuất khác.") from None
        context.retrieval_metrics["backend"] = result.get("retrieval_backend", context.retrieval_backend)
        timings = context.retrieval_metrics.setdefault("timings_ms", {})
        for name, value in result.get("timings_ms", {}).items():
            timings[name] = timings.get(name, 0.0) + value
        return [{**chunk, "source_kind": "history"} for chunk in result.get("final_context") or []]
