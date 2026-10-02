"""Small execution events; repeated query stages report aggregate active time."""

from contextlib import contextmanager
import time
from typing import Any, Callable

ProgressCallback = Callable[[dict[str, Any]], None]

LABELS = {
    "query_analysis": "Phân tích câu hỏi",
    "embedding": "Tạo embedding truy vấn",
    "bm25_search": "Truy vấn BM25",
    "fusion": "Hợp nhất kết quả",
    "rerank": "Rerank tư liệu",
    "context_selection": "Chuẩn bị nguồn",
    "prompt_preparation": "Chuẩn bị ngữ cảnh",
    "tool_selection": "Chọn công cụ",
    "attachment_search": "Tìm trong tài liệu",
    "tool:search_history": "Tìm tư liệu lịch sử",
    "tool:search_wikipedia": "Tìm Wikipedia",
    "tool:fetch_wikipedia_page": "Đọc Wikipedia",
    "tool:search_uploaded_documents": "Tìm trong tài liệu",
    "tool:search_web": "Tìm trên web",
    "tool:fetch_page": "Đọc trang web",
}


class StageProgress:
    def __init__(self, callback: ProgressCallback | None = None, backend: str | None = None):
        self.callback, self.backend = callback, backend
        self.timings: dict[str, float] = {}
        self.started: set[str] = set()

    def emit(self, stage: str, state: str):
        if self.callback:
            label = (f"Truy vấn {str(self.backend).upper() if self.backend == 'faiss' else 'Qdrant'}"
                     if stage == "dense_search" else LABELS.get(stage, "Sử dụng công cụ"))
            self.callback({"stage": stage, "state": state, "message": label,
                           "retrieval_backend": self.backend,
                           **({"latency_ms": self.timings[stage]} if state == "completed" else {})})

    @contextmanager
    def track(self, stage: str, *, finish: bool = True, accumulate: bool = False):
        if stage not in self.started:
            self.started.add(stage)
            if not accumulate:
                self.timings[stage] = 0.0
            else:
                self.timings.setdefault(stage, 0.0)
            self.emit(stage, "started")
        start = time.perf_counter_ns()
        try:
            yield
        except BaseException:
            self.emit(stage, "failed")
            raise
        else:
            self.timings[stage] = self.timings.get(stage, 0.0) + (time.perf_counter_ns() - start) / 1e6
            if finish:
                self.emit(stage, "completed")
                self.started.discard(stage)
