"""Compatibility facade for the canonical LangGraph-backed Hybrid workflow."""
from __future__ import annotations

import time
from typing import Any

from app.agents.hybrid_graph import HybridGraphDependencies, build_hybrid_graph


class HybridRAGOrchestrator:
    def __init__(self, *, retriever: Any, retrieval_runtime: Any, answerer: Any):
        self.retriever = retriever
        self.retrieval_runtime = retrieval_runtime
        self.answerer = answerer
        self.max_history_messages = retrieval_runtime.max_history_messages
        self.retrieval_history_messages = retrieval_runtime.retrieval_history_messages
        self.graph = build_hybrid_graph(HybridGraphDependencies(
            retriever=retriever,
            retrieval_runtime=retrieval_runtime,
            answerer=answerer,
        ))

    def chat(
        self,
        question: str,
        final_k: int | None = None,
        history: list[dict[str, str]] | None = None,
        owner_id: str | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        output = self.graph.invoke({
            "request_id": request_id,
            "conversation_id": conversation_id,
            "mode": "hybrid",
            "question": question,
            "final_k": max(1, int(final_k or getattr(self.retriever, "final_context_k", 6))),
            "history": list(history or []),
            "owner_id": owner_id,
            "started": time.perf_counter(),
            "graph_trace": [],
            "graph_route": [],
        })
        return output["result"]


__all__ = ["HybridRAGOrchestrator", "HybridGraphDependencies", "build_hybrid_graph"]
