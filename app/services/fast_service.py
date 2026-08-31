from __future__ import annotations

from typing import Any


class FastChatService:
    """Low-latency facade over the existing direct Hybrid RAG path."""

    def __init__(self, hybrid_service: Any, *, max_contexts: int = 3):
        if max_contexts < 1:
            raise ValueError("Fast mode max_contexts must be positive.")
        self.hybrid_service = hybrid_service
        self.max_contexts = max_contexts
        self.max_history_messages = getattr(hybrid_service, "max_history_messages", 6)
        self.retrieval_history_messages = getattr(hybrid_service, "retrieval_history_messages", 4)

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        requested = kwargs.get("final_k")
        kwargs["final_k"] = min(max(1, int(requested or self.max_contexts)), self.max_contexts)
        result = self.hybrid_service.chat(**kwargs)
        result["fast_path"] = True
        return result
