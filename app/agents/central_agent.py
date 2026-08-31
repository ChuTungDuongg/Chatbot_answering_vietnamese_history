from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from app.agents.config import AgentConfig


INSUFFICIENT_EVIDENCE_ANSWER = (
    "Mình chưa tìm thấy đủ bằng chứng đáng tin cậy để trả lời câu hỏi này. "
    "Bạn có thể bổ sung tài liệu hoặc làm rõ giai đoạn, nhân vật hay sự kiện cần hỏi."
)


class CentralAgent:
    """App-layer selector over the direct and bounded three-role runtimes.

    This class owns no model and duplicates no retrieval logic. Simple factual
    requests stay on the direct path; deeper requests reuse the existing
    Research/Evidence/History orchestration engine.
    """

    _DEEP_REQUEST_CUES = (
        "so sánh", "đối chiếu", "nguyên nhân", "vì sao", "tại sao", "ý nghĩa",
        "tác động", "hệ quả", "đánh giá", "kiểm chứng", "xác minh", "bằng chứng",
        "nguồn", "tài liệu", "file", "pdf", "wikipedia", "trang web", "mới nhất",
    )

    def __init__(
        self,
        orchestrator: Any,
        *,
        fast_service: Any,
        config: AgentConfig | None = None,
    ):
        self.orchestrator = orchestrator
        self.fast_service = fast_service
        self.config = config or AgentConfig()
        research_agent = getattr(orchestrator, "research_agent", None)
        configured_steps = getattr(research_agent, "max_steps", self.config.max_steps)
        if int(configured_steps) > self.config.max_steps:
            raise ValueError("Central Agent cannot expose more steps than AgentConfig.max_steps.")
        self.max_history_messages = getattr(orchestrator, "max_history_messages", 6)
        self.retrieval_history_messages = getattr(orchestrator, "retrieval_history_messages", 4)

    @classmethod
    def requires_deep_orchestration(cls, question: str) -> bool:
        normalized = re.sub(r"\s+", " ", question.casefold()).strip()
        return (
            len(normalized) > 180
            or normalized.count("?") > 1
            or any(cue in normalized for cue in cls._DEEP_REQUEST_CUES)
        )

    @staticmethod
    def _fallback(question: str, *, reason: str) -> dict[str, Any]:
        return {
            "question": question,
            "answer": INSUFFICIENT_EVIDENCE_ANSWER,
            "status": "insufficient_evidence",
            "source_ids": [],
            "source_chunks": [],
            "retrieval": {"question": question, "final_context": [], "tool_trace": []},
            "analysis": {"question": question},
            "tool_trace": [],
            "agentic": True,
            "inference_mode": "agent",
            "answer_provenance": {
                "mode": "agent",
                "source": "central_agent_fallback",
                "fallback_reason": reason,
            },
        }

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        started = time.perf_counter()
        question = str(kwargs.get("question") or "")
        selected_path = "agent_tools" if self.requires_deep_orchestration(question) else "fast_direct"
        if selected_path == "fast_direct":
            result = await asyncio.to_thread(self.fast_service.chat, **kwargs)
        else:
            try:
                result = await asyncio.wait_for(
                    self.orchestrator.run(**kwargs),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                result = self._fallback(question, reason="timeout")
        if not str(result.get("answer") or "").strip():
            result = self._fallback(question, reason="no_useful_evidence")
        result["central_agent"] = {
            "max_steps": self.config.max_steps,
            "timeout_seconds": self.config.timeout_seconds,
            "selected_path": selected_path,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }
        return result

    def chat(
        self,
        question: str,
        final_k: int | None = None,
        history: list[dict[str, str]] | None = None,
        owner_id: str | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(self.run(
            question=question,
            final_k=max(1, int(final_k or 6)),
            history=history,
            owner_id=owner_id,
            conversation_id=conversation_id,
            request_id=request_id,
        ))
