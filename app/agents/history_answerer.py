from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.rag.generation import RAGGenerator


class HistoryAnswererAgent:
    def __init__(self, generator: RAGGenerator):
        self.generator = generator

    def answer(
        self,
        *,
        question: str,
        contexts: list[dict[str, Any]],
        analysis: dict[str, Any],
        tool_trace: list[str],
        is_ood: bool = False,
        ood_reason: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        retrieval = {
            "question": question,
            "final_context": contexts,
            "analysis": analysis,
            "tool_trace": tool_trace,
            "is_ood": is_ood,
            "ood_reason": ood_reason,
            "global_context_count": len([c for c in contexts if c.get("source_kind") != "attachment"]),
            "temporary_context_count": len([c for c in contexts if c.get("source_kind") == "attachment"]),
            "temporary_context_relevant": any(c.get("source_kind") == "attachment" for c in contexts),
        }
        result = self.generator.answer_from_retrieval(question=question, retrieval=retrieval, history=history)
        result["tool_trace"] = tool_trace + result.get("tool_trace", [])
        return result
