from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from app.agents.history_contract import (
    SAFE_INSUFFICIENT_ANSWER,
    SAFE_OOD_ANSWER,
    build_history_answerer_messages,
    parse_history_answer_output,
)
from app.agents.model_runtime import RoleLLMBackend

if TYPE_CHECKING:
    from app.rag.generation import RAGGenerator


class HistoryAnswererAgent:
    """Active Qwen3 History role, matching the canonical History SFT contract."""

    def __init__(self, *, model_runtime: RoleLLMBackend):
        if model_runtime is None:
            raise ValueError("Active HistoryAnswererAgent requires a role model runtime.")
        self.model_runtime = model_runtime

    @staticmethod
    def _retrieval_payload(
        *,
        question: str,
        contexts: list[dict[str, Any]],
        analysis: dict[str, Any],
        tool_trace: list[str],
        is_ood: bool,
        ood_reason: str,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "final_context": contexts,
            "analysis": analysis,
            "tool_trace": tool_trace,
            "is_ood": is_ood,
            "ood_reason": ood_reason,
            "global_context_count": sum(
                item.get("source_kind") != "attachment" for item in contexts
            ),
            "temporary_context_count": sum(
                item.get("source_kind") == "attachment" for item in contexts
            ),
            "temporary_context_relevant": any(
                item.get("source_kind") == "attachment" for item in contexts
            ),
        }

    @staticmethod
    def _guard_result(
        *,
        question: str,
        retrieval: dict[str, Any],
        analysis: dict[str, Any],
        tool_trace: list[str],
        answer: str,
        status: str,
        guard_name: str,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": answer,
            "status": status,
            "source_ids": [],
            "source_chunks": [],
            "model_source_ids": [],
            "invalid_source_ids": [],
            "unsupported_years": [],
            "format_ok": True,
            "raw_output": "",
            "retrieval": retrieval,
            "analysis": analysis,
            "tool_trace": [*tool_trace, f"history_guard:{guard_name}"],
            "history_message_count": 0,
            "rewrite_used": False,
            "repair_attempted": False,
            "structured_expansion_used": False,
            "answer_provenance": {
                "source": "deterministic_guard",
                "history_adapter_called": False,
                "history_generation_calls": 0,
                "guard_short_circuit": True,
                "guard_name": guard_name,
                "guard_override": False,
            },
            "history_debug": {
                "generation_calls": 0,
                "input_evidence_ids": [],
                "cited_ids": [],
                "conversation_history_used": False,
            },
        }

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
        del history  # History SFT has no conversation-history input.
        started = time.perf_counter()
        question = str(question).strip()
        contexts = [
            dict(item)
            for item in contexts
            if str(item.get("chunk_id") or "").strip() and str(item.get("text") or "").strip()
        ]
        retrieval = self._retrieval_payload(
            question=question,
            contexts=contexts,
            analysis=analysis,
            tool_trace=tool_trace,
            is_ood=is_ood,
            ood_reason=ood_reason,
        )

        if is_ood and not contexts:
            return self._guard_result(
                question=question,
                retrieval=retrieval,
                analysis=analysis,
                tool_trace=tool_trace,
                answer=SAFE_OOD_ANSWER,
                status="blocked_off_topic",
                guard_name="off_topic_no_evidence",
            )
        if not contexts:
            return self._guard_result(
                question=question,
                retrieval=retrieval,
                analysis=analysis,
                tool_trace=tool_trace,
                answer=SAFE_INSUFFICIENT_ANSWER,
                status="blocked_no_context",
                guard_name="no_selected_evidence",
            )

        input_ids = [str(item["chunk_id"]) for item in contexts]
        messages = build_history_answerer_messages(question, contexts)
        raw_output = self.model_runtime.generate_text(
            adapter="history",
            messages=messages,
        )
        parsed = parse_history_answer_output(raw_output, allowed_source_ids=input_ids)
        by_id = {str(item["chunk_id"]): item for item in contexts}
        source_chunks = [by_id[source_id] for source_id in parsed.source_ids]
        status = "ok" if parsed.source_ids else "insufficient"

        return {
            "question": question,
            "answer": parsed.answer,
            "status": status,
            "source_ids": parsed.source_ids,
            "source_chunks": source_chunks,
            "model_source_ids": parsed.source_ids,
            "invalid_source_ids": [],
            "unsupported_years": [],
            "format_ok": True,
            "raw_output": parsed.raw_output,
            "retrieval": retrieval,
            "analysis": analysis,
            "prompt_budget": None,
            "support_score": None,
            "quality_warnings": [],
            "rewrite_used": False,
            "repair_attempted": False,
            "structured_expansion_used": False,
            "initial_quality_issues": [],
            "history_message_count": 0,
            "tool_trace": [*tool_trace, "history:adapter", "history:citation_validation"],
            "latency_sec": time.perf_counter() - started,
            "answer_provenance": {
                "source": "history_adapter",
                "history_adapter_called": True,
                "history_generation_calls": 1,
                "guard_short_circuit": False,
                "guard_name": None,
                "guard_override": False,
            },
            "history_debug": {
                "generation_calls": 1,
                "input_evidence_ids": input_ids,
                "cited_ids": parsed.source_ids,
                "conversation_history_used": False,
            },
        }


class LegacyRAGHistoryAnswerer:
    """Compatibility wrapper for the benchmark-only merged/static-RAG backend."""

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
        retrieval = HistoryAnswererAgent._retrieval_payload(
            question=question,
            contexts=contexts,
            analysis=analysis,
            tool_trace=tool_trace,
            is_ood=is_ood,
            ood_reason=ood_reason,
        )
        result = self.generator.answer_from_retrieval(
            question=question,
            retrieval=retrieval,
            history=history,
        )
        result["tool_trace"] = tool_trace + result.get("tool_trace", [])
        blocked = str(result.get("status") or "").startswith("blocked_")
        result["answer_provenance"] = {
            "source": "legacy_static_rag",
            "history_adapter_called": False,
            "history_generation_calls": None,
            "guard_short_circuit": blocked,
            "guard_name": result.get("status") if blocked else None,
            "guard_override": blocked,
        }
        return result
