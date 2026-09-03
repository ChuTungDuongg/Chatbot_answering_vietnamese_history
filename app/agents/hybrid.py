from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any
from app.agents.history_answerer import HistoryAnswererAgent
from app.telemetry import current_request_telemetry, log_event
from app.agents.common.domain_gate import _domain_gate, _scoped_response

logger = logging.getLogger(__name__)


class HybridRAGOrchestrator:
    def __init__(
        self,
        *,
        retriever: Any,
        retrieval_runtime: Any,
        answerer: HistoryAnswererAgent,
    ):
        self.retriever = retriever
        self.retrieval_runtime = retrieval_runtime
        self.answerer = answerer
        self.max_history_messages = retrieval_runtime.max_history_messages
        self.retrieval_history_messages = retrieval_runtime.retrieval_history_messages

    def chat(
        self,
        question: str,
        final_k: int | None = None,
        history: list[dict[str, str]] | None = None,
        owner_id: str | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del owner_id, conversation_id
        started = time.perf_counter()
        gate = _domain_gate(self.retriever, question)
        scoped = _scoped_response(
            question=question,
            gate=gate,
            mode="hybrid",
            started=started,
            answer_depth="standard",
        )
        if scoped is not None:
            return scoped
        selected_final_k = max(1, int(final_k or getattr(self.retriever, "final_context_k", 6)))
        normalized_history = self.retrieval_runtime.normalize_history(
            history,
            current_question=question,
        )
        retrieval_question, history_used = self.retrieval_runtime.build_retrieval_question(
            question,
            normalized_history,
        )
        retrieval_started = time.perf_counter()
        retrieval = self.retriever.retrieve(retrieval_question, final_k=selected_final_k)
        retrieval_elapsed_ms = (time.perf_counter() - retrieval_started) * 1000
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.retrieval_ms += retrieval_elapsed_ms
        contexts = list(retrieval.get("final_context", []))
        analysis = self.retriever.analyze_question(question)
        retrieval.update({
            "question": question,
            "analysis": analysis,
            "retrieval_question": retrieval_question,
            "history_used_for_retrieval": history_used,
            "global_final_context": contexts,
            "global_context_count": sum(item.get("source_kind") != "attachment" for item in contexts),
            "temporary_context_count": sum(item.get("source_kind") == "attachment" for item in contexts),
            "temporary_context_relevant": False,
            "context_title_diversity": self.retriever.context_title_diversity(contexts),
            "domain_gate_result": retrieval.get("domain_gate_result") or gate.get("domain_gate_result"),
            "domain_gate_reason": retrieval.get("domain_gate_reason") or gate.get("domain_gate_reason"),
        })
        tool_trace = [
            *retrieval.get("tool_trace", []),
            "mode:hybrid",
            "hybrid:retriever",
        ]
        result = self.answerer.answer(
            question=question,
            contexts=contexts if not retrieval.get("is_ood") else [],
            analysis=analysis,
            tool_trace=tool_trace,
            is_ood=bool(retrieval.get("is_ood")),
            ood_reason=str(retrieval.get("ood_reason") or ""),
            history=None,
            request_id=request_id,
            answer_depth="standard",
            avoid_generic_source_prefix=True,
            inference_mode="hybrid",
        )
        result["inference_mode"] = "hybrid"
        result["agentic"] = False
        result["retrieval"] = retrieval
        result["retrieval_latency_sec"] = retrieval_elapsed_ms / 1000
        result["total_latency_sec"] = time.perf_counter() - started
        provenance = result.setdefault("answer_provenance", {})
        provenance.update({
            "mode": "hybrid",
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_input_evidence_count": len(result.get("history_debug", {}).get("input_evidence_ids", [])),
            "history_input_claim_count": result.get("history_debug", {}).get("input_claim_count", 0),
            "history_input_source_kind_counts": result.get("history_debug", {}).get("input_source_kind_counts", {}),
            "total_llm_calls": int(provenance.get("history_generation_calls") or 0),
        })
        result["performance_debug"] = {
            "retrieval_latency_ms": retrieval_elapsed_ms,
            "history_first_latency_ms": result.get("history_debug", {}).get("first_latency_ms"),
            "history_retry_latency_ms": result.get("history_debug", {}).get("retry_latency_ms"),
            "history_total_latency_ms": result.get("history_debug", {}).get("total_latency_ms"),
            "total_latency_ms": result["total_latency_sec"] * 1000,
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_generation_calls": provenance.get("history_generation_calls", 0),
            "total_llm_calls": provenance.get("total_llm_calls", 0),
        }
        return result
