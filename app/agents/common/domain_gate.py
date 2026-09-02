from __future__ import annotations

import time
from typing import Any
from app.telemetry import current_request_telemetry


SCOPE_ANSWER = (
    "Xin lỗi, mình là trợ lý chuyên về lịch sử Việt Nam. Mình có thể giúp bạn "
    "với các câu hỏi về sự kiện, nhân vật, triều đại và các vấn đề lịch sử Việt Nam."
)
META_ANSWER = (
    "Mình có thể trả lời các câu hỏi về lịch sử Việt Nam, gồm sự kiện, nhân vật, "
    "triều đại, chiến tranh, văn hóa, kinh tế, tôn giáo và so sánh lịch sử."
)
AMBIGUOUS_ANSWER = (
    "Bạn có thể nói rõ câu hỏi này gắn với giai đoạn, sự kiện hoặc bối cảnh nào "
    "trong lịch sử Việt Nam không?"
)


def _domain_gate(retriever: Any, question: str) -> dict[str, Any]:
    classifier = getattr(retriever, "classify_question", None)
    if not callable(classifier):
        return {"domain_gate_result": "in_domain", "domain_gate_reason": "classifier_unavailable"}
    result = dict(classifier(question) or {})
    result.setdefault("domain_gate_result", "out_of_domain" if result.get("is_ood") else "in_domain")
    result.setdefault("domain_gate_reason", result.get("ood_reason") or "classifier")
    return result


def _scoped_response(
    *,
    question: str,
    gate: dict[str, Any],
    mode: str,
    started: float,
    answer_depth: str,
) -> dict[str, Any] | None:
    gate_result = str(gate.get("domain_gate_result") or "in_domain")
    if gate_result not in {"out_of_domain", "meta", "ambiguous"}:
        return None

    answer = {
        "out_of_domain": SCOPE_ANSWER,
        "meta": META_ANSWER,
        "ambiguous": AMBIGUOUS_ANSWER,
    }[gate_result]
    status = "blocked_off_topic" if gate_result == "out_of_domain" else gate_result
    retrieval = {
        "question": question,
        "is_ood": gate_result == "out_of_domain",
        "ood_reason": str(gate.get("ood_reason") or ""),
        "domain_gate_result": gate_result,
        "domain_gate_reason": str(gate.get("domain_gate_reason") or ""),
        "intent": gate.get("intent", {}),
        "analysis": {"question": question, "facet": gate_result, "facets": [gate_result]},
        "query_variants": [],
        "final_context": [],
        "tool_trace": [f"domain_gate:{gate_result}"],
    }
    telemetry = current_request_telemetry()
    if telemetry is not None:
        telemetry.domain_gate_result = gate_result
        telemetry.domain_gate_reason = str(gate.get("domain_gate_reason") or "")
        telemetry.history_anchor = gate.get("history_anchor")
        telemetry.ood_anchor = gate.get("ood_anchor")
        telemetry.domain_margin = gate.get("domain_margin")
        telemetry.retrieval_skipped_due_to_ood = gate_result == "out_of_domain"
        telemetry.llm_calls_skipped_due_to_ood = True
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
        "retrieval": retrieval,
        "analysis": retrieval["analysis"],
        "prompt_budget": None,
        "support_score": None,
        "quality_warnings": [],
        "rewrite_used": False,
        "repair_attempted": False,
        "structured_expansion_used": False,
        "initial_quality_issues": [],
        "raw_output": "",
        "history_message_count": 0,
        "tool_trace": retrieval["tool_trace"],
        "latency_sec": time.perf_counter() - started,
        "total_latency_sec": time.perf_counter() - started,
        "inference_mode": mode,
        "agentic": mode in {"three_llm", "central", "agentic_rag"},
        "answer_provenance": {
            "mode": mode,
            "source": "domain_gate",
            "guard_short_circuit": True,
            "guard_name": f"domain_gate:{gate_result}",
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_generation_calls": 0,
            "total_llm_calls": 0,
            "answer_depth": answer_depth,
        },
        "history_debug": {
            "generation_calls": 0,
            "input_evidence_ids": [],
            "input_claim_count": 0,
            "input_source_kind_counts": {},
            "input_evidence_preview": [],
            "answer_depth": answer_depth,
        },
    }
