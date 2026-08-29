from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from app.agents.evidence_agent import EvidenceCriticAgent
from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.research_agent import ResearchAgent
from app.telemetry import current_request_telemetry, log_event


logger = logging.getLogger(__name__)
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
        "agentic": mode == "agentic_rag",
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
            mode="hybrid_rag",
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
            "mode:hybrid_rag",
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
            inference_mode="hybrid_rag",
        )
        result["inference_mode"] = "hybrid_rag"
        result["agentic"] = False
        result["retrieval"] = retrieval
        result["retrieval_latency_sec"] = retrieval_elapsed_ms / 1000
        result["total_latency_sec"] = time.perf_counter() - started
        provenance = result.setdefault("answer_provenance", {})
        provenance.update({
            "mode": "hybrid_rag",
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_input_evidence_count": len(result.get("history_debug", {}).get("input_evidence_ids", [])),
            "history_input_claim_count": result.get("history_debug", {}).get("input_claim_count", 0),
            "history_input_source_kind_counts": result.get("history_debug", {}).get("input_source_kind_counts", {}),
            "total_llm_calls": int(provenance.get("history_generation_calls") or 0),
        })
        return result


class AgentOrchestrator:
    def __init__(
        self,
        *,
        research_agent: ResearchAgent,
        evidence_agent: EvidenceCriticAgent,
        answerer: HistoryAnswererAgent,
    ):
        self.research_agent = research_agent
        self.evidence_agent = evidence_agent
        self.answerer = answerer
        self.max_history_messages = research_agent.retrieval_runtime.max_history_messages
        self.retrieval_history_messages = (
            research_agent.retrieval_runtime.retrieval_history_messages
        )

    async def run(
        self,
        *,
        question: str,
        final_k: int,
        history: list[dict[str, str]] | None = None,
        owner_id: str | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        session_id = request_id or f"{conversation_id or 'anonymous'}:{uuid.uuid4()}"
        gate = _domain_gate(self.research_agent.retrieval_runtime.retriever, question)
        scoped = _scoped_response(
            question=question,
            gate=gate,
            mode="agentic_rag",
            started=started,
            answer_depth="deep",
        )
        if scoped is not None:
            scoped["research_debug"] = {
                "steps": 0,
                "generation_calls": 0,
                "attempts": [],
                "tools": [],
                "evidence_ids": [],
                "retrieval_question": question,
                "prefetch_used": False,
                "external_fallback_triggered": False,
            }
            scoped["evidence_critique"] = {
                "status": "insufficient",
                "selected_evidence": [],
                "selected_ids": [],
                "sufficient": False,
            }
            scoped["evidence_debug"] = {
                "input_count": 0,
                "input_ids": [],
                "model_input_evidence": [],
                "status": "insufficient",
                "selected_ids": [],
                "generation_calls": 0,
                "repair_used": False,
                "repair_path": None,
            }
            return scoped
        try:
            research = await self.research_agent.run(
                question,
                final_k=final_k,
                history=history,
                session_id=session_id,
                owner_id=owner_id,
                conversation_id=conversation_id,
                request_id=request_id,
            )
            research_attempts = [research.debug]
            critique, contexts = self.evidence_agent.compress(
                question,
                research.evidence,
                final_k=final_k,
                request_id=session_id,
            )
            tool_trace = (
                ["agent:research"]
                + research.tool_trace
                + ["agent:evidence_critic", f"evidence_selected:{len(contexts)}"]
            )
            if not critique.sufficient and self.research_agent.model_runtime is not None:
                log_event(
                    "ORCHESTRATOR_RETRY",
                    request_id=request_id,
                    reason="evidence_insufficient",
                    missing_information_count=len(critique.missing_information),
                )
                follow_up = " ".join(critique.missing_information) or question
                research = await self.research_agent.run(
                    follow_up,
                    final_k=final_k,
                    history=history,
                    session_id=session_id,
                    owner_id=owner_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
                research_attempts.append(research.debug)
                critique, contexts = self.evidence_agent.compress(
                    question,
                    research.evidence,
                    final_k=final_k,
                    request_id=session_id,
                )
                tool_trace.extend(
                    ["agent:research_retry", *research.tool_trace, "agent:evidence_critic_retry"]
                )
            result = self.answerer.answer(
                question=question,
                contexts=contexts,
                analysis=research.analysis,
                tool_trace=tool_trace,
                is_ood=research.is_ood and not contexts,
                ood_reason=research.ood_reason,
                history=history,
                request_id=request_id,
                answer_depth="deep",
                inference_mode="agentic_rag",
            )
        finally:
            self.research_agent.evidence_store.remove_session(session_id)
        result["agentic"] = True
        result["inference_mode"] = "agentic_rag"
        result["evidence_critique"] = critique.model_dump()
        telemetry = current_request_telemetry()
        result["research_debug"] = {
            "steps": sum(int(item.get("steps", 0)) for item in research_attempts),
            "generation_calls": sum(int(item.get("generation_calls", 0)) for item in research_attempts),
            "attempts": research_attempts,
            "tools": [
                tool
                for attempt in research_attempts
                for tool in attempt.get("tools", [])
            ],
            "evidence_ids": list(dict.fromkeys(
                evidence_id
                for attempt in research_attempts
                for evidence_id in attempt.get("evidence_ids", [])
            )),
            "retrieval_question": research_attempts[0].get("retrieval_question"),
            "prefetch_used": telemetry.research_prefetch_used if telemetry is not None else any(
                tool.get("deterministic_prefetch")
                for attempt in research_attempts
                for tool in attempt.get("tools", [])
            ),
            "external_fallback_triggered": telemetry.external_fallback_triggered if telemetry is not None else any(
                tool.get("external_fallback")
                for attempt in research_attempts
                for tool in attempt.get("tools", [])
            ),
        }
        result["evidence_debug"] = {
            "input_count": len(research.evidence),
            "input_ids": [str(item.get("chunk_id")) for item in research.evidence],
            "model_input_evidence": critique.model_input_evidence,
            "model_output": {
                "status": critique.status,
                "selected_evidence": [
                    item.model_dump() for item in critique.selected_evidence
                ],
                "conflicts": critique.conflicts,
                "missing_information": critique.missing_information,
                "summary": critique.summary,
            },
            "status": critique.status,
            "selected_ids": critique.selected_ids,
            "generation_calls": critique.generation_calls,
            "repair_used": critique.repair_used,
            "repair_path": critique.repair_path,
            "candidate_count": len(critique.model_input_evidence),
            "selected_count": len(critique.selected_evidence),
            "raw_candidate_count": critique.raw_candidate_count,
            "model_visible_candidate_count": critique.model_visible_candidate_count,
            "dropped_for_budget_count": critique.dropped_for_budget_count,
            "dropped_ids": critique.dropped_ids,
            "dropped_reasons": critique.dropped_reasons,
            "source_kind_counts_raw": critique.source_kind_counts_raw,
            "source_kind_counts_visible": critique.source_kind_counts_visible,
            "question_type": critique.question_type,
            "first_model_output": critique.first_model_output,
            "first_validation_issues": critique.first_validation_issues,
            "final_validation_issues": critique.final_validation_issues,
            "semantic_guard_findings": critique.semantic_guard_findings,
            "comparison_targets": critique.comparison_targets,
            "target_a_candidate_count": critique.target_a_candidate_count,
            "target_b_candidate_count": critique.target_b_candidate_count,
            "target_a_model_visible_count": critique.target_a_model_visible_count,
            "target_b_model_visible_count": critique.target_b_model_visible_count,
            "comparison_target_coverage": critique.comparison_target_coverage,
            "comparison_target_map": critique.comparison_target_map,
            "target_a_selected_evidence": critique.target_a_selected_evidence,
            "target_b_selected_evidence": critique.target_b_selected_evidence,
            "shared_selected_evidence": critique.shared_selected_evidence,
            "unknown_selected_evidence": critique.unknown_selected_evidence,
            "evidence_pruned_claim_count": critique.evidence_pruned_claim_count,
            "evidence_supplemented_count": critique.evidence_supplemented_count,
            "evidence_supplemented_ids": critique.evidence_supplemented_ids,
            "rebucket_attempted": (
                telemetry.evidence_rebucket_attempted if telemetry is not None else False
            ),
            "rebucket_succeeded": (
                telemetry.evidence_rebucket_succeeded if telemetry is not None else False
            ),
            "rebucket_moved_claim_count": (
                telemetry.evidence_rebucket_moved_claim_count if telemetry is not None else 0
            ),
            "rebucket_destination_ids": (
                telemetry.evidence_rebucket_destination_ids if telemetry is not None else []
            ),
            "final_validation_result": (
                telemetry.evidence_final_validation_result if telemetry is not None else (
                    "pass" if not critique.final_validation_issues else "fail"
                )
            ),
            "relevance_guard_triggered": (
                telemetry.evidence_relevance_guard_triggered if telemetry is not None else False
            ),
            "coverage_guard_triggered": (
                telemetry.evidence_coverage_guard_triggered if telemetry is not None else False
            ),
            "reconsideration_used": (
                telemetry.evidence_reconsideration_used if telemetry is not None else False
            ),
            "missing_information": critique.missing_information,
            "summary": critique.summary,
        }
        provenance = result.setdefault("answer_provenance", {})
        research_generation_calls = (
            telemetry.research_llm_calls if telemetry is not None else result["research_debug"]["generation_calls"]
        )
        evidence_generation_calls = (
            telemetry.evidence_generation_calls if telemetry is not None else critique.generation_calls
        )
        history_generation_calls = int(provenance.get("history_generation_calls") or 0)
        total_llm_calls = (
            telemetry.total_llm_calls
            if telemetry is not None
            else research_generation_calls + evidence_generation_calls + history_generation_calls
        )
        provenance.update({
            "mode": "agentic_rag",
            "evidence_status": critique.status,
            "selected_evidence_ids": critique.selected_ids,
            "research_steps": result["research_debug"]["steps"],
            "research_generation_calls": research_generation_calls,
            "research_json_repairs": sum(int(item.get("json_repairs", 0)) for item in research_attempts),
            "research_prefetch_used": telemetry.research_prefetch_used if telemetry is not None else result["research_debug"]["prefetch_used"],
            "external_fallback_triggered": telemetry.external_fallback_triggered if telemetry is not None else result["research_debug"]["external_fallback_triggered"],
            "wikipedia_search_count": telemetry.wikipedia_search_count if telemetry is not None else 0,
            "wikipedia_fetch_count": telemetry.wikipedia_fetch_count if telemetry is not None else 0,
            "wikipedia_query": telemetry.wikipedia_query if telemetry is not None else None,
            "wikipedia_candidate_titles": telemetry.wikipedia_candidate_titles if telemetry is not None else [],
            "wikipedia_selected_title": telemetry.wikipedia_selected_title if telemetry is not None else None,
            "wikipedia_year_conflict_rejections": telemetry.wikipedia_year_conflict_rejections if telemetry is not None else 0,
            "duplicate_inspect_skipped": telemetry.duplicate_inspect_skipped if telemetry is not None else False,
            "evidence_generation_calls": evidence_generation_calls,
            "evidence_repair_used": critique.repair_used,
            "evidence_candidate_count": len(critique.model_input_evidence),
            "evidence_candidate_count_raw": critique.raw_candidate_count,
            "evidence_candidate_count_model_visible": critique.model_visible_candidate_count,
            "evidence_dropped_for_budget_count": critique.dropped_for_budget_count,
            "evidence_dropped_ids": critique.dropped_ids,
            "evidence_source_kind_counts_raw": critique.source_kind_counts_raw,
            "evidence_source_kind_counts_visible": critique.source_kind_counts_visible,
            "evidence_question_type": critique.question_type,
            "evidence_first_validation_issues": critique.first_validation_issues,
            "evidence_final_validation_issues": critique.final_validation_issues,
            "comparison_targets": critique.comparison_targets,
            "comparison_target_coverage": critique.comparison_target_coverage,
            "comparison_target_map": critique.comparison_target_map,
            "target_a_selected_evidence": critique.target_a_selected_evidence,
            "target_b_selected_evidence": critique.target_b_selected_evidence,
            "shared_selected_evidence": critique.shared_selected_evidence,
            "unknown_selected_evidence": critique.unknown_selected_evidence,
            "external_evidence_collected_count": telemetry.external_evidence_collected_count if telemetry is not None else 0,
            "external_evidence_model_visible_count": telemetry.external_evidence_model_visible_count if telemetry is not None else 0,
            "external_evidence_selected_count": telemetry.external_evidence_selected_count if telemetry is not None else 0,
            "external_evidence_rejected_count": telemetry.external_evidence_rejected_count if telemetry is not None else 0,
            "external_evidence_rejection_reasons": telemetry.external_evidence_rejection_reasons if telemetry is not None else {},
            "evidence_selected_count": len(critique.selected_evidence),
            "evidence_pruned_claim_count": critique.evidence_pruned_claim_count,
            "evidence_supplemented_count": critique.evidence_supplemented_count,
            "evidence_supplemented_ids": critique.evidence_supplemented_ids,
            "evidence_relevance_guard_triggered": telemetry.evidence_relevance_guard_triggered if telemetry is not None else False,
            "evidence_coverage_guard_triggered": telemetry.evidence_coverage_guard_triggered if telemetry is not None else False,
            "evidence_reconsideration_used": telemetry.evidence_reconsideration_used if telemetry is not None else False,
            "evidence_rebucket_attempted": telemetry.evidence_rebucket_attempted if telemetry is not None else False,
            "evidence_rebucket_succeeded": telemetry.evidence_rebucket_succeeded if telemetry is not None else False,
            "evidence_rebucket_moved_claim_count": telemetry.evidence_rebucket_moved_claim_count if telemetry is not None else 0,
            "evidence_rebucket_destination_ids": telemetry.evidence_rebucket_destination_ids if telemetry is not None else [],
            "evidence_final_validation_result": telemetry.evidence_final_validation_result if telemetry is not None else "pass",
            "history_generation_calls": history_generation_calls,
            "history_input_evidence_count": len(result.get("history_debug", {}).get("input_evidence_ids", [])),
            "history_input_claim_count": result.get("history_debug", {}).get("input_claim_count", 0),
            "history_input_source_kind_counts": result.get("history_debug", {}).get("input_source_kind_counts", {}),
            "total_llm_calls": total_llm_calls,
        })
        result["total_latency_sec"] = time.perf_counter() - started
        logger.info(
            "agent_run_complete",
            extra={
                "request_id": session_id,
                "conversation_id": conversation_id,
                "agent_step": len(research.tool_trace),
                "latency_ms": result["total_latency_sec"] * 1000,
                "evidence_count": len(contexts),
                "answer_provenance": provenance.get("source"),
            },
        )
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
        selected_final_k = max(1, int(final_k or 6))
        return asyncio.run(
            self.run(
                question=question,
                final_k=selected_final_k,
                history=history,
                owner_id=owner_id,
                conversation_id=conversation_id,
                request_id=request_id,
            )
        )
