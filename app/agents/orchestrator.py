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
            )
        finally:
            self.research_agent.evidence_store.remove_session(session_id)
        result["agentic"] = True
        result["inference_mode"] = "agentic_rag"
        result["evidence_critique"] = critique.model_dump()
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
            "missing_information": critique.missing_information,
            "summary": critique.summary,
        }
        provenance = result.setdefault("answer_provenance", {})
        telemetry = current_request_telemetry()
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
            "evidence_generation_calls": evidence_generation_calls,
            "evidence_repair_used": critique.repair_used,
            "history_generation_calls": history_generation_calls,
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
