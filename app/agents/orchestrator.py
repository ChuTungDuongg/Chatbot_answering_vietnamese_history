from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from app.agents.evidence_agent import EvidenceCriticAgent
from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.research_agent import ResearchAgent


logger = logging.getLogger(__name__)


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
    ) -> dict[str, Any]:
        started = time.perf_counter()
        session_id = f"{conversation_id or 'anonymous'}:{uuid.uuid4()}"
        try:
            research = await self.research_agent.run(
                question,
                final_k=final_k,
                history=history,
                session_id=session_id,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            research_attempts = [research.debug]
            critique, contexts = self.evidence_agent.compress(question, research.evidence, final_k=final_k)
            tool_trace = (
                ["agent:research"]
                + research.tool_trace
                + ["agent:evidence_critic", f"evidence_selected:{len(contexts)}"]
            )
            if not critique.sufficient and self.research_agent.model_runtime is not None:
                follow_up = " ".join(critique.missing_information) or question
                research = await self.research_agent.run(
                    follow_up,
                    final_k=final_k,
                    history=history,
                    session_id=session_id,
                    owner_id=owner_id,
                    conversation_id=conversation_id,
                )
                research_attempts.append(research.debug)
                critique, contexts = self.evidence_agent.compress(
                    question,
                    research.evidence,
                    final_k=final_k,
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
            )
        finally:
            self.research_agent.evidence_store.remove_session(session_id)
        result["agentic"] = True
        result["evidence_critique"] = critique.model_dump()
        result["research_debug"] = {
            "steps": sum(int(item.get("steps", 0)) for item in research_attempts),
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
            "missing_information": critique.missing_information,
            "summary": critique.summary,
        }
        provenance = result.setdefault("answer_provenance", {})
        provenance.update({
            "evidence_status": critique.status,
            "selected_evidence_ids": critique.selected_ids,
            "research_steps": result["research_debug"]["steps"],
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
    ) -> dict[str, Any]:
        selected_final_k = max(1, int(final_k or 6))
        return asyncio.run(
            self.run(
                question=question,
                final_k=selected_final_k,
                history=history,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
        )
