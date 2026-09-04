"""Compatibility facade for the canonical LangGraph-backed three-role workflow."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from app.agents.three_llm.graph import ThreeLLMGraphDependencies, build_three_llm_graph


logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(self, *, research_agent: Any, evidence_agent: Any, answerer: Any):
        self.research_agent = research_agent
        self.evidence_agent = evidence_agent
        self.answerer = answerer
        self.max_history_messages = research_agent.retrieval_runtime.max_history_messages
        self.retrieval_history_messages = research_agent.retrieval_runtime.retrieval_history_messages
        self.graph = build_three_llm_graph(ThreeLLMGraphDependencies(
            research_agent=research_agent,
            evidence_agent=evidence_agent,
            answerer=answerer,
        ))

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
        session_id = request_id or f"{conversation_id or 'anonymous'}:{uuid.uuid4()}"
        started = time.perf_counter()
        try:
            output = await self.graph.ainvoke({
                "request_id": request_id,
                "conversation_id": conversation_id,
                "mode": "three_llm",
                "question": question,
                "final_k": max(1, int(final_k or 6)),
                "history": list(history or []),
                "owner_id": owner_id,
                "started": started,
                "session_id": session_id,
                "graph_trace": [],
                "graph_route": [],
            })
            result = output["result"]
            logger.info(
                "agent_run_complete",
                extra={
                    "request_id": session_id,
                    "conversation_id": conversation_id,
                    "agent_step": len(getattr(output.get("research"), "tool_trace", [])),
                    "latency_ms": result.get("total_latency_sec", 0.0) * 1000,
                    "evidence_count": len(result.get("source_chunks", [])),
                    "answer_provenance": result.get("answer_provenance", {}).get("source"),
                },
            )
            return result
        finally:
            self.research_agent.evidence_store.remove_session(session_id)

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


__all__ = ["AgentOrchestrator", "ThreeLLMGraphDependencies", "build_three_llm_graph"]
