from __future__ import annotations

import hashlib
import json
from typing import Any

from app.agents.model_runtime import RoleLLMBackend
from app.agents.policy_schema import (
    RESEARCH_AGENT_SYSTEM,
    FinishDecision,
    ResearchPolicyState,
    ToolDecision,
    serialize_policy_state,
    validate_runtime_decision,
)
from app.agents.schemas import ResearchResult
from app.rag.generation import RAGGenerator
from app.tools.evidence_tools import SessionEvidenceStore
from app.tools.registry import ToolExecutionContext, ToolRegistry


class ResearchAgent:
    """Bounded PLAN/ACTION/OBSERVATION agent with a deterministic fallback."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        evidence_store: SessionEvidenceStore,
        generator: RAGGenerator,
        model_runtime: RoleLLMBackend | None = None,
        max_steps: int = 6,
        max_web_searches: int = 3,
        max_page_fetches: int = 5,
    ):
        self.registry = registry
        self.evidence_store = evidence_store
        self.generator = generator
        self.model_runtime = model_runtime
        self.max_steps = max_steps
        self.max_web_searches = max_web_searches
        self.max_page_fetches = max_page_fetches

    async def run(
        self,
        question: str,
        *,
        final_k: int,
        history: list[dict[str, str]] | None = None,
        session_id: str = "default",
        owner_id: str | None = None,
        conversation_id: str | None = None,
    ) -> ResearchResult:
        normalized_history = self.generator.normalize_history(history, current_question=question)
        retrieval_question, history_used = self.generator.build_retrieval_question(question, normalized_history)
        tool_context = None
        if owner_id and conversation_id:
            tool_context = ToolExecutionContext(
                owner_id=owner_id,
                conversation_id=conversation_id,
                session_id=session_id,
            )
        if self.model_runtime is not None:
            tool_trace = await self._run_model_policy(
                question=question,
                retrieval_question=retrieval_question,
                session_id=session_id,
                tool_context=tool_context,
            )
        else:
            tool_trace = await self._run_fallback(
                question=question,
                retrieval_question=retrieval_question,
                session_id=session_id,
                final_k=final_k,
                tool_context=tool_context,
            )
        analysis = self.generator.retriever.analyze_question(question)
        classifier = getattr(self.generator.retriever, "classify_question", None)
        classification = classifier(question) if callable(classifier) else {}
        return ResearchResult(
            question=question,
            evidence=self.evidence_store.all(session_id),
            tool_trace=tool_trace,
            is_ood=bool(classification.get("is_ood")),
            ood_reason=str(classification.get("ood_reason") or ""),
            analysis={
                **analysis,
                "retrieval_question": retrieval_question,
                "history_used_for_retrieval": history_used,
                **({"intent": classification["intent"]} if classification.get("intent") else {}),
            },
        )

    async def _run_fallback(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        final_k: int,
        tool_context: ToolExecutionContext | None,
    ) -> list[str]:
        attachment_chunks, tool_trace = await self._collect_attachment_evidence(
            question=question,
            retrieval_question=retrieval_question,
            session_id=session_id,
            final_k=final_k,
            tool_context=tool_context,
        )
        chunks, record = await self.registry.call(
            "search_history",
            {"query": retrieval_question, "top_k": max(final_k, 6)},
        )
        tool_trace.append(
            f"{record.name}:{record.result_count}" if not record.error else f"{record.name}:error"
        )
        chunks = chunks or []
        self.evidence_store.add_documents(session_id, chunks)
        if not attachment_chunks and not chunks and self.max_steps > 1 and "search_web" in self.registry.names():
            web_rows, web_record = await self.registry.call("search_web", {"query": question, "top_k": 5})
            tool_trace.append(
                f"{web_record.name}:{web_record.result_count}" if not web_record.error else f"{web_record.name}:error"
            )
            self.evidence_store.add_documents(session_id, web_rows or [])
        return tool_trace

    async def _run_model_policy(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        tool_context: ToolExecutionContext | None,
    ) -> list[str]:
        trace: list[str] = []
        observations: list[dict[str, Any]] = []
        web_searches = 0
        page_fetches = 0
        tools = [
            tool
            for tool in self.registry.describe()
            if tool["name"] != "search_uploaded_documents" or tool_context is not None
        ]
        available_tools = {tool["name"] for tool in tools}
        for step in range(1, self.max_steps + 1):
            state = ResearchPolicyState.model_validate({
                "question": question,
                "retrieval_question": retrieval_question,
                "step": step,
                "limits": {
                    "max_steps": self.max_steps,
                    "web_searches_left": self.max_web_searches - web_searches,
                    "page_fetches_left": self.max_page_fetches - page_fetches,
                },
                "tools": tools,
                "observations": observations,
                "evidence_ids": list(dict.fromkeys(
                    evidence_id
                    for observation in observations
                    for evidence_id in observation.get("evidence_ids", [])
                    if evidence_id
                )),
            })
            raw_decision = self.model_runtime.generate_json(
                adapter="research",
                messages=[
                    {
                        "role": "system",
                        "content": RESEARCH_AGENT_SYSTEM,
                    },
                    {"role": "user", "content": serialize_policy_state(state)},
                ],
                max_new_tokens=384,
            )
            try:
                decision = validate_runtime_decision(raw_decision, tool_names=available_tools)
            except (TypeError, ValueError) as exc:
                observations.append({"tool": "policy", "error": f"invalid_decision: {exc}"})
                trace.append("agent:invalid_decision")
                continue
            if isinstance(decision, FinishDecision):
                trace.append(f"agent:finish:{step}")
                break
            assert isinstance(decision, ToolDecision)
            tool_name = decision.tool_name
            arguments = dict(decision.arguments)
            if tool_name == "search_web":
                if web_searches >= self.max_web_searches:
                    observations.append({"tool": tool_name, "error": "web_search_budget_exhausted"})
                    continue
                web_searches += 1
            if tool_name == "fetch_web_page":
                if page_fetches >= self.max_page_fetches:
                    observations.append({"tool": tool_name, "error": "page_fetch_budget_exhausted"})
                    continue
                page_fetches += 1
            if tool_name in {"retrieve_evidence", "inspect_evidence"}:
                arguments["session_id"] = session_id
            result, record = await self.registry.call(
                tool_name,
                arguments,
                context=tool_context if tool_name == "search_uploaded_documents" else None,
            )
            trace.append(f"{record.name}:{record.result_count}" if not record.error else f"{record.name}:error")
            if record.error:
                observations.append({"tool": tool_name, "error": record.error})
                continue
            rows = self._evidence_rows(tool_name, result)
            if (
                tool_name == "search_uploaded_documents"
                and not self.generator.temporary_context_is_relevant(
                    str(arguments.get("query") or question),
                    rows,
                )
            ):
                rows = []
                trace.append("attachment_relevant:false")
            self.evidence_store.add_documents(session_id, rows)
            observations.append(
                {
                    "tool": tool_name,
                    "result_count": len(rows),
                    "evidence_ids": [row.get("chunk_id") for row in rows[:10]],
                }
            )
        return trace

    async def _collect_attachment_evidence(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        final_k: int,
        tool_context: ToolExecutionContext | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if tool_context is None or "search_uploaded_documents" not in self.registry.names():
            return [], []

        result, record = await self.registry.call(
            "search_uploaded_documents",
            {"query": retrieval_question, "top_k": max(final_k, 6)},
            context=tool_context,
        )
        trace = [
            f"{record.name}:{record.result_count}" if not record.error else f"{record.name}:error"
        ]
        if record.error:
            return [], trace

        rows = self._evidence_rows("search_uploaded_documents", result)
        if not self.generator.temporary_context_is_relevant(question, rows):
            trace.append("attachment_relevant:false")
            return [], trace

        self.evidence_store.add_documents(session_id, rows)
        return rows, trace

    @staticmethod
    def _evidence_rows(tool_name: str, result: Any) -> list[dict[str, Any]]:
        rows = result if isinstance(result, list) else [result] if isinstance(result, dict) else []
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            if not item.get("chunk_id"):
                identity = str(item.get("url") or item.get("text") or json.dumps(item, sort_keys=True))
                item["chunk_id"] = f"web_{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]}"
            source_kind = {
                "search_history": "history",
                "search_uploaded_documents": "attachment",
            }.get(tool_name, "web")
            item.setdefault("source_kind", source_kind)
            normalized.append(item)
        return normalized
