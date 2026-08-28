from __future__ import annotations

import hashlib
import json
import logging
import time
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
from app.tools.evidence_tools import SessionEvidenceStore
from app.tools.registry import ToolExecutionContext, ToolRegistry
from app.telemetry import current_request_telemetry, log_event


logger = logging.getLogger(__name__)


def owner_context_ready(context: ToolExecutionContext) -> bool:
    return bool(context.owner_id and context.conversation_id)


class ResearchAgent:
    """Bounded PLAN/ACTION/OBSERVATION agent with a deterministic fallback."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        evidence_store: SessionEvidenceStore,
        retrieval_runtime: Any,
        model_runtime: RoleLLMBackend | None = None,
        max_steps: int = 6,
        max_web_searches: int = 3,
        max_page_fetches: int = 5,
    ):
        self.registry = registry
        self.evidence_store = evidence_store
        self.retrieval_runtime = retrieval_runtime
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
        request_id: str | None = None,
    ) -> ResearchResult:
        attempt_started = time.perf_counter()
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.research_attempts += 1
        normalized_history = self.retrieval_runtime.normalize_history(
            history,
            current_question=question,
        )
        retrieval_question, history_used = self.retrieval_runtime.build_retrieval_question(
            question,
            normalized_history,
        )
        tool_context = None
        if owner_id or conversation_id or request_id:
            tool_context = ToolExecutionContext(
                owner_id=owner_id,
                conversation_id=conversation_id,
                session_id=session_id,
                request_id=request_id,
            )
        if self.model_runtime is not None:
            tool_trace, policy_steps = await self._run_model_policy(
                question=question,
                retrieval_question=retrieval_question,
                session_id=session_id,
                tool_context=tool_context,
                request_id=request_id,
            )
        else:
            tool_trace, policy_steps = await self._run_fallback(
                question=question,
                retrieval_question=retrieval_question,
                session_id=session_id,
                final_k=final_k,
                tool_context=tool_context,
            )
        elapsed_ms = (time.perf_counter() - attempt_started) * 1000
        if telemetry is not None:
            telemetry.research_steps += len(policy_steps)
            telemetry.research_ms += elapsed_ms
        analysis = self.retrieval_runtime.retriever.analyze_question(question)
        classifier = getattr(self.retrieval_runtime.retriever, "classify_question", None)
        classification = classifier(question) if callable(classifier) else {}
        all_evidence = {
            str(item.get("chunk_id")): item
            for item in self.evidence_store.all(session_id)
            if item.get("chunk_id")
        }
        observed_ids = list(dict.fromkeys(
            str(item.get("chunk_id"))
            for step in policy_steps
            for item in step.get("evidence", [])
            if item.get("chunk_id")
        ))
        evidence = [all_evidence[evidence_id] for evidence_id in observed_ids if evidence_id in all_evidence]
        return ResearchResult(
            question=question,
            evidence=evidence,
            tool_trace=tool_trace,
            is_ood=bool(classification.get("is_ood")),
            ood_reason=str(classification.get("ood_reason") or ""),
            analysis={
                **analysis,
                "retrieval_question": retrieval_question,
                "history_used_for_retrieval": history_used,
                **({"intent": classification["intent"]} if classification.get("intent") else {}),
            },
            debug={
                "steps": len(policy_steps),
                "generation_calls": (
                    sum(int(step.get("actual_generation_calls", 0)) for step in policy_steps)
                    if self.model_runtime is not None
                    else 0
                ),
                "json_repairs": sum(int(step.get("json_repairs", 0)) for step in policy_steps),
                "elapsed_ms": elapsed_ms,
                "tools": policy_steps,
                "evidence_ids": [str(item.get("chunk_id")) for item in evidence],
                "retrieval_question": retrieval_question,
                "history_used_for_retrieval": history_used,
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
    ) -> tuple[list[str], list[dict[str, Any]]]:
        attachment_chunks, tool_trace = await self._collect_attachment_evidence(
            question=question,
            retrieval_question=retrieval_question,
            session_id=session_id,
            final_k=final_k,
            tool_context=tool_context,
        )
        policy_steps: list[dict[str, Any]] = []
        if tool_trace:
            policy_steps.append({
                "step": len(policy_steps) + 1,
                "action": "tool",
                "tool_name": "search_uploaded_documents",
                "arguments": {"query": retrieval_question, "top_k": max(final_k, 6)},
                "result_count": len(attachment_chunks),
                "evidence": [self._evidence_preview(item) for item in attachment_chunks[:10]],
            })
        chunks, record = await self.registry.call(
            "search_history",
            {"query": retrieval_question, "top_k": max(final_k, 6)},
        )
        tool_trace.append(
            f"{record.name}:{record.result_count}" if not record.error else f"{record.name}:error"
        )
        chunks = chunks or []
        self.evidence_store.add_documents(session_id, chunks)
        policy_steps.append({
            "step": len(policy_steps) + 1,
            "action": "tool",
            "tool_name": "search_history",
            "arguments": {"query": retrieval_question, "top_k": max(final_k, 6)},
            "result_count": len(chunks),
            "error": record.error,
            "evidence": [self._evidence_preview(item) for item in chunks[:10]],
        })
        if not attachment_chunks and not chunks and self.max_steps > 1 and "search_web" in self.registry.names():
            web_rows, web_record = await self.registry.call("search_web", {"query": question, "top_k": 5})
            tool_trace.append(
                f"{web_record.name}:{web_record.result_count}" if not web_record.error else f"{web_record.name}:error"
            )
            self.evidence_store.add_documents(session_id, web_rows or [])
            policy_steps.append({
                "step": len(policy_steps) + 1,
                "action": "tool",
                "tool_name": "search_web",
                "arguments": {"query": question, "top_k": 5},
                "result_count": len(web_rows or []),
                "error": web_record.error,
                "evidence": [self._evidence_preview(item) for item in (web_rows or [])[:10]],
            })
        return tool_trace, policy_steps

    async def _run_model_policy(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        tool_context: ToolExecutionContext | None,
        request_id: str | None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        trace: list[str] = []
        policy_steps: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        web_searches = 0
        page_fetches = 0
        tools = [
            tool
            for tool in self.registry.describe()
            if tool["name"] != "search_uploaded_documents" or (tool_context is not None and owner_context_ready(tool_context))
        ]
        available_tools = {tool["name"] for tool in tools}
        for step in range(1, self.max_steps + 1):
            step_started = time.perf_counter()
            telemetry = current_request_telemetry()
            calls_before = telemetry.total_llm_calls if telemetry is not None else 0
            repairs_before = telemetry.research_json_repairs if telemetry is not None else 0
            log_event(
                "RESEARCH_STEP_START",
                request_id=request_id,
                attempt=telemetry.research_attempts if telemetry is not None else None,
                step=step,
                max_steps=self.max_steps,
            )
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
            telemetry = current_request_telemetry()
            calls_after_policy = telemetry.total_llm_calls if telemetry is not None else calls_before + 1
            repairs_after_policy = telemetry.research_json_repairs if telemetry is not None else repairs_before
            try:
                decision = validate_runtime_decision(raw_decision, tool_names=available_tools)
            except (TypeError, ValueError) as exc:
                elapsed_ms = (time.perf_counter() - step_started) * 1000
                log_event(
                    "RESEARCH_POLICY_RESULT",
                    request_id=request_id,
                    attempt=telemetry.research_attempts if telemetry is not None else None,
                    step=step,
                    decision_type="invalid",
                    tool_name=None,
                    actual_generation_calls_for_step=calls_after_policy - calls_before,
                    elapsed_ms=elapsed_ms,
                )
                observations.append({"tool": "policy", "error": f"invalid_decision: {exc}"})
                trace.append("agent:invalid_decision")
                policy_steps.append({
                    "step": step,
                    "action": "invalid",
                    "error": f"invalid_decision:{type(exc).__name__}",
                    "actual_generation_calls": calls_after_policy - calls_before,
                    "json_repairs": repairs_after_policy - repairs_before,
                    "elapsed_ms": elapsed_ms,
                })
                continue
            if isinstance(decision, FinishDecision):
                elapsed_ms = (time.perf_counter() - step_started) * 1000
                log_event(
                    "RESEARCH_POLICY_RESULT",
                    request_id=request_id,
                    attempt=telemetry.research_attempts if telemetry is not None else None,
                    step=step,
                    decision_type="finish",
                    tool_name=None,
                    actual_generation_calls_for_step=calls_after_policy - calls_before,
                    elapsed_ms=elapsed_ms,
                )
                trace.append(f"agent:finish:{step}")
                policy_steps.append({
                    "step": step,
                    "action": "finish",
                    "sufficient": decision.sufficient,
                    "missing_information": decision.missing_information,
                    "actual_generation_calls": calls_after_policy - calls_before,
                    "json_repairs": repairs_after_policy - repairs_before,
                    "elapsed_ms": elapsed_ms,
                })
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
            tool_started = time.perf_counter()
            log_event(
                "RESEARCH_TOOL_START",
                request_id=request_id,
                tool_name=tool_name,
            )
            result, record = await self.registry.call(
                tool_name,
                arguments,
                context=tool_context,
            )
            tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000
            telemetry = current_request_telemetry()
            if telemetry is not None:
                telemetry.tool_calls += 1
                if tool_name == "search_history":
                    telemetry.retrieval_ms += tool_elapsed_ms
            log_event(
                "RESEARCH_TOOL_END",
                request_id=request_id,
                tool_name=tool_name,
                elapsed_ms=tool_elapsed_ms,
                result_count=record.result_count,
                error_type=type(record.error).__name__ if record.error else None,
            )
            trace.append(f"{record.name}:{record.result_count}" if not record.error else f"{record.name}:error")
            if record.error:
                elapsed_ms = (time.perf_counter() - step_started) * 1000
                log_event(
                    "RESEARCH_POLICY_RESULT",
                    request_id=request_id,
                    attempt=telemetry.research_attempts if telemetry is not None else None,
                    step=step,
                    decision_type="tool",
                    tool_name=tool_name,
                    actual_generation_calls_for_step=calls_after_policy - calls_before,
                    elapsed_ms=elapsed_ms,
                )
                observations.append({"tool": tool_name, "error": record.error})
                policy_steps.append({
                    "step": step,
                    "action": "tool",
                    "tool_name": tool_name,
                    "arguments": {
                        key: value for key, value in decision.arguments.items()
                        if key != "session_id"
                    },
                    "result_count": 0,
                    "error": record.error,
                    "evidence": [],
                    "actual_generation_calls": calls_after_policy - calls_before,
                    "json_repairs": repairs_after_policy - repairs_before,
                    "elapsed_ms": elapsed_ms,
                })
                continue
            rows = self._evidence_rows(tool_name, result)
            if (
                tool_name == "search_uploaded_documents"
                and not self.retrieval_runtime.temporary_context_is_relevant(
                    str(arguments.get("query") or question),
                    rows,
                )
            ):
                rows = []
                trace.append("attachment_relevant:false")
            self.evidence_store.add_documents(session_id, rows)
            elapsed_ms = (time.perf_counter() - step_started) * 1000
            log_event(
                "RESEARCH_POLICY_RESULT",
                request_id=request_id,
                attempt=telemetry.research_attempts if telemetry is not None else None,
                step=step,
                decision_type="tool",
                tool_name=tool_name,
                actual_generation_calls_for_step=calls_after_policy - calls_before,
                elapsed_ms=elapsed_ms,
            )
            policy_steps.append({
                "step": step,
                "action": "tool",
                "tool_name": tool_name,
                "arguments": {
                    key: value for key, value in decision.arguments.items()
                    if key != "session_id"
                },
                "result_count": len(rows),
                "error": None,
                "evidence": [self._evidence_preview(item) for item in rows[:10]],
                "actual_generation_calls": calls_after_policy - calls_before,
                "json_repairs": repairs_after_policy - repairs_before,
                "elapsed_ms": elapsed_ms,
            })
            observations.append(
                {
                    "tool": tool_name,
                    "result_count": len(rows),
                    "evidence_ids": [row.get("chunk_id") for row in rows[:10]],
                }
            )
        telemetry = current_request_telemetry()
        log_event(
            "RESEARCH_COMPLETE",
            request_id=request_id,
            attempt=telemetry.research_attempts if telemetry is not None else None,
            steps=len(policy_steps),
            actual_llm_calls=sum(int(step.get("actual_generation_calls", 0)) for step in policy_steps),
            json_repairs=sum(int(step.get("json_repairs", 0)) for step in policy_steps),
            tool_calls=sum(1 for step in policy_steps if step.get("action") == "tool"),
            evidence_count=len(self.evidence_store.all(session_id)),
            elapsed_ms=sum(float(step.get("elapsed_ms", 0.0)) for step in policy_steps),
        )
        return trace, policy_steps

    async def _collect_attachment_evidence(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        final_k: int,
        tool_context: ToolExecutionContext | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if (
            tool_context is None
            or not owner_context_ready(tool_context)
            or "search_uploaded_documents" not in self.registry.names()
        ):
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
        if not self.retrieval_runtime.temporary_context_is_relevant(question, rows):
            trace.append("attachment_relevant:false")
            return [], trace

        self.evidence_store.add_documents(session_id, rows)
        return rows, trace

    @staticmethod
    def _evidence_preview(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": str(item.get("chunk_id") or ""),
            "title": item.get("title"),
            "text_preview": str(item.get("text") or "")[:220],
            "best_dense_score": item.get("best_dense_score"),
            "best_bm25_score": item.get("best_bm25_score"),
            "rrf_score": item.get("rrf_score"),
            "reranker_score": item.get("reranker_score"),
            "final_retrieval_score": item.get("final_retrieval_score"),
        }

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
