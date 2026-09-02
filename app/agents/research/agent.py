from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

from app.agents.common.model_runtime import RoleLLMBackend
from app.agents.common.model_registry import ROLE_MODELS
from app.agents.research.policy import (
    RESEARCH_AGENT_SYSTEM,
    FinishDecision,
    ResearchPolicyState,
    ToolDecision,
    serialize_policy_state,
    validate_runtime_decision,
)
from app.agents.research.schemas import ResearchResult
from app.rag.retrieval import build_comparison_target_queries
from app.tools.evidence_tools import SessionEvidenceStore
from app.tools.registry import ToolExecutionContext, ToolRegistry
from app.telemetry import current_request_telemetry, log_event


logger = logging.getLogger(__name__)
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|9[0-9]{2})\b")
EXTERNAL_TOOL_NAMES = {"search_wikipedia", "fetch_wikipedia_page", "search_web", "fetch_web_page"}
VERIFICATION_CUES = {
    "co that su",
    "co dung la",
    "tin don",
    "thuc hu",
    "bang chung",
    "kiem chung",
    "that hay khong",
}
DISPUTED_CUES = {
    "tranh cai",
    "gay tranh cai",
    "bat dong",
    "mau thuan",
    "phu nhan",
}
EVALUATIVE_CUES = {
    "gioi nhat",
    "xuat sac nhat",
    "quan trong nhat",
    "tot nhat",
    "vi dai nhat",
    "duoc danh gia cao nhat",
}


def owner_context_ready(context: ToolExecutionContext) -> bool:
    return bool(context.owner_id and context.conversation_id)


def _years(value: str) -> set[str]:
    return set(YEAR_RE.findall(str(value)))


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", str(value).lower())
        if len(token) > 1 and token not in {"năm", "nao", "nào", "nhu", "như", "the", "thế", "nào", "chiến", "thắng"}
    }


def _normalize_text(value: str) -> str:
    value = str(value).lower()
    replacements = {
        "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a", "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a", "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
        "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e", "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
        "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
        "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o", "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o", "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
        "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u", "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
        "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y", "đ": "d",
    }
    return "".join(replacements.get(char, char) for char in value)


def _external_research_reason(question: str, local_state: dict[str, Any] | None = None) -> str | None:
    normalized = _normalize_text(question)
    local_state = local_state or {}
    local_count = int(local_state.get("local_result_count") or 0)
    source_kinds = set(local_state.get("source_kinds") or [])
    if any(cue in normalized for cue in VERIFICATION_CUES):
        return "verification_or_rumor"
    if any(cue in normalized for cue in DISPUTED_CUES):
        return "disputed_claim"
    if any(cue in normalized for cue in EVALUATIVE_CUES):
        return "evaluative_superlative"
    if local_count == 0 and re.search(
        r"\b(lich su|su kien|tran|chien dich|chien thang|nha [a-z]|nam \d{3,4}|vua|tuong|khoi nghia)\b",
        normalized,
    ):
        return "no_local_evidence"
    if source_kinds and source_kinds <= {"history"} and re.search(r"\b(kiem chung|danh gia|tranh cai)\b", normalized):
        return "source_diversity"
    return None


def needs_external_research(question: str, local_state: dict[str, Any] | None = None) -> bool:
    return _external_research_reason(question, local_state) is not None


def _wikipedia_candidate_score(question: str, row: dict[str, Any]) -> tuple[float, bool]:
    question_years = _years(question)
    title = str(row.get("title") or "")
    text = str(row.get("text") or "")
    title_years = _years(title)
    text_years = _years(text)
    conflict = bool(question_years and title_years and not (question_years & title_years))
    if conflict:
        return -100.0, True
    query_terms = _tokens(question)
    candidate_terms = _tokens(f"{title} {text}")
    overlap = len(query_terms & candidate_terms) / max(len(query_terms), 1)
    year_bonus = 2.0 if question_years & (title_years | text_years) else 0.0
    title_bonus = 0.6 if query_terms & _tokens(title) else 0.0
    return overlap + year_bonus + title_bonus, False


def _select_wikipedia_candidate(question: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, int]:
    ranked = [
        (*_wikipedia_candidate_score(question, row), index, row)
        for index, row in enumerate(rows)
    ]
    if not ranked:
        return None, 0
    year_conflict_rejections = sum(1 for _, conflict, _, _ in ranked if conflict)
    ranked.sort(key=lambda item: (item[0], -item[2]), reverse=True)
    selected = ranked[0][3]
    if ranked[0][0] < 0:
        return None, year_conflict_rejections
    return selected, year_conflict_rejections


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
        max_wikipedia_searches: int = 2,
        max_web_searches: int = 3,
        max_page_fetches: int = 5,
    ):
        self.registry = registry
        self.evidence_store = evidence_store
        self.retrieval_runtime = retrieval_runtime
        self.model_runtime = model_runtime
        self.max_steps = max_steps
        self.max_wikipedia_searches = max_wikipedia_searches
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
        analysis = self.retrieval_runtime.retriever.analyze_question(question)
        classifier = getattr(self.retrieval_runtime.retriever, "classify_question", None)
        classification = classifier(question) if callable(classifier) else {}
        gate_result = str(classification.get("domain_gate_result") or ("out_of_domain" if classification.get("is_ood") else "in_domain"))
        if gate_result in {"out_of_domain", "meta", "ambiguous"}:
            elapsed_ms = (time.perf_counter() - attempt_started) * 1000
            if telemetry is not None:
                telemetry.research_ms += elapsed_ms
                telemetry.retrieval_skipped_due_to_ood = gate_result == "out_of_domain"
                telemetry.llm_calls_skipped_due_to_ood = True
            return ResearchResult(
                question=question,
                evidence=[],
                tool_trace=[f"domain_gate:{gate_result}"],
                is_ood=gate_result == "out_of_domain",
                ood_reason=str(classification.get("ood_reason") or ""),
                analysis={
                    **analysis,
                    "retrieval_question": retrieval_question,
                    "history_used_for_retrieval": history_used,
                    "domain_gate_result": gate_result,
                    "domain_gate_reason": classification.get("domain_gate_reason"),
                    **({"intent": classification["intent"]} if classification.get("intent") else {}),
                },
                debug={
                    "steps": 0,
                    "generation_calls": 0,
                    "json_repairs": 0,
                    "elapsed_ms": elapsed_ms,
                    "tools": [],
                    "evidence_ids": [],
                    "retrieval_question": retrieval_question,
                    "history_used_for_retrieval": history_used,
                    "domain_gate_result": gate_result,
                },
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
                final_k=final_k,
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
        external_steps = [
            step for step in policy_steps
            if step.get("external_research") or step.get("external_fallback")
        ]
        external_tools_called = [
            str(step.get("tool_name"))
            for step in external_steps
            if step.get("action") == "tool" and step.get("tool_name")
        ]
        external_needed = any(bool(step.get("external_research_needed")) for step in policy_steps)
        external_available = any(bool(step.get("external_research_available")) for step in policy_steps)
        external_skip_reason = next(
            (
                str(step.get("external_research_skip_reason"))
                for step in policy_steps
                if step.get("external_research_skip_reason")
            ),
            None,
        )
        external_reason = next(
            (
                str(step.get("external_research_reason"))
                for step in policy_steps
                if step.get("external_research_reason")
            ),
            None,
        )
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
                "external_research_needed": external_needed,
                "external_research_available": external_available,
                "external_research_reason": external_reason,
                "external_research_skip_reason": external_skip_reason,
                "external_tools_called": external_tools_called,
                "external_results_count": sum(int(step.get("result_count") or 0) for step in external_steps),
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
            "target_specific_queries": build_comparison_target_queries(
                question,
                self.retrieval_runtime.retriever.analyze_question(question),
            ),
        })
        local_state = self._local_state_from_observations([
            {"tool": "search_uploaded_documents", "result_count": len(attachment_chunks), "result": attachment_chunks[:5]},
            {"tool": "search_history", "result_count": len(chunks), "result": chunks[:5]},
        ])
        external_reason = _external_research_reason(question, local_state)
        if external_reason:
            external_trace, external_steps, _ = await self._run_external_research(
                question=question,
                session_id=session_id,
                start_step=len(policy_steps) + 1,
                tool_context=tool_context,
                request_id=None,
                reason=external_reason,
                fallback=external_reason == "no_local_evidence",
            )
            tool_trace.extend(external_trace)
            policy_steps.extend(external_steps)
        return tool_trace, policy_steps

    async def _run_model_policy(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        final_k: int,
        tool_context: ToolExecutionContext | None,
        request_id: str | None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        prefetch_trace, prefetch_steps, observations = await self._prefetch_local_evidence(
            question=question,
            retrieval_question=retrieval_question,
            session_id=session_id,
            final_k=final_k,
            tool_context=tool_context,
            request_id=request_id,
        )
        trace: list[str] = prefetch_trace
        policy_steps: list[dict[str, Any]] = prefetch_steps
        external_reason = _external_research_reason(
            question,
            self._local_state_from_observations(observations),
        )
        if external_reason:
            external_trace, external_steps, external_observations = await self._run_external_research(
                question=question,
                session_id=session_id,
                start_step=len(policy_steps) + 1,
                tool_context=tool_context,
                request_id=request_id,
                reason=external_reason,
                fallback=external_reason == "no_local_evidence",
            )
            trace.extend(external_trace)
            policy_steps.extend(external_steps)
            observations.extend(external_observations)
        seen_tool_requests: set[str] = set()
        wikipedia_searches = 0
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
                max_new_tokens=int(ROLE_MODELS["research"].generation["max_new_tokens"]),
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
                local_evidence_count = self._local_evidence_count(observations)
                missing_information = list(decision.missing_information)
                if local_evidence_count:
                    missing_information = [
                        item for item in missing_information
                        if "no supporting local evidence was retrieved" not in item.lower()
                    ]
                    if not missing_information and not decision.sufficient:
                        missing_information = [
                            "Local evidence was retrieved, but more specific support may be needed."
                        ]
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
                    "missing_information": missing_information,
                    "actual_generation_calls": calls_after_policy - calls_before,
                    "json_repairs": repairs_after_policy - repairs_before,
                    "elapsed_ms": elapsed_ms,
                })
                break
            assert isinstance(decision, ToolDecision)
            tool_name = decision.tool_name
            arguments = dict(decision.arguments)
            local_evidence_count = self._local_evidence_count(observations)
            if tool_name in {"search_history", "retrieve_evidence", "inspect_evidence"}:
                elapsed_ms = (time.perf_counter() - step_started) * 1000
                action = "prefetch_satisfied" if local_evidence_count else "prefetch_exhausted"
                if tool_name == "inspect_evidence" and local_evidence_count:
                    action = "duplicate_inspect_skipped"
                    if telemetry is not None:
                        telemetry.duplicate_inspect_skipped = True
                trace.append(f"agent:{action}:{tool_name}")
                policy_steps.append({
                    "step": step,
                    "action": action,
                    "tool_name": tool_name,
                    "arguments": {
                        key: value for key, value in arguments.items()
                        if key != "session_id"
                    },
                    "result_count": local_evidence_count,
                    "error": None,
                    "evidence": [
                        self._evidence_preview(item)
                        for item in self.evidence_store.all(session_id)[:10]
                    ],
                    "actual_generation_calls": calls_after_policy - calls_before,
                    "json_repairs": repairs_after_policy - repairs_before,
                    "elapsed_ms": elapsed_ms,
                    "deterministic_prefetch_resolution": True,
                })
                log_event(
                    "RESEARCH_PREFETCH_RESOLVED_TOOL_DECISION",
                    request_id=request_id,
                    step=step,
                    requested_tool=tool_name,
                    action=action,
                    local_evidence_count=local_evidence_count,
                    actual_generation_calls_for_step=calls_after_policy - calls_before,
                )
                break
            request_fingerprint = json.dumps(
                {"tool_name": tool_name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
            )
            if request_fingerprint in seen_tool_requests:
                observations.append({"tool": tool_name, "error": "duplicate_tool_request_skipped"})
                continue
            seen_tool_requests.add(request_fingerprint)
            if tool_name == "search_wikipedia":
                if wikipedia_searches >= self.max_wikipedia_searches:
                    observations.append({"tool": tool_name, "error": "wikipedia_search_budget_exhausted"})
                    continue
                wikipedia_searches += 1
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
                telemetry.tool_calls_by_type[tool_name] = telemetry.tool_calls_by_type.get(tool_name, 0) + 1
                if tool_name == "search_history":
                    telemetry.retrieval_ms += tool_elapsed_ms
                if tool_name in {"search_wikipedia", "fetch_wikipedia_page"}:
                    telemetry.wikipedia_calls += 1
                    if tool_name == "search_wikipedia":
                        telemetry.wikipedia_search_count += 1
                    if tool_name == "fetch_wikipedia_page":
                        telemetry.wikipedia_fetch_count += 1
                    telemetry.wikipedia_ms += tool_elapsed_ms
                if tool_name in {"search_web", "fetch_web_page"}:
                    telemetry.generic_web_calls += 1
                    telemetry.generic_web_ms += tool_elapsed_ms
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
                    "result": [self._evidence_preview(item) for item in rows[:5]],
                }
            )
        if self._needs_external_fallback(policy_steps, observations):
            fallback_trace, fallback_steps, fallback_observations = await self._run_external_research(
                question=question,
                session_id=session_id,
                start_step=len(policy_steps) + 1,
                tool_context=tool_context,
                request_id=request_id,
                reason="insufficient_local_evidence",
                fallback=True,
            )
            trace.extend(fallback_trace)
            policy_steps.extend(fallback_steps)
            observations.extend(fallback_observations)
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

    async def _prefetch_local_evidence(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        final_k: int,
        tool_context: ToolExecutionContext | None,
        request_id: str | None,
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        trace: list[str] = []
        policy_steps: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.research_prefetch_used = True

        attachment_chunks, attachment_trace = await self._collect_attachment_evidence(
            question=question,
            retrieval_question=retrieval_question,
            session_id=session_id,
            final_k=final_k,
            tool_context=tool_context,
        )
        trace.extend(attachment_trace)
        if attachment_trace:
            policy_steps.append({
                "step": len(policy_steps) + 1,
                "action": "tool",
                "tool_name": "search_uploaded_documents",
                "arguments": {"query": retrieval_question, "top_k": max(final_k, 8)},
                "result_count": len(attachment_chunks),
                "error": None,
                "evidence": [self._evidence_preview(item) for item in attachment_chunks[:10]],
                "actual_generation_calls": 0,
                "json_repairs": 0,
                "elapsed_ms": 0.0,
                "deterministic_prefetch": True,
            })
            observations.append({
                "tool": "search_uploaded_documents",
                "result_count": len(attachment_chunks),
                "evidence_ids": [item.get("chunk_id") for item in attachment_chunks[:10] if item.get("chunk_id")],
                "result": [self._evidence_preview(item) for item in attachment_chunks[:5]],
            })

        tool_started = time.perf_counter()
        chunks, record = await self.registry.call(
            "search_history",
            {"query": retrieval_question, "top_k": max(final_k, 8)},
            context=tool_context,
        )
        tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000
        chunks = chunks or []
        trace.append(
            f"{record.name}:{record.result_count}" if not record.error else f"{record.name}:error"
        )
        if not record.error:
            rows = self._evidence_rows("search_history", chunks)
            self.evidence_store.add_documents(session_id, rows)
        else:
            rows = []
        if telemetry is not None:
            telemetry.tool_calls += 1
            telemetry.tool_calls_by_type["search_history"] = telemetry.tool_calls_by_type.get("search_history", 0) + 1
            telemetry.retrieval_ms += tool_elapsed_ms
        step = {
            "step": len(policy_steps) + 1,
            "action": "tool",
            "tool_name": "search_history",
            "arguments": {"query": retrieval_question, "top_k": max(final_k, 8)},
            "result_count": len(rows),
            "error": record.error,
            "evidence": [self._evidence_preview(item) for item in rows[:10]],
            "actual_generation_calls": 0,
            "json_repairs": 0,
            "elapsed_ms": tool_elapsed_ms,
            "deterministic_prefetch": True,
            "target_specific_queries": build_comparison_target_queries(
                question,
                self.retrieval_runtime.retriever.analyze_question(question),
            ),
        }
        policy_steps.append(step)
        observations.append({
            "tool": "search_history",
            "result_count": len(rows),
            "evidence_ids": [row.get("chunk_id") for row in rows[:10] if row.get("chunk_id")],
            "result": [self._evidence_preview(item) for item in rows[:5]],
            **({"error": record.error} if record.error else {}),
        })
        log_event(
            "RESEARCH_PREFETCH_COMPLETE",
            request_id=request_id,
            search_history_count=len(rows),
            attachment_count=len(attachment_chunks),
            elapsed_ms=tool_elapsed_ms,
        )
        return trace, policy_steps, observations

    @staticmethod
    def _local_evidence_count(observations: list[dict[str, Any]]) -> int:
        return sum(
            int(observation.get("result_count") or 0)
            for observation in observations
            if observation.get("tool") in {
                "search_history",
                "search_uploaded_documents",
                "retrieve_evidence",
                "inspect_evidence",
            }
        )

    @staticmethod
    def _needs_external_fallback(
        policy_steps: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> bool:
        finish = next((step for step in reversed(policy_steps) if step.get("action") == "finish"), None)
        exhausted = next((step for step in reversed(policy_steps) if step.get("action") == "prefetch_exhausted"), None)
        if finish and bool(finish.get("sufficient")):
            return False
        if not finish and not exhausted:
            return False
        external_success = any(
            observation.get("tool") in {"search_wikipedia", "fetch_wikipedia_page", "search_web", "fetch_web_page"}
            and int(observation.get("result_count") or 0) > 0
            and not observation.get("error")
            for observation in observations
        )
        return not external_success

    @staticmethod
    def _local_state_from_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
        local_tools = {"search_history", "search_uploaded_documents", "retrieve_evidence", "inspect_evidence"}
        local_result_count = sum(
            int(observation.get("result_count") or 0)
            for observation in observations
            if observation.get("tool") in local_tools
        )
        source_kinds: set[str] = set()
        titles: list[str] = []
        for observation in observations:
            if observation.get("tool") not in local_tools:
                continue
            for item in observation.get("result") or []:
                if not isinstance(item, dict):
                    continue
                source_kinds.add(str(item.get("source_kind") or item.get("source_type") or "history"))
                if item.get("title"):
                    titles.append(str(item.get("title")))
        return {
            "local_result_count": local_result_count,
            "source_kinds": sorted(source_kinds),
            "titles": list(dict.fromkeys(titles)),
        }

    async def _run_external_research(
        self,
        *,
        question: str,
        session_id: str,
        start_step: int,
        tool_context: ToolExecutionContext | None,
        request_id: str | None,
        reason: str,
        fallback: bool,
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        registry_names = set(self.registry.names())
        available_names = registry_names & EXTERNAL_TOOL_NAMES
        if not available_names or (
            self.max_wikipedia_searches <= 0
            and self.max_web_searches <= 0
        ):
            telemetry = current_request_telemetry()
            if telemetry is not None:
                telemetry.external_research_needed = True
                telemetry.external_research_available = False
                telemetry.external_research_reason = reason
                telemetry.external_research_skip_reason = "no_configured_external_tools"
            return [
                "external_research:unavailable"
            ], [{
                "step": start_step,
                "action": "external_research_skipped",
                "external_research": True,
                "external_research_needed": True,
                "external_research_available": False,
                "external_research_reason": reason,
                "external_research_skip_reason": "no_configured_external_tools",
                "result_count": 0,
                "actual_generation_calls": 0,
                "json_repairs": 0,
                "elapsed_ms": 0.0,
            }], []

        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.external_fallback_triggered = telemetry.external_fallback_triggered or fallback
            telemetry.external_research_needed = True
            telemetry.external_research_available = True
            telemetry.external_research_reason = reason
        log_event("RESEARCH_EXTERNAL_RESEARCH_START", request_id=request_id, reason=reason)
        trace: list[str] = []
        policy_steps: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []

        search_query = question
        if "search_wikipedia" in registry_names and self.max_wikipedia_searches > 0:
            search_started = time.perf_counter()
            search_rows, search_record = await self.registry.call(
                "search_wikipedia",
                {"query": search_query, "language": "vi", "top_k": 5},
                context=tool_context,
            )
            search_elapsed_ms = (time.perf_counter() - search_started) * 1000
            search_rows = search_rows or []
            selected_wiki, year_conflicts = _select_wikipedia_candidate(question, search_rows)
            if selected_wiki is None and _years(question):
                reformulated_query = " ".join(
                    token
                    for token in ["Trận Bạch Đằng", *sorted(_years(question))]
                    if token
                )
                search_query = reformulated_query
                reform_started = time.perf_counter()
                search_rows, search_record = await self.registry.call(
                    "search_wikipedia",
                    {"query": search_query, "language": "vi", "top_k": 5},
                    context=tool_context,
                )
                search_elapsed_ms += (time.perf_counter() - reform_started) * 1000
                search_rows = search_rows or []
                selected_wiki, reform_conflicts = _select_wikipedia_candidate(question, search_rows)
                year_conflicts += reform_conflicts
            search_evidence = self._evidence_rows("search_wikipedia", search_rows)
            self.evidence_store.add_documents(session_id, search_evidence)
            trace.append(
                f"{search_record.name}:{search_record.result_count}" if not search_record.error else f"{search_record.name}:error"
            )
            if telemetry is not None:
                telemetry.tool_calls += 1
                telemetry.tool_calls_by_type["search_wikipedia"] = telemetry.tool_calls_by_type.get("search_wikipedia", 0) + 1
                telemetry.wikipedia_calls += 1
                telemetry.wikipedia_search_count += 1
                telemetry.wikipedia_ms += search_elapsed_ms
                telemetry.wikipedia_query = search_query
                telemetry.wikipedia_candidate_titles = [str(row.get("title") or "") for row in search_evidence]
                telemetry.wikipedia_selected_title = str(selected_wiki.get("title") or "") if selected_wiki else None
                telemetry.wikipedia_year_conflict_rejections += year_conflicts
                telemetry.external_tools_called = list(dict.fromkeys([
                    *telemetry.external_tools_called,
                    "search_wikipedia",
                ]))
                telemetry.external_results_count += len(search_evidence)
            policy_steps.append({
                "step": start_step,
                "action": "tool",
                "tool_name": "search_wikipedia",
                "arguments": {"query": search_query, "language": "vi", "top_k": 5},
                "result_count": len(search_evidence),
                "error": search_record.error,
                "evidence": [self._evidence_preview(item) for item in search_evidence[:10]],
                "actual_generation_calls": 0,
                "json_repairs": 0,
                "elapsed_ms": search_elapsed_ms,
                "external_fallback": fallback,
                "external_research": True,
                "external_research_needed": True,
                "external_research_available": True,
                "external_research_reason": reason,
                "wikipedia_candidate_titles": [str(row.get("title") or "") for row in search_evidence],
                "wikipedia_selected_title": str(selected_wiki.get("title") or "") if selected_wiki else None,
                "wikipedia_year_conflict_rejections": year_conflicts,
            })
            observations.append({
                "tool": "search_wikipedia",
                "result_count": len(search_evidence),
                "evidence_ids": [row.get("chunk_id") for row in search_evidence[:10] if row.get("chunk_id")],
                "result": [self._evidence_preview(item) for item in search_evidence[:5]],
                **({"error": search_record.error} if search_record.error else {}),
            })
            if not search_record.error and selected_wiki is not None and "fetch_wikipedia_page" in registry_names:
                fetch_trace, fetch_steps, fetch_observations = await self._fetch_best_wikipedia_page(
                    question=question,
                    session_id=session_id,
                    start_step=start_step + 1,
                    tool_context=tool_context,
                    best=selected_wiki,
                    fallback=fallback,
                    reason=reason,
                )
                trace.extend(fetch_trace)
                policy_steps.extend(fetch_steps)
                observations.extend(fetch_observations)
            return trace, policy_steps, observations

        if "search_web" in registry_names and self.max_web_searches > 0:
            search_started = time.perf_counter()
            web_rows, web_record = await self.registry.call(
                "search_web",
                {"query": question, "top_k": 5},
                context=tool_context,
            )
            search_elapsed_ms = (time.perf_counter() - search_started) * 1000
            web_evidence = self._evidence_rows("search_web", web_rows or [])
            self.evidence_store.add_documents(session_id, web_evidence)
            trace.append(
                f"{web_record.name}:{web_record.result_count}" if not web_record.error else f"{web_record.name}:error"
            )
            if telemetry is not None:
                telemetry.tool_calls += 1
                telemetry.tool_calls_by_type["search_web"] = telemetry.tool_calls_by_type.get("search_web", 0) + 1
                telemetry.generic_web_calls += 1
                telemetry.generic_web_ms += search_elapsed_ms
                telemetry.external_tools_called = list(dict.fromkeys([
                    *telemetry.external_tools_called,
                    "search_web",
                ]))
                telemetry.external_results_count += len(web_evidence)
            policy_steps.append({
                "step": start_step,
                "action": "tool",
                "tool_name": "search_web",
                "arguments": {"query": question, "top_k": 5},
                "result_count": len(web_evidence),
                "error": web_record.error,
                "evidence": [self._evidence_preview(item) for item in web_evidence[:10]],
                "actual_generation_calls": 0,
                "json_repairs": 0,
                "elapsed_ms": search_elapsed_ms,
                "external_fallback": fallback,
                "external_research": True,
                "external_research_needed": True,
                "external_research_available": True,
                "external_research_reason": reason,
            })
            observations.append({
                "tool": "search_web",
                "result_count": len(web_evidence),
                "evidence_ids": [row.get("chunk_id") for row in web_evidence[:10] if row.get("chunk_id")],
                "result": [self._evidence_preview(item) for item in web_evidence[:5]],
                **({"error": web_record.error} if web_record.error else {}),
            })
            return trace, policy_steps, observations

        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.external_research_needed = True
            telemetry.external_research_available = False
            telemetry.external_research_reason = reason
            telemetry.external_research_skip_reason = "external_tool_budget_exhausted"
        return [
            "external_research:unavailable"
        ], [{
            "step": start_step,
            "action": "external_research_skipped",
            "external_research": True,
            "external_research_needed": True,
            "external_research_available": False,
            "external_research_reason": reason,
            "external_research_skip_reason": "external_tool_budget_exhausted",
            "result_count": 0,
            "actual_generation_calls": 0,
            "json_repairs": 0,
            "elapsed_ms": 0.0,
        }], []

    async def _fetch_best_wikipedia_page(
        self,
        *,
        question: str,
        session_id: str,
        start_step: int,
        tool_context: ToolExecutionContext | None,
        best: dict[str, Any],
        fallback: bool,
        reason: str,
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        trace: list[str] = []
        policy_steps: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        metadata = best.get("metadata") if isinstance(best.get("metadata"), dict) else {}
        page_key = str(metadata.get("page_id") or best.get("title") or "").strip()
        if not page_key:
            return trace, policy_steps, observations

        fetch_started = time.perf_counter()
        fetched, fetch_record = await self.registry.call(
            "fetch_wikipedia_page",
            {"page_id_or_title": page_key, "language": "vi", "max_chars": 8000},
            context=tool_context,
        )
        fetch_elapsed_ms = (time.perf_counter() - fetch_started) * 1000
        fetched_rows = self._evidence_rows("fetch_wikipedia_page", fetched) if fetched else []
        self.evidence_store.add_documents(session_id, fetched_rows)
        trace.append(
            f"{fetch_record.name}:{fetch_record.result_count}" if not fetch_record.error else f"{fetch_record.name}:error"
        )
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.tool_calls += 1
            telemetry.tool_calls_by_type["fetch_wikipedia_page"] = telemetry.tool_calls_by_type.get("fetch_wikipedia_page", 0) + 1
            telemetry.wikipedia_calls += 1
            telemetry.wikipedia_fetch_count += 1
            telemetry.wikipedia_ms += fetch_elapsed_ms
            telemetry.external_tools_called = list(dict.fromkeys([
                *telemetry.external_tools_called,
                "fetch_wikipedia_page",
            ]))
            telemetry.external_results_count += len(fetched_rows)
        policy_steps.append({
            "step": start_step + 1,
            "action": "tool",
            "tool_name": "fetch_wikipedia_page",
            "arguments": {"page_id_or_title": page_key, "language": "vi", "max_chars": 8000},
            "result_count": len(fetched_rows),
            "error": fetch_record.error,
            "evidence": [self._evidence_preview(item) for item in fetched_rows[:5]],
            "actual_generation_calls": 0,
            "json_repairs": 0,
            "elapsed_ms": fetch_elapsed_ms,
            "external_fallback": fallback,
            "external_research": True,
            "external_research_needed": True,
            "external_research_available": True,
            "external_research_reason": reason,
        })
        observations.append({
            "tool": "fetch_wikipedia_page",
            "result_count": len(fetched_rows),
            "evidence_ids": [row.get("chunk_id") for row in fetched_rows[:5] if row.get("chunk_id")],
            "result": [self._evidence_preview(item) for item in fetched_rows[:3]],
            **({"error": fetch_record.error} if fetch_record.error else {}),
        })
        log_event(
            "RESEARCH_EXTERNAL_RESEARCH_END",
            request_id=tool_context.request_id if tool_context is not None else None,
            wikipedia_fetch_count=len(fetched_rows),
        )
        return trace, policy_steps, observations

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
            "retrieval_query_roles": item.get("retrieval_query_roles") or [],
            "comparison_target": item.get("comparison_target"),
            "incidental_target_penalty": item.get("incidental_target_penalty"),
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
                "search_wikipedia": "wikipedia",
                "fetch_wikipedia_page": "wikipedia",
            }.get(tool_name, "web")
            item.setdefault("source_kind", source_kind)
            normalized.append(item)
        return normalized
