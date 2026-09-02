from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from app.agents.central_citations import check_citations, expand_citations, citation_display_map
from app.agents.central_evidence import SynthesisEvidence, build_evidence_packet, render_evidence_packet, select_evidence, select_synthesis_evidence
from app.agents.central_analytical import annotate_evidence, coverage_report, coverage_select, evidence_targets, target_mentions
from app.agents.central_grounding import grounding_risks
from app.agents.central_model_runtime import CentralGeneration, CentralLLMBackend, CentralToolCall
from app.agents.central_policy import CentralRequestPolicy, HistoryGroundingPolicy
from app.agents.central_prompt import BIOGRAPHY_CONTRACT, CENTRAL_SYSTEM_PROMPT, REPAIR_CONTRACT, SYNTHESIS_CONTRACT, COMPARISON_CONTRACT, VIEWPOINT_CONTRACT
from app.agents.central_question import analytical_answer_issues, analyze_central_question, plan_analytical_queries
from app.agents.central_state import CentralAgentState, CentralPhase
from app.agents.central_tools import EXTERNAL_TOOLS, bounded_tool_arguments, normalize_tool_result, qwen_tool_schemas
from app.agents.config import CentralAgentConfig
from app.telemetry import current_request_telemetry
from app.tools.registry import ToolExecutionContext, ToolRegistry


INSUFFICIENT_EVIDENCE_ANSWER = (
    "Mình chưa tìm thấy đủ bằng chứng đáng tin cậy để trả lời câu hỏi này. "
    "Bạn có thể bổ sung tài liệu hoặc làm rõ giai đoạn, nhân vật hay sự kiện cần hỏi."
)


class CentralAgent:
    """A bounded, grounded state machine backed only by the Central model."""

    def __init__(
        self,
        *,
        model_runtime: CentralLLMBackend,
        tool_registry: ToolRegistry,
        config: CentralAgentConfig | None = None,
        has_uploaded_documents: Callable[[str, str], bool] | None = None,
        request_policy: CentralRequestPolicy | None = None,
    ):
        self.model_runtime = model_runtime
        self.tool_registry = tool_registry
        self.config = config or CentralAgentConfig()
        self.has_uploaded_documents = has_uploaded_documents
        self.request_policy = request_policy or HistoryGroundingPolicy()
        self.max_history_messages = 6

    def _allowed_tools(self, owner_id: str | None, conversation_id: str | None) -> set[str]:
        names = set(self.tool_registry.names())
        allowed: set[str] = set()
        if self.config.enable_history and "search_history" in names:
            allowed.add("search_history")
        if self.config.enable_wikipedia:
            allowed.update(names & {"search_wikipedia", "fetch_wikipedia_page"})
        external_web_usable = self.config.web_search_provider.strip().casefold() not in {
            "", "none", "disabled", "local-only",
        }
        if self.config.enable_web and external_web_usable:
            allowed.update(names & {"search_web", "fetch_web_page"})
        attachments_exist = False
        if (
            self.config.enable_documents
            and owner_id
            and conversation_id
            and self.has_uploaded_documents is not None
        ):
            try:
                attachments_exist = self.has_uploaded_documents(owner_id, conversation_id)
            except Exception:
                attachments_exist = False
        if attachments_exist and "search_uploaded_documents" in names:
            allowed.add("search_uploaded_documents")
        return allowed

    @staticmethod
    def _messages(question: str, history: list[dict[str, str]] | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": CENTRAL_SYSTEM_PROMPT}]
        for item in (history or [])[-6:]:
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _assistant_tool_message(calls: tuple[CentralToolCall, ...]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in calls
            ],
        }

    def _ensure_model_ready(self) -> Any:
        ensure_ready = getattr(self.model_runtime, "ensure_ready", None)
        return ensure_ready() if callable(ensure_ready) else self.model_runtime

    def _runtime_snapshot(self) -> dict[str, Any]:
        runtime = self.model_runtime
        instance = getattr(runtime, "_instance", None)
        target = instance if instance is not None else runtime
        lazy_not_loaded = instance is None and hasattr(runtime, "loaded")
        if lazy_not_loaded:
            target = None
        snapshot: dict[str, Any] = {
            "central_model_ready": bool(getattr(runtime, "is_ready", True)) if not lazy_not_loaded else False,
            "central_model_load_ms": float(getattr(runtime, "load_elapsed_ms", 0.0) or 0.0),
            "central_model_load_error": getattr(runtime, "load_error", None),
            "central_model_id": None,
            "central_adapter_configured": False,
            "central_adapter_loaded": False,
            "central_adapter_path": None,
            "central_adapter_source": "none",
        }
        if target is not None:
            adapter_path = getattr(target, "adapter_path", None)
            snapshot.update({
                "central_model_id": getattr(target, "model_id", None),
                "central_adapter_configured": bool(getattr(target, "adapter_configured", adapter_path is not None)),
                "central_adapter_loaded": bool(getattr(target, "adapter_loaded", False)),
                "central_adapter_path": str(adapter_path) if adapter_path else None,
                "central_adapter_source": str(getattr(target, "adapter_source", "peft" if adapter_path else "none")),
                "model_placement": dict(getattr(target, "placement", {}) or {}),
            })
            snapshot.update(dict(getattr(target, "cache_info", {}) or {}))
        return snapshot

    async def _generate(
        self,
        state: CentralAgentState,
        *,
        stage: str,
        tools: list[dict[str, Any]],
        max_new_tokens: int,
        progress: dict[str, Any] | None,
    ) -> CentralGeneration:
        if progress is not None:
            progress["timeout_stage"] = f"generation_{stage}"

        def invoke() -> CentralGeneration:
            return self.model_runtime.generate(
                messages=state.messages,
                tools=tools,
                max_new_tokens=max_new_tokens,
                stage=stage,
                deadline=state.deadline_monotonic,
            )

        generation = await asyncio.to_thread(invoke)
        state.model_calls += 1
        state.generation_ms += generation.generation_ms
        state.input_tokens += generation.input_tokens
        state.output_tokens += generation.output_tokens
        state.tool_parse_failures += generation.tool_parse_failures
        state.malformed_tool_calls.extend(generation.malformed_tool_calls)
        state.generation_metrics.append({
            "generation_stage": stage,
            "generation_stop_reason": generation.generation_stop_reason,
            "generation_hit_token_limit": generation.generation_hit_token_limit,
            "generation_hit_time_limit": generation.generation_hit_time_limit,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "generation_ms": generation.generation_ms,
        })
        return generation

    async def _execute_tool_calls(
        self,
        state: CentralAgentState,
        calls: tuple[CentralToolCall, ...],
        *,
        context: ToolExecutionContext,
        trace_phase: str,
        grounding_targets: dict[str, str] | None = None,
        progress: dict[str, Any] | None = None,
    ) -> None:
        state.messages.append(self._assistant_tool_message(calls))
        prepared: list[tuple[CentralToolCall, dict[str, Any], str | None, int | None]] = []
        pending: list[tuple[str, dict[str, Any]]] = []
        for call in calls:
            result_limit = self._retrieval_limit(state, call.name)
            arguments = bounded_tool_arguments(call.name, call.arguments, max_results=result_limit)
            signature = json.dumps([call.name, arguments], ensure_ascii=False, sort_keys=True)
            if signature in state.executed_tool_signatures:
                prepared.append((call, arguments, "duplicate_tool_call_prevented", None))
            elif call.name not in state.allowed_tools:
                prepared.append((call, arguments, "tool_not_available", None))
            else:
                state.executed_tool_signatures.add(signature)
                prepared.append((call, arguments, None, len(pending)))
                pending.append((call.name, arguments))

        async def execute(name: str, arguments: dict[str, Any]):
            started = time.perf_counter()
            try:
                result, record = await asyncio.wait_for(
                    self.tool_registry.call(name, arguments, context=context),
                    timeout=min(self.config.tool_timeout_seconds, max(0.01, state.remaining_seconds)),
                )
                error = record.error
                result_count = record.result_count
            except asyncio.TimeoutError:
                result, error, result_count = None, "tool_timeout", None
            return result, error, result_count, (time.perf_counter() - started) * 1000

        if progress is not None and pending:
            progress["timeout_stage"] = "tool_execution"
        executed = await asyncio.gather(*(execute(name, arguments) for name, arguments in pending)) if pending else []
        if executed:
            state.tool_ms += max(item[3] for item in executed)

        for call, arguments, immediate_error, pending_index in prepared:
            signature = json.dumps([call.name, arguments], ensure_ascii=False, sort_keys=True)
            target = (grounding_targets or {}).get(call.id)
            if target is None and state.question_analysis.comparison_targets:
                matches = [value for value in state.question_analysis.comparison_targets
                           if target_mentions(str(arguments.get("query") or arguments.get("page_id_or_title") or ""), value)]
                target = matches[0] if len(matches) == 1 else None
            comparison_target = target if target in state.question_analysis.comparison_targets else None
            sources: list[dict[str, Any]] = []
            error = immediate_error
            result_count: int | None = None
            elapsed = 0.0
            reused = bool(immediate_error == "duplicate_tool_call_prevented" and grounding_targets and signature in state.tool_results)
            if immediate_error is None or reused:
                if reused:
                    result, error, result_count = state.tool_results[signature]
                else:
                    assert pending_index is not None
                    result, error, result_count, elapsed = executed[pending_index]
                    state.tool_results[signature] = (result, error, result_count)
                    state.tool_calls += 1
                    state.tool_calls_by_name[call.name] += 1
                remaining = max(0, self.config.observation_char_budget - state.observation_chars)
                if error:
                    observation = json.dumps({"error": error}, ensure_ascii=False, separators=(",", ":"))
                else:
                    rows = result if isinstance(result, list) else ([] if result is None else [result])
                    result_limit = self._retrieval_limit(state, call.name)
                    rows = [dict(row) if isinstance(row, dict) else {"text": str(row)} for row in rows[:result_limit]]
                    rows = [{**row, "retrieval_tool": call.name, "retrieval_query": arguments.get("query"),
                             "citable": call.name != "search_wikipedia",
                             **({"comparison_target": comparison_target, "comparison_targets": [comparison_target]} if comparison_target else {})}
                            for row in rows]
                    state.retrieval_candidates.extend(rows)
                    filtered, filter_debug = select_evidence(rows, state.question_analysis, self.config, target=comparison_target)
                    state.retrieval_filter_events.append({"tool": call.name, "comparison_target": comparison_target, **filter_debug})
                    if comparison_target:
                        state.target_rankings.setdefault(comparison_target, []).extend(filtered)
                    observation, sources = normalize_tool_result(
                        call.name,
                        filtered,
                        max_results=result_limit,
                        char_budget=result_limit * 4500,
                    )
                    sources = [source for source in sources if str(source.get("text") or "").strip()]
                    if call.name == "search_wikipedia":
                        state.wikipedia_search_results.extend(sources)
                        sources = []  # Search snippets guide fetch, never establish sufficiency.
                    for source in sources:
                        source = annotate_evidence(source, state.question_analysis, comparison_target)
                        source_id = str(source["chunk_id"])
                        existing = state.source_by_id.get(source_id)
                        if existing is not None:
                            origins = list(dict.fromkeys(evidence_targets(existing) + evidence_targets(source)))
                            queries = list(dict.fromkeys([*existing.get("retrieval_queries", []), *source.get("retrieval_queries", []), str(arguments.get("query") or "")]))
                            if len(str(existing.get("text") or "")) > len(str(source.get("text") or "")):
                                source = dict(existing)
                            if origins:
                                source.update(comparison_target=origins[0], comparison_targets=origins)
                            source["retrieval_rank"] = min(existing.get("retrieval_rank", 999), source.get("retrieval_rank", 999))
                            source["retrieval_queries"] = queries
                        else:
                            source["retrieval_queries"] = [str(arguments.get("query") or "")]
                        state.source_by_id[source_id] = source
                    if call.name == "search_history":
                        state.local_evidence_count += len(sources)
                    elif call.name in EXTERNAL_TOOLS:
                        state.external_evidence_count += len(sources)
            else:
                observation = json.dumps({"error": immediate_error, "previous_observation": state.tool_observations.get(signature),
                                          "next_step": "Already executed; fetch a result, choose a different query, or finish. Do not repeat this signature."}, ensure_ascii=False)

            remaining = max(0, self.config.observation_char_budget - state.observation_chars)
            if len(observation) > remaining:
                compact_error = json.dumps({"observation": "Results retained host-side; observation preview budget exhausted.", "source_count": len(sources)}, separators=(",", ":"))
                observation = compact_error if len(compact_error) <= remaining else "{}"
            state.observation_chars += len(observation)
            state.tool_observations[signature] = observation
            if target is not None:
                state.initial_grounding_coverage[target] = sum(target in evidence_targets(source) for source in state.source_by_id.values()) if comparison_target else len(sources)
            state.tool_trace.append({
                "phase": trace_phase,
                "name": call.name,
                "arguments": arguments,
                "result_count": result_count,
                "error": error,
                "latency_ms": elapsed,
                "source_ids": [str(source["chunk_id"]) for source in sources],
                "grounding_target": target,
                "result_reused": reused,
            })
            state.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": observation,
            })

    def _retrieval_limit(self, state: CentralAgentState, tool_name: str) -> int:
        if tool_name == "search_history" and state.question_analysis.question_type in {"cause", "comparison"}:
            return self.config.analytical_retrieval_candidates
        return min(10, self.config.max_tool_results)

    def _initial_grounding_calls(self, state: CentralAgentState) -> tuple[tuple[CentralToolCall, ...], dict[str, str]]:
        if not state.grounding_required or "search_history" not in state.allowed_tools:
            return (), {}
        plan = plan_analytical_queries(state.question_analysis, self.config.analytical_query_variants)
        if state.question_analysis.comparison_targets:
            state.target_specific_queries = plan
        state.retrieval_query_variants = list(dict.fromkeys(query for queries in plan.values() for query in queries))
        calls: list[CentralToolCall] = []
        target_by_call: dict[str, str] = {}
        for target, queries in plan.items():
            for query in queries:
                call_id = f"central_ground_{len(calls) + 1:02d}"
                arguments = {"query": query, "top_k": self._retrieval_limit(state, "search_history")}
                if state.question_analysis.question_type in {"cause", "comparison"}:
                    arguments["candidate_pool"] = True
                calls.append(CentralToolCall(call_id, "search_history", arguments))
                target_by_call[call_id] = target
        return tuple(calls), target_by_call

    def _evidence_sufficient(self, state: CentralAgentState) -> bool:
        state.selected_sources = select_synthesis_evidence(list(state.source_by_id.values()), state.question_analysis, self.config)
        sufficient, coverage = coverage_report(state.selected_sources, state.retrieval_candidates, state.question_analysis, self.config)
        state.evidence_debug.update(coverage)
        checker = getattr(self.request_policy, "evidence_is_sufficient", None)
        if callable(checker):
            sufficient = sufficient and bool(checker(state))
        state.evidence_debug["evidence_sufficient"] = sufficient
        return sufficient

    async def _fetch_wikipedia_results(self, state: CentralAgentState, context: ToolExecutionContext, progress) -> None:
        if "fetch_wikipedia_page" not in state.allowed_tools:
            return
        calls = []
        targets = {}
        # At most one canonical fetched page per target/query. Snippets are ranked
        # by entity consistency and overview status before attempting a fetch.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in state.wikipedia_search_results:
            key = str(row.get("comparison_target") or row.get("retrieval_query") or "")
            if key not in state.fetched_wikipedia_targets:
                grouped.setdefault(key, []).append(row)
        for key, rows in grouped.items():
            state.fetched_wikipedia_targets.add(key)
            target = key if key in state.question_analysis.comparison_targets else state.question_analysis.event or state.question_analysis.subject
            annotated = [annotate_evidence(row, state.question_analysis, target) for row in rows]
            relevant = [row for row in annotated if row["target_consistent"]]
            if not relevant:
                continue
            best = coverage_select(relevant, state.question_analysis, 1)[0]
            metadata = best.get("metadata") or {}
            page = metadata.get("page_id") or best.get("page_id") or best.get("title")
            if not page:
                continue
            call_id = f"central_wiki_fetch_{len(state.tool_trace)}_{len(calls)}"
            calls.append(CentralToolCall(call_id, "fetch_wikipedia_page", {
                "page_id_or_title": str(page), "language": metadata.get("language", "vi"), "max_chars": 6000,
            }))
            if key in state.question_analysis.comparison_targets:
                targets[call_id] = key
        if calls:
            await self._execute_tool_calls(state, tuple(calls), context=context, trace_phase="wikipedia_fetch", grounding_targets=targets, progress=progress)

    async def _comparison_fallback(self, state: CentralAgentState, context: ToolExecutionContext, progress) -> None:
        if not {"search_wikipedia", "fetch_wikipedia_page"} <= state.allowed_tools:
            return
        self._evidence_sufficient(state)
        calls = []
        targets = {}
        for target, balance in state.evidence_debug["comparison_balance"].items():
            if balance["adequate"]:
                continue
            call_id = f"central_wiki_search_{len(calls)}"
            calls.append(CentralToolCall(call_id, "search_wikipedia", {"query": target, "language": "vi", "top_k": self.config.max_tool_results}))
            targets[call_id] = target
            if target not in state.target_specific_queries.setdefault(target, []):
                state.target_specific_queries[target].append(target)
        if calls:
            await self._execute_tool_calls(state, tuple(calls), context=context, trace_phase="wikipedia_search", grounding_targets=targets, progress=progress)
            await self._fetch_wikipedia_results(state, context, progress)

    def _prepare_synthesis(self, state: CentralAgentState) -> list[SynthesisEvidence]:
        self._evidence_sufficient(state)
        selected = state.selected_sources
        packet = build_evidence_packet(selected)
        rendered = render_evidence_packet(packet)
        reasons: Counter[str] = Counter()
        entity_reasons: Counter[str] = Counter()
        for event in state.retrieval_filter_events:
            reasons.update(event["retrieval_filter_reasons"])
            entity_reasons.update(event.get("entity_disambiguation_filter_reasons", {}))
        filtered_count = len(state.retrieval_candidates) - len(packet)
        unaccounted = filtered_count - sum(reasons.values())
        if unaccounted > 0:
            reasons["duplicate_empty_or_budget_limited"] += unaccounted
        display_map = citation_display_map(packet)
        by_id = {item["source_id"]: item for item in display_map.values()}
        state.selected_sources = [{**row, **by_id[str(row["chunk_id"])]} for row in selected]
        state.evidence_debug.update({
            **state.question_analysis.telemetry(),
            "retrieval_candidates_before_filter": len(state.retrieval_candidates),
            "retrieval_candidates_after_filter": len(packet),
            "retrieval_filtered_count": filtered_count,
            "retrieval_filter_reasons": dict(reasons),
            "biography_entity": state.question_analysis.subject,
            "biography_exact_title_hits": sum(event["biography_exact_title_hits"] for event in state.retrieval_filter_events),
            "evidence_input_chars": len(rendered),
            "evidence_source_count": len(packet),
            "citation_aliases": {item.alias: item.real_source_id for item in packet},
            "citation_display_map": display_map,
            "target_specific_queries": state.target_specific_queries,
            "target_rankings": state.target_rankings,
            "retrieval_query_variants": state.retrieval_query_variants,
            "entity_disambiguation_filtered_count": sum(entity_reasons.values()),
            "entity_disambiguation_filter_reasons": dict(entity_reasons),
            "chronology_downranked_count": sum(bool(row.get("chronology_downranked")) for row in state.source_by_id.values()),
        })
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.central_quality.update(state.evidence_debug)
        instruction = SYNTHESIS_CONTRACT + "\n" + VIEWPOINT_CONTRACT
        if state.question_analysis.question_type == "biography":
            instruction += "\n" + BIOGRAPHY_CONTRACT
        if len(state.question_analysis.comparison_targets) >= 2:
            instruction += "\n" + COMPARISON_CONTRACT
        # Discard action observations, including rejected rows and long source IDs.
        state.messages = self._messages(state.question, state.history)
        state.messages.append({"role": "user", "content": instruction + "\n\nGói bằng chứng:\n" + rendered})
        return packet

    def _check_answer(self, state: CentralAgentState, packet: list[SynthesisEvidence], *, stage: str):
        citations = check_citations(state.final_answer, packet)
        state.final_answer = citations.answer
        state.invalid_source_ids = citations.invalid
        risks = grounding_risks(citations.answer, state.question, packet)
        state.grounding_risk_checks.append({"stage": stage, **risks})
        issues: list[str] = []
        if risks["unsupported_named_claims"] or risks["unsupported_years"]:
            issues.append("unsupported_evidence_claim")
        if not citations.source_ids:
            issues.append("missing_valid_citations")
        if citations.invalid:
            issues.append("invalid_citation_aliases")
        if citations.uncited_paragraphs:
            issues.append("uncited_factual_paragraphs")
        if citations.target_mismatches:
            issues.append("comparison_citation_target_mismatch")
        if citations.unattributed_viewpoints:
            issues.append("unattributed_viewpoint")
        state.evidence_debug["citation_target_mismatches"] = citations.target_mismatches
        issues.extend(analytical_answer_issues(
            analysis=state.question_analysis, answer=citations.answer,
            source_ids=citations.source_ids, evidence_available=bool(packet),
        ))
        return list(dict.fromkeys(issues)), citations

    def _repair_budget(self, generation: CentralGeneration, answer: str) -> int:
        # Runtime token count is authoritative; fakes/older backends may omit it.
        estimated_tokens = generation.output_tokens or max(1, (len(answer) + 2) // 3)
        return min(self.config.repair_max_new_tokens, max(
            self.config.repair_min_new_tokens, estimated_tokens + self.config.repair_token_margin,
        ))

    async def _run(
        self,
        *,
        question: str,
        history: list[dict[str, str]] | None,
        owner_id: str | None,
        conversation_id: str | None,
        request_id: str | None,
        started: float,
        progress: dict[str, Any] | None,
    ) -> dict[str, Any]:
        allowed_tools = self._allowed_tools(owner_id, conversation_id)
        schemas = qwen_tool_schemas(self.tool_registry, allowed_tools)
        analysis = analyze_central_question(question)
        decision = self.request_policy.grounding_for(question)
        state = CentralAgentState(
            question=question,
            history=list(history or []),
            messages=self._messages(question, history),
            allowed_tools=allowed_tools,
            tool_schemas=schemas,
            question_analysis=analysis,
            grounding_required=decision.required,
            grounding_reason=decision.reason,
            deadline_monotonic=time.monotonic() + self.config.timeout_seconds,
        )
        context = ToolExecutionContext(
            owner_id=owner_id,
            conversation_id=conversation_id,
            request_id=request_id,
            session_id=request_id or conversation_id or "central",
        )

        state.transition(CentralPhase.INITIAL_GROUNDING)
        initial_calls, targets = self._initial_grounding_calls(state)
        if initial_calls:
            await self._execute_tool_calls(
                state,
                initial_calls,
                context=context,
                trace_phase=CentralPhase.INITIAL_GROUNDING.value,
                grounding_targets=targets,
                progress=progress,
            )

        sufficient = self._evidence_sufficient(state)
        if not sufficient and analysis.comparison_targets:
            await self._comparison_fallback(state, context, progress)
            sufficient = self._evidence_sufficient(state)
        # Comparison retrieval is fully deterministic and bounded. Exhausting its
        # search/fetch plan returns insufficient evidence, not a speculative answer.
        if not sufficient and not analysis.comparison_targets and schemas and self.config.max_action_rounds > 0:
            state.transition(CentralPhase.ACTION)
            for round_index in range(self.config.max_action_rounds):
                state.messages.append({"role": "user", "content": (
                    "Các lệnh sau đã được thực hiện, không lặp lại. Dùng kết quả, fetch trang, đổi truy vấn hoặc kết thúc:\n"
                    + "\n".join(sorted(state.executed_tool_signatures))
                )})
                generation = await self._generate(
                    state,
                    stage="action",
                    tools=schemas,
                    max_new_tokens=self.config.action_max_new_tokens,
                    progress=progress,
                )
                if generation.generation_hit_time_limit:
                    break
                if not generation.tool_calls:
                    break
                state.transition(CentralPhase.TOOL_EXECUTION)
                await self._execute_tool_calls(
                    state,
                    generation.tool_calls,
                    context=context,
                    trace_phase=f"action_round_{round_index + 1}",
                    progress=progress,
                )
                await self._fetch_wikipedia_results(state, context, progress)
                sufficient = self._evidence_sufficient(state)
                # A completed Wikipedia search already received its deterministic
                # fetch. Do not spend another generation repeating that search.
                completed_search = any(call.name == "search_wikipedia" for call in generation.tool_calls) and any(
                    row.get("name") == "search_wikipedia" and not row.get("error")
                    for row in state.tool_trace if row["phase"] == f"action_round_{round_index + 1}"
                )
                duplicate_only = all(row.get("error") == "duplicate_tool_call_prevented" for row in state.tool_trace if row["phase"] == f"action_round_{round_index + 1}")
                if sufficient or completed_search or duplicate_only or round_index + 1 >= self.config.max_action_rounds:
                    break
                state.transition(CentralPhase.ACTION)

        packet = self._prepare_synthesis(state)
        source_ids: list[str] = []
        if packet and state.evidence_debug["evidence_sufficient"]:
            if state.phase in {CentralPhase.INITIAL_GROUNDING, CentralPhase.ACTION, CentralPhase.TOOL_EXECUTION}:
                state.transition(CentralPhase.SYNTHESIS)
            generation = await self._generate(
                state,
                stage="synthesis",
                tools=[],
                max_new_tokens=self.config.final_max_new_tokens,
                progress=progress,
            )
            if not generation.tool_calls:
                state.final_answer = generation.content.strip()

            quality_issues, citations = self._check_answer(state, packet, stage="synthesis")
            if quality_issues and self.config.repair_max_generations:
                state.repair_attempted = True
                state.repair_reason = quality_issues[0]
                state.repair_budget = self._repair_budget(generation, state.final_answer)
                state.transition(CentralPhase.QUALITY_REPAIR)
                state.messages.extend([
                    {"role": "assistant", "content": state.final_answer},
                    {
                        "role": "user",
                        "content": (
                            REPAIR_CONTRACT + "\nLỗi cần sửa: " + ", ".join(quality_issues) + "."
                            "\nRủi ro: " + json.dumps(state.grounding_risk_checks[-1], ensure_ascii=False)
                        ),
                    },
                ])
                repaired = await self._generate(
                    state,
                    stage="quality_repair",
                    tools=[],
                    max_new_tokens=state.repair_budget,
                    progress=progress,
                )
                state.repair_used = True
                if repaired.content.strip() and not repaired.tool_calls:
                    state.final_answer = repaired.content.strip()
                quality_issues, citations = self._check_answer(state, packet, stage="quality_repair")
            elif not quality_issues:
                state.repair_avoided_reason = "citation_normalized" if citations.normalized else "valid_first_synthesis"
            else:
                state.repair_avoided_reason = "repair_disabled"
            if any(issue in quality_issues for issue in (
                "unsupported_evidence_claim", "missing_valid_citations", "invalid_citation_aliases", "uncited_factual_paragraphs",
                "comparison_citation_target_mismatch", "comparison_target_missing", "unattributed_viewpoint",
                "comparison_similarity_missing", "comparison_difference_missing",
            )):
                # A second failure cannot trigger a third call or pass as a grounded answer.
                state.final_answer = ""
            else:
                state.final_answer = expand_citations(state.final_answer, packet)
                source_ids = citations.source_ids
            state.transition(CentralPhase.FINAL)
        else:
            quality_issues = []
            if state.phase in {CentralPhase.INITIAL_GROUNDING, CentralPhase.ACTION, CentralPhase.TOOL_EXECUTION}:
                state.transition(CentralPhase.FINAL)

        status = "ok" if state.final_answer else "insufficient_evidence"
        if not state.final_answer:
            state.final_answer = INSUFFICIENT_EVIDENCE_ANSWER
        final_risks = state.grounding_risk_checks[-1] if state.grounding_risk_checks else {}
        quality_debug = {
            **state.evidence_debug,
            **analysis.telemetry(),
            "repair_reason": state.repair_reason,
            "repair_used": state.repair_used,
            "repair_avoided_reason": state.repair_avoided_reason,
            "repair_budget": state.repair_budget,
            "unsupported_named_claims": final_risks.get("unsupported_named_claims", []),
            "unsupported_years": final_risks.get("unsupported_years", []),
            "answer_quality_issues": quality_issues,
        }
        elapsed_ms = (time.perf_counter() - started) * 1000
        runtime = self._runtime_snapshot()
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.central_quality.update(quality_debug)
            telemetry.tool_calls += state.tool_calls
            for name, count in state.tool_calls_by_name.items():
                telemetry.tool_calls_by_type[name] = telemetry.tool_calls_by_type.get(name, 0) + count
            telemetry.external_results_count += state.external_evidence_count
            telemetry.external_tools_called.extend(
                name for name in state.tool_calls_by_name
                if name in EXTERNAL_TOOLS and name not in telemetry.external_tools_called
            )
            telemetry.central_tool_ms += state.tool_ms
            telemetry.central_external_results_count += state.external_evidence_count
            telemetry.central_tool_schema_count = len(schemas)
            telemetry.central_tools_exposed_to_model = [item["function"]["name"] for item in schemas]
            telemetry.central_tool_parse_failures += state.tool_parse_failures
            telemetry.central_malformed_tool_calls += len(state.malformed_tool_calls)

        provenance = {
            "mode": "central",
            "source": "central_qwen3_8b_v2",
            **runtime,
            "central_model_calls": state.model_calls,
            "central_tool_calls": state.tool_calls,
            "central_tool_calls_by_type": dict(state.tool_calls_by_name),
            "central_external_results_count": state.external_evidence_count,
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_generation_calls": 0,
            "total_llm_calls": state.model_calls,
            "repair_generation_used": state.repair_used,
            "repair_generation_attempted": state.repair_attempted,
            "repair_reason": state.repair_reason,
            "generation_metrics": state.generation_metrics,
            "state_phase_trace": state.phase_trace,
            "grounding_required": state.grounding_required,
            "grounding_reason": state.grounding_reason,
            "local_evidence_count": state.local_evidence_count,
            **quality_debug,
            "question_type": analysis.question_type,
            "comparison_targets": list(analysis.comparison_targets),
        }
        performance = {
            **quality_debug,
            "central_model_calls": state.model_calls,
            "central_tool_calls": state.tool_calls,
            "central_tool_calls_by_type": dict(state.tool_calls_by_name),
            "central_generation_ms": state.generation_ms,
            "central_tool_ms": state.tool_ms,
            "central_total_latency_ms": elapsed_ms,
            "central_input_tokens": state.input_tokens,
            "central_output_tokens": state.output_tokens,
            "central_external_results_count": state.external_evidence_count,
            "central_tool_schema_count": len(schemas),
            "central_tool_parse_failures": state.tool_parse_failures,
            "generation_metrics": state.generation_metrics,
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_generation_calls": 0,
        }
        return {
            "question": question,
            "answer": state.final_answer,
            "status": status,
            "source_ids": source_ids,
            "source_chunks": state.selected_sources,
            "model_source_ids": source_ids,
            "invalid_source_ids": state.invalid_source_ids,
            "unsupported_years": quality_debug["unsupported_years"],
            "unsupported_named_claims": quality_debug["unsupported_named_claims"],
            "retrieval": {
                "question": question,
                "final_context": state.selected_sources,
                "candidates20": state.retrieval_candidates,
                "target_specific_queries": state.target_specific_queries,
                "target_rankings": state.target_rankings,
                "target_retrieval_results": state.target_rankings,
                "comparison_balance": state.evidence_debug.get("comparison_balance", {}),
                "query_variants": state.retrieval_query_variants,
                "tool_trace": [f"central:{item['name']}" for item in state.tool_trace],
            },
            "analysis": {
                "question": question,
                "analytical": analysis.analytical,
                **analysis.telemetry(),
                "answer_quality_issues": quality_issues,
            },
            "tool_trace": [f"central:{item['name']}" for item in state.tool_trace],
            "central_debug": {
                **quality_debug,
                "retrieval_filter_events": state.retrieval_filter_events,
                "grounding_risk_checks": state.grounding_risk_checks,
                "tools": state.tool_trace,
                "allowed_tools": sorted(allowed_tools),
                "tool_schema_count": len(schemas),
                "tools_exposed_to_model": [item["function"]["name"] for item in schemas],
                "tool_parse_failures": state.tool_parse_failures,
                "malformed_tool_calls": state.malformed_tool_calls[:5],
                "question_type": analysis.question_type,
                "subject": analysis.subject,
                "comparison_targets": list(analysis.comparison_targets),
                "phase": state.phase.value,
                "phase_trace": state.phase_trace,
                "grounding_required": state.grounding_required,
                "grounding_reason": state.grounding_reason,
                "initial_grounding_coverage": state.initial_grounding_coverage,
                **runtime,
            },
            "agentic": True,
            "inference_mode": "central",
            "answer_provenance": provenance,
            "performance_debug": performance,
            "latency_sec": elapsed_ms / 1000,
            "total_latency_sec": elapsed_ms / 1000,
        }

    def _timeout_result(
        self,
        *,
        question: str,
        timeout_stage: str,
        started: float,
        analysis: Any,
    ) -> dict[str, Any]:
        elapsed_ms = (time.perf_counter() - started) * 1000
        runtime = self._runtime_snapshot()
        telemetry = current_request_telemetry()
        model_calls = telemetry.central_model_calls if telemetry is not None else 0
        provenance = {
            "mode": "central",
            "source": "central_timeout",
            "timeout_stage": timeout_stage,
            **runtime,
            "central_model_calls": model_calls,
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_generation_calls": 0,
            "total_llm_calls": model_calls,
        }
        return {
            "question": question,
            "answer": INSUFFICIENT_EVIDENCE_ANSWER,
            "status": "insufficient_evidence",
            "source_ids": [],
            "source_chunks": [],
            "retrieval": {"question": question, "final_context": [], "tool_trace": []},
            "analysis": {
                "question": question,
                "analytical": analysis.analytical,
                **analysis.telemetry(),
            },
            "tool_trace": [],
            "central_debug": {"tools": [], "timeout_stage": timeout_stage, **runtime},
            "agentic": True,
            "inference_mode": "central",
            "answer_provenance": provenance,
            "performance_debug": {
                "timeout_stage": timeout_stage,
                "central_model_calls": model_calls,
                "central_total_latency_ms": elapsed_ms,
                "research_generation_calls": 0,
                "evidence_generation_calls": 0,
                "history_generation_calls": 0,
            },
            "latency_sec": elapsed_ms / 1000,
            "total_latency_sec": elapsed_ms / 1000,
        }

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        question = str(kwargs.get("question") or "").strip()
        started = time.perf_counter()
        analysis = analyze_central_question(question)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._ensure_model_ready),
                timeout=self.config.model_load_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._timeout_result(
                question=question,
                timeout_stage="model_initialization",
                started=started,
                analysis=analysis,
            )
        progress: dict[str, Any] = {"timeout_stage": "agent_budget"}
        try:
            return await asyncio.wait_for(
                self._run(
                    question=question,
                    history=kwargs.get("history"),
                    owner_id=kwargs.get("owner_id"),
                    conversation_id=kwargs.get("conversation_id"),
                    request_id=kwargs.get("request_id"),
                    started=started,
                    progress=progress,
                ),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._timeout_result(
                question=question,
                timeout_stage=str(progress.get("timeout_stage") or "agent_budget"),
                started=started,
                analysis=analysis,
            )

    def chat(
        self,
        question: str,
        final_k: int | None = None,
        history: list[dict[str, str]] | None = None,
        owner_id: str | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del final_k
        return asyncio.run(self.run(
            question=question,
            history=history,
            owner_id=owner_id,
            conversation_id=conversation_id,
            request_id=request_id,
        ))
