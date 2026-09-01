from __future__ import annotations

import contextvars
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class GenerationMetric:
    adapter: str
    input_tokens: int
    output_tokens: int
    max_new_tokens: int
    lock_wait_ms: float = 0.0
    adapter_switch_ms: float = 0.0
    generation_ms: float = 0.0
    decode_ms: float = 0.0
    tokens_per_sec: float = 0.0
    peak_allocated_gb: float | None = None


@dataclass
class RequestTelemetry:
    request_id: str
    inference_mode: str | None = None
    selected_inference_mode: str | None = None
    deployment_id: str | None = None
    gpu: str | None = None
    started: float = field(default_factory=time.perf_counter)
    cold_start_included: bool = False
    failed_stage: str | None = None
    failure_code: str | None = None
    research_attempts: int = 0
    research_steps: int = 0
    research_ms: float = 0.0
    research_json_repairs: int = 0
    research_prefetch_used: bool = False
    external_fallback_triggered: bool = False
    external_research_needed: bool = False
    external_research_available: bool = False
    external_research_reason: str | None = None
    external_research_skip_reason: str | None = None
    external_tools_called: list[str] = field(default_factory=list)
    external_results_count: int = 0
    tool_calls: int = 0
    tool_calls_by_type: dict[str, int] = field(default_factory=dict)
    central_tool_ms: float = 0.0
    central_external_results_count: int = 0
    domain_gate_result: str | None = None
    domain_gate_reason: str | None = None
    history_anchor: float | None = None
    ood_anchor: float | None = None
    domain_margin: float | None = None
    retrieval_skipped_due_to_ood: bool = False
    llm_calls_skipped_due_to_ood: bool = False
    retrieval_ms: float = 0.0
    wikipedia_calls: int = 0
    wikipedia_search_count: int = 0
    wikipedia_fetch_count: int = 0
    wikipedia_ms: float = 0.0
    generic_web_calls: int = 0
    generic_web_ms: float = 0.0
    evidence_attempts: int = 0
    evidence_ms: float = 0.0
    evidence_candidate_count: int = 0
    evidence_candidate_count_raw: int = 0
    evidence_candidate_count_model_visible: int = 0
    evidence_model_input_chars: int = 0
    evidence_model_input_tokens: int = 0
    evidence_dropped_for_budget_count: int = 0
    evidence_dropped_ids: list[str] = field(default_factory=list)
    evidence_source_kind_counts_raw: dict[str, int] = field(default_factory=dict)
    evidence_source_kind_counts_visible: dict[str, int] = field(default_factory=dict)
    external_evidence_collected_count: int = 0
    external_evidence_model_visible_count: int = 0
    external_evidence_selected_count: int = 0
    external_evidence_rejected_count: int = 0
    external_evidence_rejection_reasons: dict[str, str] = field(default_factory=dict)
    evidence_selected_count: int = 0
    evidence_relevance_guard_triggered: bool = False
    evidence_reconsideration_used: bool = False
    evidence_coverage_guard_triggered: bool = False
    evidence_recovery_used: bool = False
    evidence_repair_used: bool = False
    evidence_rebucket_attempted: bool = False
    evidence_rebucket_succeeded: bool = False
    evidence_rebucket_moved_claim_count: int = 0
    evidence_rebucket_destination_ids: list[str] = field(default_factory=list)
    evidence_final_validation_result: str | None = None
    evidence_pruned_claim_count: int = 0
    evidence_supplemented_count: int = 0
    evidence_supplemented_ids: list[str] = field(default_factory=list)
    comparison_targets: list[str] = field(default_factory=list)
    target_a_candidate_count: int = 0
    target_b_candidate_count: int = 0
    target_a_model_visible_count: int = 0
    target_b_model_visible_count: int = 0
    comparison_target_coverage: dict[str, bool] = field(default_factory=dict)
    candidate_roles: dict[str, str] = field(default_factory=dict)
    direct_subject_scores: dict[str, float] = field(default_factory=dict)
    affiliation_constraint_pass: dict[str, bool] = field(default_factory=dict)
    broad_summary_facets_requested: list[str] = field(default_factory=list)
    broad_summary_facets_covered: list[str] = field(default_factory=list)
    evidence_first_pass_latency_ms: float = 0.0
    evidence_guard_latency_ms: float = 0.0
    evidence_reconsideration_latency_ms: float = 0.0
    duplicate_inspect_skipped: bool = False
    wikipedia_query: str | None = None
    wikipedia_candidate_titles: list[str] = field(default_factory=list)
    wikipedia_selected_title: str | None = None
    wikipedia_year_conflict_rejections: int = 0
    history_ms: float = 0.0
    history_generation_calls: int = 0
    history_retry_used: bool = False
    history_retry_reason: str | None = None
    history_first_answer_chars: int = 0
    history_first_answer_words: int = 0
    history_final_answer_chars: int = 0
    history_final_answer_words: int = 0
    history_first_quality_issues: list[str] = field(default_factory=list)
    history_final_quality_issues: list[str] = field(default_factory=list)
    history_first_latency_ms: float = 0.0
    history_retry_latency_ms: float = 0.0
    history_total_latency_ms: float = 0.0
    history_input_evidence_count: int = 0
    history_input_claim_count: int = 0
    history_input_source_kind_counts: dict[str, int] = field(default_factory=dict)
    generation_metrics: list[GenerationMetric] = field(default_factory=list)

    def next_call_index(self) -> int:
        return len(self.generation_metrics) + 1

    def add_generation(self, metric: GenerationMetric) -> None:
        self.generation_metrics.append(metric)

    @property
    def total_llm_calls(self) -> int:
        return len(self.generation_metrics)

    @property
    def research_llm_calls(self) -> int:
        return sum(1 for item in self.generation_metrics if item.adapter == "research")

    @property
    def evidence_generation_calls(self) -> int:
        return sum(1 for item in self.generation_metrics if item.adapter == "evidence")

    @property
    def central_model_calls(self) -> int:
        return sum(1 for item in self.generation_metrics if item.adapter == "central")

    @property
    def central_generation_ms(self) -> float:
        return sum(item.generation_ms for item in self.generation_metrics if item.adapter == "central")

    @property
    def central_input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.generation_metrics if item.adapter == "central")

    @property
    def central_output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.generation_metrics if item.adapter == "central")

    @property
    def total_input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.generation_metrics)

    @property
    def total_output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.generation_metrics)

    @property
    def average_generation_tokens_per_sec(self) -> float:
        rates = [item.tokens_per_sec for item in self.generation_metrics if item.tokens_per_sec > 0]
        return float(statistics.mean(rates)) if rates else 0.0

    @property
    def peak_allocated_vram_gb(self) -> float | None:
        peaks = [item.peak_allocated_gb for item in self.generation_metrics if item.peak_allocated_gb is not None]
        return max(peaks) if peaks else None

    def summary(self, *, result: str) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "inference_mode": self.inference_mode,
            "selected_inference_mode": self.selected_inference_mode or self.inference_mode,
            "deployment_id": self.deployment_id,
            "gpu": self.gpu,
            "result": result,
            "failed_stage": self.failed_stage,
            "failure_code": self.failure_code,
            "cold_start_included": self.cold_start_included,
            "total_ms": (time.perf_counter() - self.started) * 1000,
            "research_attempts": self.research_attempts,
            "research_steps": self.research_steps,
            "research_ms": self.research_ms,
            "research_llm_calls": self.research_llm_calls,
            "research_json_repairs": self.research_json_repairs,
            "research_prefetch_used": self.research_prefetch_used,
            "research_generation_calls": self.research_llm_calls,
            "external_fallback_triggered": self.external_fallback_triggered,
            "external_research_needed": self.external_research_needed,
            "external_research_available": self.external_research_available,
            "external_research_reason": self.external_research_reason,
            "external_research_skip_reason": self.external_research_skip_reason,
            "external_tools_called": self.external_tools_called,
            "external_results_count": self.external_results_count,
            "tool_calls": self.tool_calls,
            "tool_calls_by_type": self.tool_calls_by_type,
            "central_model_calls": self.central_model_calls,
            "central_tool_calls": self.tool_calls if self.inference_mode == "central" else 0,
            "central_tool_calls_by_type": self.tool_calls_by_type if self.inference_mode == "central" else {},
            "central_generation_ms": self.central_generation_ms,
            "central_tool_ms": self.central_tool_ms,
            "central_total_latency_ms": (time.perf_counter() - self.started) * 1000 if self.inference_mode == "central" else 0.0,
            "central_input_tokens": self.central_input_tokens,
            "central_output_tokens": self.central_output_tokens,
            "central_external_results_count": self.central_external_results_count,
            "domain_gate_result": self.domain_gate_result,
            "domain_gate_reason": self.domain_gate_reason,
            "history_anchor": self.history_anchor,
            "ood_anchor": self.ood_anchor,
            "domain_margin": self.domain_margin,
            "retrieval_skipped_due_to_ood": self.retrieval_skipped_due_to_ood,
            "llm_calls_skipped_due_to_ood": self.llm_calls_skipped_due_to_ood,
            "retrieval_ms": self.retrieval_ms,
            "wikipedia_calls": self.wikipedia_calls,
            "wikipedia_search_count": self.wikipedia_search_count,
            "wikipedia_fetch_count": self.wikipedia_fetch_count,
            "wikipedia_ms": self.wikipedia_ms,
            "generic_web_calls": self.generic_web_calls,
            "generic_web_ms": self.generic_web_ms,
            "evidence_attempts": self.evidence_attempts,
            "evidence_ms": self.evidence_ms,
            "evidence_candidate_count": self.evidence_candidate_count,
            "evidence_candidate_count_raw": self.evidence_candidate_count_raw,
            "evidence_candidate_count_model_visible": self.evidence_candidate_count_model_visible,
            "evidence_model_input_chars": self.evidence_model_input_chars,
            "evidence_model_input_tokens": self.evidence_model_input_tokens,
            "evidence_dropped_for_budget_count": self.evidence_dropped_for_budget_count,
            "evidence_dropped_ids": self.evidence_dropped_ids,
            "evidence_source_kind_counts_raw": self.evidence_source_kind_counts_raw,
            "evidence_source_kind_counts_visible": self.evidence_source_kind_counts_visible,
            "external_evidence_collected_count": self.external_evidence_collected_count,
            "external_evidence_model_visible_count": self.external_evidence_model_visible_count,
            "external_evidence_selected_count": self.external_evidence_selected_count,
            "external_evidence_rejected_count": self.external_evidence_rejected_count,
            "external_evidence_rejection_reasons": self.external_evidence_rejection_reasons,
            "evidence_selected_count": self.evidence_selected_count,
            "evidence_generation_calls": self.evidence_generation_calls,
            "evidence_relevance_guard_triggered": self.evidence_relevance_guard_triggered,
            "evidence_reconsideration_used": self.evidence_reconsideration_used,
            "evidence_coverage_guard_triggered": self.evidence_coverage_guard_triggered,
            "evidence_recovery_used": self.evidence_recovery_used,
            "evidence_repair_used": self.evidence_repair_used,
            "evidence_rebucket_attempted": self.evidence_rebucket_attempted,
            "evidence_rebucket_succeeded": self.evidence_rebucket_succeeded,
            "evidence_rebucket_moved_claim_count": self.evidence_rebucket_moved_claim_count,
            "evidence_rebucket_destination_ids": self.evidence_rebucket_destination_ids,
            "evidence_final_validation_result": self.evidence_final_validation_result,
            "evidence_pruned_claim_count": self.evidence_pruned_claim_count,
            "evidence_supplemented_count": self.evidence_supplemented_count,
            "evidence_supplemented_ids": self.evidence_supplemented_ids,
            "comparison_targets": self.comparison_targets,
            "target_a_candidate_count": self.target_a_candidate_count,
            "target_b_candidate_count": self.target_b_candidate_count,
            "target_a_model_visible_count": self.target_a_model_visible_count,
            "target_b_model_visible_count": self.target_b_model_visible_count,
            "comparison_target_coverage": self.comparison_target_coverage,
            "candidate_roles": self.candidate_roles,
            "direct_subject_scores": self.direct_subject_scores,
            "affiliation_constraint_pass": self.affiliation_constraint_pass,
            "broad_summary_facets_requested": self.broad_summary_facets_requested,
            "broad_summary_facets_covered": self.broad_summary_facets_covered,
            "evidence_first_pass_latency_ms": self.evidence_first_pass_latency_ms,
            "evidence_guard_latency_ms": self.evidence_guard_latency_ms,
            "evidence_reconsideration_latency_ms": self.evidence_reconsideration_latency_ms,
            "duplicate_inspect_skipped": self.duplicate_inspect_skipped,
            "wikipedia_query": self.wikipedia_query,
            "wikipedia_candidate_titles": self.wikipedia_candidate_titles,
            "wikipedia_selected_title": self.wikipedia_selected_title,
            "wikipedia_year_conflict_rejections": self.wikipedia_year_conflict_rejections,
            "history_ms": self.history_ms,
            "history_generation_calls": self.history_generation_calls,
            "history_retry_used": self.history_retry_used,
            "history_retry_reason": self.history_retry_reason,
            "history_first_answer_chars": self.history_first_answer_chars,
            "history_first_answer_words": self.history_first_answer_words,
            "history_final_answer_chars": self.history_final_answer_chars,
            "history_final_answer_words": self.history_final_answer_words,
            "history_first_quality_issues": self.history_first_quality_issues,
            "history_final_quality_issues": self.history_final_quality_issues,
            "history_first_latency_ms": self.history_first_latency_ms,
            "history_retry_latency_ms": self.history_retry_latency_ms,
            "history_total_latency_ms": self.history_total_latency_ms,
            "history_input_evidence_count": self.history_input_evidence_count,
            "history_input_claim_count": self.history_input_claim_count,
            "history_input_source_kind_counts": self.history_input_source_kind_counts,
            "research_latency_ms": self.research_ms,
            "history_latency_ms": self.history_ms,
            "total_latency_ms": (time.perf_counter() - self.started) * 1000,
            "total_llm_calls": self.total_llm_calls,
            "per_role_latency_ms": {
                "research": sum(item.generation_ms for item in self.generation_metrics if item.adapter == "research"),
                "evidence": sum(item.generation_ms for item in self.generation_metrics if item.adapter == "evidence"),
                "history": sum(item.generation_ms for item in self.generation_metrics if item.adapter == "history"),
                "central": self.central_generation_ms,
            },
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "average_generation_tokens_per_sec": self.average_generation_tokens_per_sec,
            "peak_allocated_vram_gb": self.peak_allocated_vram_gb,
        }


_current: contextvars.ContextVar[RequestTelemetry | None] = contextvars.ContextVar(
    "request_telemetry",
    default=None,
)


def current_request_telemetry() -> RequestTelemetry | None:
    return _current.get()


def set_request_telemetry(telemetry: RequestTelemetry | None) -> contextvars.Token:
    return _current.set(telemetry)


def reset_request_telemetry(token: contextvars.Token) -> None:
    _current.reset(token)


def log_event(name: str, **fields: Any) -> None:
    logger.info(name, extra={"event": name, **fields})
