from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CentralPhase(str, Enum):
    PREPARE = "prepare"
    INITIAL_GROUNDING = "initial_grounding"
    ACTION = "action"
    TOOL_EXECUTION = "tool_execution"
    SYNTHESIS = "synthesis"
    CITATION_REPAIR = "citation_repair"
    QUALITY_REPAIR = "quality_repair"
    FINAL = "final"


_ALLOWED_TRANSITIONS = {
    CentralPhase.PREPARE: {CentralPhase.INITIAL_GROUNDING},
    CentralPhase.INITIAL_GROUNDING: {CentralPhase.ACTION, CentralPhase.SYNTHESIS, CentralPhase.FINAL},
    CentralPhase.ACTION: {CentralPhase.TOOL_EXECUTION, CentralPhase.SYNTHESIS, CentralPhase.FINAL},
    CentralPhase.TOOL_EXECUTION: {CentralPhase.ACTION, CentralPhase.SYNTHESIS, CentralPhase.FINAL},
    CentralPhase.SYNTHESIS: {CentralPhase.CITATION_REPAIR, CentralPhase.QUALITY_REPAIR, CentralPhase.FINAL},
    CentralPhase.CITATION_REPAIR: {CentralPhase.QUALITY_REPAIR, CentralPhase.FINAL},
    CentralPhase.QUALITY_REPAIR: {CentralPhase.FINAL},
    CentralPhase.FINAL: set(),
}


@dataclass
class CentralAgentState:
    question: str
    history: list[dict[str, str]]
    messages: list[dict[str, Any]]
    allowed_tools: set[str]
    tool_schemas: list[dict[str, Any]]
    question_analysis: Any
    grounding_required: bool
    grounding_reason: str
    deadline_monotonic: float
    phase: CentralPhase = CentralPhase.PREPARE
    phase_trace: list[str] = field(default_factory=lambda: [CentralPhase.PREPARE.value])
    source_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_sources: list[dict[str, Any]] = field(default_factory=list)
    wikipedia_search_results: list[dict[str, Any]] = field(default_factory=list)
    fetched_wikipedia_targets: set[str] = field(default_factory=set)
    target_specific_queries: dict[str, list[str]] = field(default_factory=dict)
    retrieval_query_variants: list[str] = field(default_factory=list)
    target_rankings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    tool_observations: dict[str, str] = field(default_factory=dict)
    tool_results: dict[str, tuple[Any, str | None, int | None]] = field(default_factory=dict)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    executed_tool_signatures: set[str] = field(default_factory=set)
    initial_grounding_coverage: dict[str, int] = field(default_factory=dict)
    local_evidence_count: int = 0
    external_evidence_count: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    tool_calls_by_name: Counter[str] = field(default_factory=Counter)
    generation_metrics: list[dict[str, Any]] = field(default_factory=list)
    generation_ms: float = 0.0
    tool_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    observation_chars: int = 0
    tool_parse_failures: int = 0
    malformed_tool_calls: list[str] = field(default_factory=list)
    final_answer: str = ""
    invalid_source_ids: list[str] = field(default_factory=list)
    repair_attempted: bool = False
    repair_used: bool = False
    repair_reason: str | None = None
    repair_avoided_reason: str | None = None
    repair_budget: int | None = None
    retrieval_candidates: list[dict[str, Any]] = field(default_factory=list)
    retrieval_filter_events: list[dict[str, Any]] = field(default_factory=list)
    evidence_debug: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=lambda: {
        "model_was_cold": False, "model_load_wait_ms": 0.0,
        "model_load_overlap_enabled": False, "model_load_overlap_ms_saved_estimate": 0.0,
        "retrieval_queries_planned": [], "retrieval_queries_executed": [], "retrieval_queries_skipped": [],
        "retrieval_query_skip_reasons": {}, "retrieval_wall_ms": 0.0,
        "citation_normalization_used": False, "citation_alignment_used": False,
        "citation_alignment_ms": 0.0, "citation_alignment_success": False,
        "citation_alignment_confidence_by_paragraph": {}, "citation_repair_used": False,
        "citation_repair_tokens": 0, "citation_repair_ms": 0.0,
        "full_quality_repair_used": False, "final_failure_reason": None,
    })
    grounding_risk_checks: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, next_phase: CentralPhase) -> None:
        if next_phase not in _ALLOWED_TRANSITIONS[self.phase]:
            raise RuntimeError(f"invalid Central phase transition: {self.phase.value} -> {next_phase.value}")
        self.phase = next_phase
        self.phase_trace.append(next_phase.value)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_monotonic - time.monotonic())
