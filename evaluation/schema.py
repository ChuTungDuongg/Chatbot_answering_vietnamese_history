"""Versioned, JSON-only contracts; absent observations are unknown, never zero."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Text = Annotated[str, Field(min_length=1)]
Ratio = Annotated[float, Field(ge=0, le=1)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Question(Contract):
    id: Text
    question: Text
    category: Text
    expected_question_type: str | None = None
    expected_event: str | None = None
    expected_actors: list[str] | None = None
    expected_facets: list[str] | None = None
    expected_answer_depth: str | None = None
    expected_tool_behavior: dict[str, JsonValue] | None = None
    expected_source_ids: list[str] | None = None
    notes: str | None = None


class Annotation(Contract):
    """Optional review labels, with reviewer provenance. Scores use [0, 1]."""
    reviewer: Text
    method: Literal["human", "external_judge"]
    rubric_version: Text
    historical_correctness: Ratio | None = None
    completeness: Ratio | None = None
    relevance: Ratio | None = None
    neutrality: Ratio | None = None
    coherence: Ratio | None = None
    evidence_sufficiency_correct: bool | None = None
    unnecessary_tool_calls: Annotated[int, Field(ge=0)] | None = None
    partial_answer_correct: bool | None = None
    viewpoint_should_flag: bool | None = None
    # Relative to the paired BASE answer, never inferred from heuristic metrics.
    preference: Literal["win", "tie", "loss"] | None = None
    notes: str | None = None


class RunMetadata(Contract):
    schema_version: Literal[1] = 1
    run_id: Text
    variant: Literal["base", "adapted"]
    timestamp: datetime
    git_commit: Text
    model_id: Text
    model_revision: Text
    adapter_path: str | None = None
    adapter_sha256: str | None = None
    adapter_enabled: bool
    dataset_version: Text
    dataset_sha256: Text
    retrieval_index_sha256: Text
    prompt_sha256: Text
    generation_settings: dict[str, JsonValue]
    retrieval_settings: dict[str, JsonValue]
    tools: list[str]
    context_budgets: dict[str, JsonValue]
    host_config: dict[str, JsonValue]
    seed: int
    hardware: dict[str, JsonValue] | None = None
    hardware_class: str | None = None
    environment: dict[str, JsonValue]
    graph_name: str | None = None
    graph_version: str | None = None
    graph_topology_fingerprint: str | None = None

    @model_validator(mode="after")
    def adapter_contract(self):
        if self.variant == "base" and (self.adapter_enabled or self.adapter_path or self.adapter_sha256):
            raise ValueError("BASE must have no adapter")
        if self.variant == "adapted" and (not self.adapter_enabled or not self.adapter_path or not self.adapter_sha256):
            raise ValueError("ADAPTED requires an enabled, fingerprinted Central V2 adapter")
        return self


class EvaluationRecord(Contract):
    schema_version: Literal[1] = 1
    run_id: Text
    variant: Literal["base", "adapted"]
    question_id: Text
    question: Text
    category: Text
    parsed_semantics: dict[str, JsonValue] | None = None
    answer: str | None = None
    sources: list[dict[str, JsonValue]] | None = None
    selected_evidence: list[dict[str, JsonValue]] | None = None
    tool_trace: list[dict[str, JsonValue]] | None = None
    validation_issues: list[str] | None = None
    citations: dict[str, JsonValue] | None = None
    repairs: dict[str, JsonValue] | None = None
    usage: dict[str, JsonValue] | None = None
    status: str | None = None
    final_failure_reason: str | None = None
    adapter_configured: bool | None = None
    adapter_loaded: bool | None = None
    graph_name: str | None = None
    graph_version: str | None = None
    graph_topology_fingerprint: str | None = None
    graph_nodes_executed: list[str] | None = None
    graph_route: list[str] | None = None
    node_timings: dict[str, float] | None = None
    node_model_calls: dict[str, int] | None = None
    node_tool_calls: dict[str, int] | None = None
    graph_trace: list[dict[str, JsonValue]] | None = None
    # Derived trace signals have documented units/denominators in metrics/specs.py.
    signals: dict[str, bool | float | None] = Field(default_factory=dict)
    annotation: Annotation | None = None
    raw_result: dict[str, JsonValue] = Field(default_factory=dict)
