from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceChunk(BaseModel):
    chunk_id: str
    title: str | None = None
    text: str = ""
    source_kind: str = "history"
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchResult(BaseModel):
    question: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[str] = Field(default_factory=list)
    is_ood: bool = False
    ood_reason: str = ""
    analysis: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


class SelectedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    claims: list[str] = Field(default_factory=list)
    compressed_text: str = ""


class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    source_type: str = "local"
    title: str | None = None
    url: str | None = None
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    retrieval_score: float | None = None


class EvidenceAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    max_selected: int = Field(default=8, ge=1)
    evidence: list[EvidenceCandidate]

    @model_validator(mode="after")
    def validate_candidate_ids(self):
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence candidate IDs must be unique")
        return self


class EvidenceModelOutput(BaseModel):
    """Canonical semantic JSON emitted by both training targets and runtime model."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["sufficient", "insufficient", "conflicting"]
    selected_evidence: list[SelectedEvidence]
    conflicts: list[str]
    missing_information: list[str]
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_semantics(self):
        selected_ids = [item.evidence_id for item in self.selected_evidence]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected_evidence contains duplicate evidence IDs")
        if any(not item.compressed_text.strip() for item in self.selected_evidence):
            raise ValueError("selected evidence requires non-empty compressed_text")
        if any(not item.claims or any(not claim.strip() for claim in item.claims) for item in self.selected_evidence):
            raise ValueError("selected evidence requires non-empty grounded claims")
        if self.status == "sufficient" and not self.selected_evidence:
            raise ValueError("sufficient output requires selected evidence")
        if self.status == "sufficient" and self.missing_information:
            raise ValueError("sufficient output must not contain missing_information")
        if self.status == "conflicting" and not self.conflicts:
            raise ValueError("conflicting output requires a non-empty conflicts list")
        if self.status != "conflicting" and self.conflicts:
            raise ValueError("conflicts are only valid when status is conflicting")
        if self.status == "insufficient" and not self.missing_information:
            raise ValueError("insufficient output requires missing_information")
        return self


class EvidenceCritique(BaseModel):
    status: Literal["sufficient", "insufficient", "conflicting"] = "insufficient"
    selected_evidence: list[SelectedEvidence] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)
    compressed_context: str = ""
    conflicts: list[str] = Field(default_factory=list)
    sufficient: bool = False
    warnings: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    summary: str = ""
    model_input_evidence: list[dict[str, Any]] = Field(default_factory=list)
    raw_candidate_count: int = 0
    model_visible_candidate_count: int = 0
    dropped_for_budget_count: int = 0
    dropped_ids: list[str] = Field(default_factory=list)
    dropped_reasons: dict[str, str] = Field(default_factory=dict)
    source_kind_counts_raw: dict[str, int] = Field(default_factory=dict)
    source_kind_counts_visible: dict[str, int] = Field(default_factory=dict)
    question_type: str = "general"
    first_model_output: dict[str, Any] | None = None
    first_validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    final_validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    semantic_guard_findings: dict[str, Any] = Field(default_factory=dict)
    comparison_targets: list[str] = Field(default_factory=list)
    target_a_candidate_count: int = 0
    target_b_candidate_count: int = 0
    target_a_model_visible_count: int = 0
    target_b_model_visible_count: int = 0
    comparison_target_coverage: dict[str, bool] = Field(default_factory=dict)
    generation_calls: int = 0
    repair_used: bool = False
    repair_path: str | None = None

    @model_validator(mode="after")
    def validate_disjoint_ids(self):
        if set(self.selected_ids) & set(self.rejected_ids):
            raise ValueError("selected_ids and rejected_ids must be disjoint")
        structured_ids = [item.evidence_id for item in self.selected_evidence]
        if structured_ids != self.selected_ids:
            raise ValueError("selected_evidence IDs must match selected_ids")
        return self
