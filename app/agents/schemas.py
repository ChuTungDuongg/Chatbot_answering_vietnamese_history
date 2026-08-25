from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class SelectedEvidence(BaseModel):
    evidence_id: str
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    claims: list[str] = Field(default_factory=list)
    compressed_text: str = ""


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

    @model_validator(mode="after")
    def validate_disjoint_ids(self):
        if set(self.selected_ids) & set(self.rejected_ids):
            raise ValueError("selected_ids and rejected_ids must be disjoint")
        structured_ids = [item.evidence_id for item in self.selected_evidence]
        if structured_ids and structured_ids != self.selected_ids:
            raise ValueError("selected_evidence IDs must match selected_ids")
        return self
