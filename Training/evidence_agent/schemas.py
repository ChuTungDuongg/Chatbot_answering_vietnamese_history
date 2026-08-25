from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceCandidate(BaseModel):
    chunk_id: str
    title: str | None = None
    text: str
    score: float | None = None


class SelectedEvidence(BaseModel):
    evidence_id: str
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    claims: list[str] = Field(default_factory=list)
    compressed_text: str = ""


class EvidenceCritiqueOutput(BaseModel):
    status: Literal["sufficient", "insufficient", "conflicting"] = "insufficient"
    selected_evidence: list[SelectedEvidence] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)
    compressed_context: str = ""
    conflicts: list[str] = Field(default_factory=list)
    sufficient: bool = False
    missing_information: list[str] = Field(default_factory=list)
    summary: str = ""

    @model_validator(mode="after")
    def no_duplicate_selected_ids(self):
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("selected_ids contains duplicates")
        if set(self.selected_ids) & set(self.rejected_ids):
            raise ValueError("selected_ids and rejected_ids must be disjoint")
        return self
