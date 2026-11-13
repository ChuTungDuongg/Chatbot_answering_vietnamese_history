from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_ID = "vietnamese-history-eval-v1"
QUOTAS = {
    "Factual QA": 6,
    "Comparative Analysis": 12,
    "Ranking / Historical Judgment": 8,
    "Multi-hop / Causal Reasoning": 8,
    "Ambiguous / Controversial": 4,
    "Out-of-domain / Uncertainty": 2,
}
Category = Literal[
    "Factual QA",
    "Comparative Analysis",
    "Ranking / Historical Judgment",
    "Multi-hop / Causal Reasoning",
    "Ambiguous / Controversial",
    "Out-of-domain / Uncertainty",
]


def now():
    return datetime.now(timezone.utc).isoformat()


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    dataset_id: Literal["vietnamese-history-eval-v1"] = DATASET_ID
    question: str = ""
    category: Category = "Factual QA"
    difficulty: Literal["easy", "medium", "hard", "expert"] = "medium"
    tags: list[str] = Field(default_factory=list)
    smoke: bool = False
    status: Literal["draft", "needs_review", "approved"] = "draft"
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    version: int = Field(default=1, ge=1)
    reference_answer: str = ""
    expected_facts: list[str] = Field(default_factory=list)
    must_cover_points: list[str] = Field(default_factory=list)
    reference_sources: list[str] = Field(default_factory=list)
    acceptable_alternatives: list[str] = Field(default_factory=list)
    common_mistakes: list[str] = Field(default_factory=list)
    uncertainty_notes: str = ""
    annotator_notes: str = ""
    expected_reasoning_depth: str = ""
    required_evidence_coverage: str = ""
    historical_scope: str = ""
    question_type: str = ""
    ranking_criteria: list[str] = Field(default_factory=list)
    rubric_overrides: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def approval_rules(self):
        if self.status == "approved":
            missing = [
                name
                for name in (
                    "question",
                    "reference_answer",
                    "expected_facts",
                    "must_cover_points",
                    "reference_sources",
                )
                if not getattr(self, name)
                or (isinstance(getattr(self, name), str) and not getattr(self, name).strip())
                or (isinstance(getattr(self, name), list) and not any(x.strip() for x in getattr(self, name)))
            ]
            if self.category == "Ranking / Historical Judgment" and not any(
                x.strip() for x in self.ranking_criteria
            ):
                missing.append("ranking_criteria")
            if self.category == "Ambiguous / Controversial" and not self.uncertainty_notes.strip():
                missing.append("uncertainty_notes")
            if missing:
                raise ValueError("Approval requires: " + ", ".join(missing))
        return self


class ResponseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    answer: str = Field(min_length=1)
    provider: str = ""
    model: str = ""
    chat_mode: str = ""
    reasoning_effort: str = ""
    retrieval_config: str = ""
    retrieved_evidence: str = ""


class Scores(BaseModel):
    accuracy: int = Field(ge=0, le=5)
    coverage: int = Field(ge=0, le=5)
    reasoning: int = Field(ge=0, le=5)
    evidence: int = Field(ge=0, le=5)
    clarity: int = Field(ge=0, le=5)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_id: str
    scores: Scores
    decision: Literal["accept", "reject", "needs_review"]
    error_tags: list[str] = Field(default_factory=list)
    notes: str = ""
    strengths: str = ""
    weaknesses: str = ""
    reviewer: str = ""
    diagnosis_tags: list[str] = Field(default_factory=list)
    point_evidence: dict[str, str] = Field(default_factory=dict)

    def weighted_score(self):
        s = self.scores
        return round(
            (s.accuracy * 30 + s.coverage * 20 + s.reasoning * 20 + s.evidence * 20 + s.clarity * 10) / 5, 2
        )
