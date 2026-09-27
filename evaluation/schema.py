"""Versioned question labels for model-free baseline evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: str = Field(min_length=1)
    gold_answer: str | None = None
    relevant_chunk_ids: list[str] | None = None
    relevant_source_ids: list[str] | None = None
    gold_citation_source_ids: list[str] | None = None
    required_facts: list[str] | None = None
    factual_paragraph_indices: list[int] | None = None
    answerable: bool | None = None
    in_domain: bool | None = None
    notes: str = ""
    # Optional benchmark provenance. Existing unlabeled fixtures remain valid.
    difficulty: Literal["easy", "medium", "hard"] | None = None
    question_type: str | None = None
    benchmark_split: Literal["dev", "test"] | None = None
    review_status: Literal["accepted"] | None = None
    retrieval_reviewed: bool | None = None
    answer_reviewed: bool | None = None
    citation_reviewed: bool | None = None
    reviewed_at: str | None = None
    deep_reviewed_at: str | None = None
    review_version: int | None = Field(default=None, ge=1)
    review_policy_version: int | None = Field(default=None, ge=1)
    reviewer_id: str | None = None
    # Provenance makes automatic evaluation labels distinguishable from human gold.
    annotation_origin: Literal["automatic", "human"] | None = None
    annotation_status: Literal["draft", "auto_reviewed", "needs_human_review", "human_accepted"] | None = None
    confidence: Literal["high", "medium", "low"] | None = None


def load_questions(path: str | Path) -> list[Question]:
    questions: list[Question] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    questions.append(Question.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"invalid question at line {line_number}: {exc}") from exc
    ids = [item.id for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question IDs")
    return questions
