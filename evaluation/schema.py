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
