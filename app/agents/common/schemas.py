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
