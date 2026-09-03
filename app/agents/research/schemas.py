from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchResult(BaseModel):
    question: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[str] = Field(default_factory=list)
    is_ood: bool = False
    ood_reason: str = ""
    analysis: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)
