"""Prepared final-generation input for both inference modes."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedAnswer:
    messages: list[dict[str, str]]
    contexts: list[dict[str, Any]] = field(default_factory=list)
    retrieval: dict[str, Any] = field(default_factory=dict)
    retrieval_ms: float | None = None
    prompt_build_ms: float | None = None
    model_calls_before_final: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_parse_failures: int = 0
    action_rounds: int = 0
