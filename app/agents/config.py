from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Centralized, bounded application-level agent controls."""

    max_steps: int = 6
    max_tool_results: int = 10
    observation_char_budget: int = 24_000
    timeout_seconds: float = 120.0
    enable_web: bool = True
    enable_wikipedia: bool = True
    enable_document_search: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= 12:
            raise ValueError("Agent max_steps must be between 1 and 12.")
        if not 1 <= self.max_tool_results <= 50:
            raise ValueError("Agent max_tool_results must be between 1 and 50.")
        if self.observation_char_budget < 1_000:
            raise ValueError("Agent observation_char_budget must be at least 1000 characters.")
        if self.timeout_seconds <= 0:
            raise ValueError("Agent timeout_seconds must be positive.")


@dataclass(frozen=True)
class CentralAgentConfig:
    """Latency, generation, and evidence bounds for Central V2."""

    max_action_rounds: int = 2
    repair_max_generations: int = 1
    action_max_new_tokens: int = 256
    final_max_new_tokens: int = 1536
    repair_max_new_tokens: int = 1024
    repair_min_new_tokens: int = 192
    repair_token_margin: int = 96
    biography_max_sources: int = 4
    biography_min_exact_hits: int = 2
    reranker_tail_gap_ratio: float = 0.75
    reranker_score_mode: str = "raw"
    reranker_score_floor: float | None = None
    reranker_strong_score: float = 0.5
    max_tool_results: int = 6
    observation_char_budget: int = 12_000
    timeout_seconds: float = 180.0
    model_load_timeout_seconds: float = 300.0
    tool_timeout_seconds: float = 30.0
    enable_history: bool = True
    enable_documents: bool = True
    enable_wikipedia: bool = True
    enable_web: bool = True
    web_search_provider: str = "local-only"
    # Deprecated constructor aliases retained for older callers/tests only.
    max_steps: int | None = None
    hard_max_steps: int = 4
    max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps is not None:
            object.__setattr__(self, "max_action_rounds", max(0, min(2, self.max_steps - 1)))
        if self.max_new_tokens is not None:
            object.__setattr__(self, "final_max_new_tokens", self.max_new_tokens)
        if not 0 <= self.max_action_rounds <= 4:
            raise ValueError("Central max_action_rounds must be between 0 and 4.")
        if not 0 <= self.repair_max_generations <= 1:
            raise ValueError("Central repair_max_generations must be 0 or 1.")
        if self.action_max_new_tokens < 32:
            raise ValueError("Central action_max_new_tokens must be at least 32.")
        if self.final_max_new_tokens < 128 or self.repair_max_new_tokens < 128:
            raise ValueError("Central final/repair token budgets must be at least 128.")
        if self.repair_min_new_tokens < 1 or self.repair_token_margin < 0:
            raise ValueError("Central repair minimum must be positive and margin nonnegative.")
        if not 1 <= self.biography_max_sources <= 10 or not 1 <= self.biography_min_exact_hits <= 10:
            raise ValueError("Central biography evidence bounds must be between 1 and 10.")
        if not 0.5 < self.reranker_tail_gap_ratio <= 1:
            raise ValueError("Central reranker tail gap ratio must be in (0.5, 1].")
        if self.reranker_score_mode not in {"raw", "probability"}:
            raise ValueError("Central reranker score mode must be raw or probability.")
        if self.reranker_score_floor is not None and not 0 <= self.reranker_score_floor <= 1:
            raise ValueError("Central reranker probability floor must be between 0 and 1.")
        if not 0 <= self.reranker_strong_score <= 1:
            raise ValueError("Central reranker strong probability score must be between 0 and 1.")
        if not 1 <= self.max_tool_results <= 10:
            raise ValueError("Central max_tool_results must be between 1 and 10.")
        if self.observation_char_budget < 1_000:
            raise ValueError("Central observation_char_budget must be at least 1000 characters.")
        if self.timeout_seconds <= 0:
            raise ValueError("Central timeout_seconds must be positive.")
        if self.model_load_timeout_seconds <= 0:
            raise ValueError("Central model_load_timeout_seconds must be positive.")
        if self.tool_timeout_seconds <= 0:
            raise ValueError("Central tool_timeout_seconds must be positive.")
