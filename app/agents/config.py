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
    """Latency and evidence bounds for the standalone Central model loop."""

    max_steps: int = 3
    hard_max_steps: int = 3
    max_new_tokens: int = 1536
    max_tool_results: int = 6
    observation_char_budget: int = 12_000
    timeout_seconds: float = 120.0
    enable_history: bool = True
    enable_documents: bool = True
    enable_wikipedia: bool = True
    enable_web: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= self.hard_max_steps <= 3:
            raise ValueError("Central max_steps must be between 1 and hard_max_steps (maximum 3).")
        if self.max_new_tokens < 256:
            raise ValueError("Central max_new_tokens must be at least 256.")
        if not 1 <= self.max_tool_results <= 10:
            raise ValueError("Central max_tool_results must be between 1 and 10.")
        if self.observation_char_budget < 1_000:
            raise ValueError("Central observation_char_budget must be at least 1000 characters.")
        if self.timeout_seconds <= 0:
            raise ValueError("Central timeout_seconds must be positive.")
