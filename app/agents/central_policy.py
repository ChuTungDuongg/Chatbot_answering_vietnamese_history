from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.agents.central_state import CentralAgentState


@dataclass(frozen=True)
class GroundingDecision:
    required: bool
    reason: str


class CentralRequestPolicy(Protocol):
    """Hook for a future domain guard in front of Central."""

    def grounding_for(self, question: str) -> GroundingDecision: ...


class HistoryGroundingPolicy:
    """Current policy: callers have already supplied an in-domain history QA request."""

    def grounding_for(self, question: str) -> GroundingDecision:
        del question
        return GroundingDecision(required=True, reason="in_domain_history_grounded_by_default")

    def evidence_is_sufficient(self, state: CentralAgentState) -> bool:
        targets = tuple(state.question_analysis.comparison_targets or ())
        if len(targets) >= 2:
            local_coverage = all(state.initial_grounding_coverage.get(target, 0) > 0 for target in targets[:2])
            return local_coverage or state.external_evidence_count >= 2
        return state.local_evidence_count > 0 or state.external_evidence_count > 0
