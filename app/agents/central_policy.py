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
        # Computed from the selected, bounded packet, never from raw tool counts.
        return bool(state.evidence_debug.get("evidence_sufficient"))
