"""Compatibility exports; each prompt has one owning role."""
from app.agents.research.policy import RESEARCH_AGENT_SYSTEM
from app.agents.evidence.prompts import EVIDENCE_AGENT_SYSTEM, EVIDENCE_CRITIC_SYSTEM

__all__ = ["RESEARCH_AGENT_SYSTEM", "EVIDENCE_AGENT_SYSTEM", "EVIDENCE_CRITIC_SYSTEM"]
