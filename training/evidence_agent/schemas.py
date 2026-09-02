from app.agents.evidence.schemas import EvidenceAgentRequest, EvidenceCandidate, EvidenceModelOutput, SelectedEvidence


# Compatibility alias. Training and runtime intentionally validate the same model output.
EvidenceCritiqueOutput = EvidenceModelOutput

__all__ = [
    "EvidenceAgentRequest",
    "EvidenceCandidate",
    "EvidenceCritiqueOutput",
    "EvidenceModelOutput",
    "SelectedEvidence",
]
