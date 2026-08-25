from __future__ import annotations

import pytest

from app.agents.evidence_agent import EvidenceCriticAgent
from app.tools.evidence_tools import SessionEvidenceStore


def test_evidence_agent_selects_contexts():
    agent = EvidenceCriticAgent(max_contexts=2)
    critique, contexts = agent.compress(
        "Bach Dang?",
        [
            {"chunk_id": "c1", "text": "Tran Bach Dang nam 938"},
            {"chunk_id": "c2", "text": ""},
            {"chunk_id": "c3", "text": "Ngo Quyen"},
        ],
        final_k=2,
    )
    assert critique.selected_ids == ["c1", "c3"]
    assert "c2" in critique.rejected_ids
    assert len(contexts) == 2

