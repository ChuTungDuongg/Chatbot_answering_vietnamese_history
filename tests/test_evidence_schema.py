from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.evidence.schemas import EvidenceCritique
from training.evidence_agent.schemas import EvidenceCritiqueOutput


def test_evidence_schema_rejects_overlap():
    with pytest.raises(ValidationError):
        EvidenceCritique(selected_ids=["c1"], rejected_ids=["c1"])


def test_training_schema_rejects_duplicate_selected_ids():
    with pytest.raises(ValidationError):
        EvidenceCritiqueOutput(
            status="sufficient",
            selected_evidence=[
                {
                    "evidence_id": "c1",
                    "relevance": 1.0,
                    "claims": ["Một fact được evidence hỗ trợ."],
                    "compressed_text": "Một fact được evidence hỗ trợ.",
                },
                {
                    "evidence_id": "c1",
                    "relevance": 1.0,
                    "claims": ["Một fact được evidence hỗ trợ."],
                    "compressed_text": "Một fact được evidence hỗ trợ.",
                },
            ],
            conflicts=[],
            missing_information=[],
            summary="Evidence đủ.",
        )



