from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.schemas import EvidenceCritique
from training.evidence_agent.schemas import EvidenceCritiqueOutput


def test_evidence_schema_rejects_overlap():
    with pytest.raises(ValidationError):
        EvidenceCritique(selected_ids=["c1"], rejected_ids=["c1"])


def test_training_schema_rejects_duplicate_selected_ids():
    with pytest.raises(ValidationError):
        EvidenceCritiqueOutput(selected_ids=["c1", "c1"])



