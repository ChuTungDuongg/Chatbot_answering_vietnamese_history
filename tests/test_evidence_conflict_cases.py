from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.schemas import EvidenceModelOutput
from tests.evidence_v2_fixtures import sanity_rows


def test_conflict_fixture_names_both_existing_evidence_ids():
    row = next(item for item in sanity_rows() if item["id"] == "case-e")
    conflict = row["output"]["conflicts"][0]
    assert row["output"]["status"] == "conflicting"
    assert "ev_41" in conflict and "ev_42" in conflict


def test_conflicting_status_requires_a_conflict_description():
    with pytest.raises(ValidationError, match="conflicting output requires"):
        EvidenceModelOutput(
            status="conflicting",
            selected_evidence=[],
            conflicts=[],
            missing_information=[],
            summary="Có mâu thuẫn.",
        )

