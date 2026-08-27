from __future__ import annotations

from tests.evidence_v2_fixtures import sanity_rows


def test_duplicate_fixture_keeps_one_supporting_copy():
    row = next(item for item in sanity_rows() if item["id"] == "case-d")
    selected = [item["evidence_id"] for item in row["output"]["selected_evidence"]]
    duplicate_ids = {
        item for item in row["metadata"]["augmented_evidence_ids"]
    }
    assert row["output"]["status"] == "sufficient"
    assert selected == ["ev_31"]
    assert duplicate_ids == {"ev_32"}
    assert not duplicate_ids.intersection(selected)
