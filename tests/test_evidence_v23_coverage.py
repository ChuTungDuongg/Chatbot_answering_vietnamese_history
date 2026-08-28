from __future__ import annotations

from collections import defaultdict

from training.evidence_agent.prepare_dataset import build_dataset_v2, v23_source_fixtures
from training.evidence_agent.validate_dataset import validate_rows


def fixture_rows():
    return build_dataset_v2(
        v23_source_fixtures(),
        duplicate_ratio=0.0,
        conflict_ratio=0.0,
        partial_ratio=0.0,
        include_v23_fixtures=False,
    )


def test_date_topic_full_and_both_leave_one_out_partials():
    rows = [row for row in fixture_rows() if row["original_sample_id"] == "v23-date-topic"]
    full = next(row for row in rows if row["output"]["status"] == "sufficient")
    partials = [row for row in rows if row["behavior"] == "partial"]
    assert {item["evidence_id"] for item in full["output"]["selected_evidence"]} == {
        "ev_31f9a040", "ev_31f9a041"
    }
    assert len(partials) == 2
    assert all(row["output"]["selected_evidence"] for row in partials)
    assert {tuple(item["evidence_id"] for item in row["output"]["selected_evidence"]) for row in partials} == {
        ("ev_31f9a040",), ("ev_31f9a041",)
    }


def test_leader_opponent_full_partial_and_one_source_multi_slot():
    grouped = defaultdict(list)
    for row in fixture_rows():
        grouped[row["original_sample_id"]].append(row)
    leader = grouped["v23-leader-opponent"]
    assert len(next(row for row in leader if row["output"]["status"] == "sufficient")["output"]["selected_evidence"]) == 2
    assert len([row for row in leader if row["behavior"] == "partial"]) == 2
    one_source = next(
        row for row in grouped["v23-one-source-multi"] if row["output"]["status"] == "sufficient"
    )
    assert len(one_source["output"]["selected_evidence"]) == 1


def test_relation_partial_retains_context_and_all_v23_invariants_pass():
    rows = fixture_rows()
    relation = [row for row in rows if row["original_sample_id"] == "v23-explicit-relation"]
    partial = next(row for row in relation if row["behavior"] == "partial")
    assert partial["output"]["status"] == "insufficient"
    assert [item["evidence_id"] for item in partial["output"]["selected_evidence"]] == ["ev_d194c210"]
    assert any("liên hệ trực tiếp" in value for value in partial["output"]["missing_information"])
    report = validate_rows(rows, require_v2_behaviors=False)
    assert report["sufficient_selected_subset_incomplete"] == 0
    assert report["insufficient_partial_selected_empty"] == 0
    assert report["insufficient_no_support_selected_nonempty"] == 0
    assert report["valid"], report["errors"]

