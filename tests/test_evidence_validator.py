from __future__ import annotations

import json
from copy import deepcopy

from tests.evidence_v2_fixtures import sanity_rows
from training.evidence_agent.evaluate import evaluate_rows
from training.evidence_agent.validate_dataset import validate_rows


def _sync_assistant(row):
    row["messages"][2]["content"] = json.dumps(row["output"], ensure_ascii=False, sort_keys=True)


def _sync_user(row):
    row["messages"][1]["content"] = json.dumps(row["input"], ensure_ascii=False, sort_keys=True)


def test_validator_accepts_complete_sanity_fixture_set():
    report = validate_rows(sanity_rows())
    assert report["valid"], report["errors"]
    assert report["split_group_overlap"] == {"train_eval": 0, "train_test": 0, "eval_test": 0}
    assert set(report["status_distribution"]) == {"sufficient", "insufficient", "conflicting"}


def test_validator_rejects_invented_id_and_generic_summary():
    rows = sanity_rows()
    broken = deepcopy(rows[0])
    broken["id"] = "broken"
    broken["group_id"] = "broken-group"
    broken["output"]["selected_evidence"][0]["evidence_id"] = "invented"
    broken["output"]["summary"] = "Evidence đã được lọc từ context huấn luyện."
    _sync_assistant(broken)
    rows[0] = broken

    report = validate_rows(rows)
    assert not report["valid"]
    assert any("do not exist" in error for error in report["errors"])
    assert any("generic template" in error for error in report["errors"])


def test_evaluator_scores_exact_empty_selection_as_a_match():
    rows = sanity_rows()
    report = evaluate_rows(rows, rows)
    assert report["runtime_schema_validity"] == 1.0
    assert report["selected_evidence_f1"] == 1.0
    assert report["invented_id_rate"] == 0.0


def test_validator_rejects_partial_input_with_full_coverage_and_generic_missing_text():
    rows = sanity_rows()
    broken = next(item for item in rows if item["id"] == "case-partial")
    full = "Ngô Quyền sinh ngày 17 tháng 4 năm 898 và mất ngày 14 tháng 2 năm 944."
    broken["evidence"][0]["text"] = full
    broken["input"]["evidence"][0]["text"] = full
    broken["metadata"]["gold_answer"] = full
    broken["output"]["missing_information"] = [
        "Thiếu evidence cho các phần còn lại của câu hỏi: Ngô Quyền sinh và mất khi nào?"
    ]
    _sync_user(broken)
    _sync_assistant(broken)

    report = validate_rows(rows)
    assert not report["valid"]
    assert report["partial_with_full_gold_answer_coverage"] == 1
    assert report["generic_missing_information_count"] == 1
