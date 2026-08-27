from tests.test_no_evidence_leakage import phase6_row
from training.research_agent.build_history_trajectories import build_trajectory_samples
from training.research_agent.validate_dataset import validate_rows


def test_research_dataset_validator_accepts_grounded_rows():
    report = validate_rows(build_trajectory_samples(phase6_row()))
    assert report["valid"], report["errors"]
    assert report["unique_trajectory_ids"] == 1
    assert report["unique_groups"] == 1


def test_research_dataset_validator_rejects_question_leakage():
    rows = build_trajectory_samples(phase6_row())
    rows[0]["training_prompt"]["question"] += "\nTài liệu tham khảo: secret"
    report = validate_rows(rows)
    assert not report["valid"]


def test_research_dataset_validator_rejects_reused_trajectory_steps():
    rows = build_trajectory_samples(phase6_row())
    report = validate_rows([*rows, *rows])
    assert not report["valid"]
    assert any("duplicate trajectory_id" in error for error in report["errors"])
