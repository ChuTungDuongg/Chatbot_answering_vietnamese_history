"""The optional judge is a strict, offline import/export contract."""

import json
import sqlite3

import pytest
from pydantic import ValidationError

from app.eval_studio.store import ConflictError, EvalStore
from app.eval_studio.v11 import StudioV11


@pytest.fixture
def records(tmp_path):
    store = EvalStore(tmp_path / "judge.sqlite")
    question = store.questions()[0]
    store.save_question(
        {
            **question,
            "question": "Năm nào?",
            "reference_answer": "Năm 938",
            "annotator_notes": "PRIVATE ANNOTATOR",
        }
    )
    store.snapshot("judge-v1")
    studio = StudioV11(store)
    experiment = studio.create_experiment(
        {"name": "Offline", "snapshot_id": "judge-v1", "mcp_config": "PRIVATE CONFIG"}
    )
    run = studio.create_run(
        {
            "experiment_id": experiment["id"],
            "question_id": question["id"],
            "answer": "Năm 938 [S1]",
            "evidence": [{"chunk_id": "c1", "excerpt": "Năm 938", "selected": True}],
        }
    )
    return store, studio, run


def judges(records):
    from app.eval_studio.judge import JudgeStore

    return JudgeStore(records[0])


def profile(store, **updates):
    return store.create_profile(
        {"name": "History rubric", "provider": "offline", "model": "judge-model", "version": "1", **updates}
    )


def score(store, item, run, value=4, **updates):
    return store.record_score(
        {
            "profile_id": item["id"],
            "run_id": run["id"],
            "scores": dict.fromkeys(("accuracy", "coverage", "reasoning", "evidence", "clarity"), value),
            "external_id": "import-1",
            "uncertainty": "medium",
            "notes": "A concise assessment.",
            **updates,
        }
    )


def human(records, value):
    return records[0].create_review(
        {
            "response_id": records[2]["response_id"],
            "scores": dict.fromkeys(("accuracy", "coverage", "reasoning", "evidence", "clarity"), value),
            "decision": "accept",
        }
    )


def test_profiles_are_immutable_versions_and_storage_survives_reopen(records):
    judge = judges(records)
    item = profile(judge)
    assert [
        item["rubric"][key]["weight"] for key in ("accuracy", "coverage", "reasoning", "evidence", "clarity")
    ] == [30, 20, 20, 20, 10]
    with pytest.raises(ConflictError):
        profile(judge, model="changed")
    second = profile(judge, version="2")
    assert len(judges(records).profiles()) == 2
    assert second["id"] != item["id"]
    assert judges(records).profiles()[-1]["model"] == "judge-model"


def test_judge_schema_is_additive_and_requires_v11(tmp_path):
    from app.eval_studio.judge import JudgeStore

    old = EvalStore(tmp_path / "v10.sqlite")
    with pytest.raises(ValueError, match="v1.1"):
        JudgeStore(old)
    StudioV11(old)
    JudgeStore(old)
    assert len(old.questions()) == 40
    with old.connect() as db:
        assert db.execute("PRAGMA foreign_key_list(evaluation_judge_scores)").fetchall()


def test_structured_scores_are_strict_and_idempotent(records):
    judge = judges(records)
    item = profile(judge)
    result = score(judge, item, records[2])
    assert result["weighted_score"] == 80.0
    assert score(judge, item, records[2]) == result
    assert len(judge.scores(records[2]["id"])) == 1
    with pytest.raises(ConflictError):
        score(judge, item, records[2], value=3)
    with pytest.raises(KeyError):
        score(judge, item, {"id": "missing"}, external_id="new")


@pytest.mark.parametrize("bad", [-1, 6, True, "4", 4.5])
def test_judge_rejects_invalid_or_coerced_scores(records, bad):
    judge = judges(records)
    item = profile(judge)
    with pytest.raises(ValidationError):
        score(judge, item, records[2], value=bad)


def test_judge_rubric_requires_five_criteria_weights_summing_to_100(records):
    judge = judges(records)
    defaults = profile(judge)["rubric"]
    invalid = {**defaults, "accuracy": {"description": "Accuracy", "weight": 31}}
    with pytest.raises(ValidationError):
        profile(judge, version="bad-sum", rubric=invalid)
    with pytest.raises(ValidationError):
        profile(judge, version="bad-key", rubric={"accuracy": defaults["accuracy"]})
    weights = {key: {**criterion, "weight": 20} for key, criterion in defaults.items()}
    custom = profile(judge, version="equal", rubric=weights)
    result = score(
        judge,
        custom,
        records[2],
        scores={"accuracy": 5, "coverage": 0, "reasoning": 0, "evidence": 0, "clarity": 0},
    )
    assert result["weighted_score"] == 20.0


def test_judge_contract_contains_frozen_reference_and_safe_evidence_only(records):
    judge = judges(records)
    item = profile(judge)
    original = records[0].questions()[0]
    records[0].save_question({**original, "reference_answer": "Edited live reference"})
    contract = judge.contract(item["id"], records[2]["id"])
    assert contract["execution_enabled"] is False
    assert contract["input"]["question"]["reference_answer"] == "Năm 938"
    assert contract["input"]["answer"] == "Năm 938 [S1]"
    assert contract["input"]["evidence"][0]["excerpt"] == "Năm 938"
    assert contract["output_schema"]["additionalProperties"] is False
    assert contract["input_schema"]["additionalProperties"] is False
    assert contract["output_schema"]["properties"]["run_id"]["const"] == records[2]["id"]
    assert "PRIVATE" not in json.dumps(contract)
    assert "annotator_notes" not in contract["input"]["question"]


def test_judge_calibration_pairs_latest_human_and_judge_once_per_run(records):
    judge = judges(records)
    item = profile(judge)
    assert judge.calibration(item["id"]) == {"count": 0, "mean_abs_error": None, "pairs": []}
    first = human(records, 2)
    score(judge, item, records[2], value=1, human_review_id=first["id"])
    latest_human = human(records, 4)
    latest_judge = score(judge, item, records[2], value=3, external_id="latest")
    calibration = judge.calibration(item["id"])
    assert calibration["count"] == 1
    assert calibration["mean_abs_error"] == 20.0
    assert calibration["pairs"][0]["human_review_id"] == latest_human["id"]
    assert calibration["pairs"][0]["judge_score_id"] == latest_judge["id"]
    assert calibration["pairs"][0]["human_score"] == 80.0
    assert calibration["pairs"][0]["judge_score"] == 60.0


def test_linked_human_review_must_belong_to_scored_run(records):
    judge = judges(records)
    item = profile(judge)
    exp = records[1].experiments()[0]
    other = records[1].create_run(
        {"experiment_id": exp["id"], "question_id": "HIST-EVAL-002", "answer": "Other"}
    )
    review = records[0].create_review(
        {
            "response_id": other["response_id"],
            "scores": dict.fromkeys(("accuracy", "coverage", "reasoning", "evidence", "clarity"), 3),
            "decision": "accept",
        }
    )
    with pytest.raises(ValueError, match="review"):
        score(judge, item, records[2], human_review_id=review["id"])
    assert judge.scores() == []


def test_judge_rejects_credentials_and_hidden_reasoning_fields(records):
    judge = judges(records)
    with pytest.raises(ValidationError):
        profile(judge, name="api_key=abcdefghijklmnopqrstuvwxyz")
    item = profile(judge)
    with pytest.raises(ValidationError):
        score(judge, item, records[2], notes="Bearer abcdefghijklmnopqrstuvwxyz12345")
    with pytest.raises(ValidationError):
        score(judge, item, records[2], chain_of_thought="private")
    assert judge.scores() == []
    with records[0].connect() as db:
        run = {**records[2], "answer": "password=abcdefghijklmnopqrstuvwxyz"}
        db.execute("UPDATE evaluation_runs SET data=? WHERE id=?", (json.dumps(run), run["id"]))
    with pytest.raises(ValueError, match="credential"):
        judge.contract(item["id"], records[2]["id"])


def test_score_profile_and_run_foreign_keys_prevent_orphans(records):
    judge = judges(records)
    item = profile(judge)
    score(judge, item, records[2])
    with records[0].connect() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM evaluation_runs WHERE id=?", (records[2]["id"],))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM evaluation_judge_profiles WHERE id=?", (item["id"],))


def test_contract_rejects_credentials_split_across_reference_mapping(records):
    judge = judges(records)
    item = profile(judge)
    with records[0].connect() as db:
        data = json.loads(
            db.execute("SELECT data FROM dataset_versions WHERE version='judge-v1'").fetchone()[0]
        )
        data["cases"][0]["rubric_overrides"] = {"api_key": "abcdefghijklmnopqrstuvwxyz"}
        db.execute("UPDATE dataset_versions SET data=? WHERE version='judge-v1'", (json.dumps(data),))
    with pytest.raises(ValueError, match="credential"):
        judge.contract(item["id"], records[2]["id"])
