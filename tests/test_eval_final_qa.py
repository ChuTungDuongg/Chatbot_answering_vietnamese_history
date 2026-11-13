"""Regression coverage for the final production audit (no external inference)."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from app.eval_studio.batch import apply_batch, preview_batch
from app.eval_studio.store import ConflictError, EvalStore
from app.eval_studio.v11 import StudioV11


@pytest.fixture
def studio(tmp_path):
    store = EvalStore(tmp_path / "qa.sqlite")
    store.snapshot("first")
    return StudioV11(store)


def experiment(studio, name="A"):
    return studio.create_experiment({"name": name, "snapshot_id": "first"})


def scores(value):
    return dict.fromkeys(("accuracy", "coverage", "reasoning", "evidence", "clarity"), value)


def test_import_preview_pins_experiment_snapshot_and_configuration(studio):
    exp = experiment(studio)
    content = json.dumps([{"experiment_id": exp["id"], "question_id": "HIST-EVAL-001", "answer": "A"}])
    preview = preview_batch(studio, "runs", "json", content)
    studio.store.snapshot("second")
    studio.update_experiment(exp["id"], {**exp, "snapshot_id": "second", "model_id": "changed"})
    with pytest.raises(ConflictError, match="changed since preview"):
        apply_batch(studio, "runs", "json", content, preview["token"])
    assert studio.runs() == []


def test_cached_import_does_not_claim_deleted_records_exist(studio):
    exp = experiment(studio)
    content = json.dumps([{"experiment_id": exp["id"], "question_id": "HIST-EVAL-001", "answer": "A"}])
    preview = preview_batch(studio, "runs", "json", content)
    result = apply_batch(studio, "runs", "json", content, preview["token"])
    assert apply_batch(studio, "runs", "json", content, preview["token"]) == result
    studio.delete_run(studio.runs()[0]["id"])
    fresh = preview_batch(studio, "runs", "json", content)
    with pytest.raises(ConflictError, match="previous import"):
        apply_batch(studio, "runs", "json", content, fresh["token"])
    assert not studio.runs()


def test_batch_external_ids_are_scoped_to_experiment(studio):
    a, b = experiment(studio), experiment(studio, "B")
    content = json.dumps(
        [
            {"experiment_id": e["id"], "question_id": "HIST-EVAL-001", "answer": "A", "external_id": "same"}
            for e in (a, b)
        ]
    )
    preview = preview_batch(studio, "runs", "json", content)
    assert not preview["errors"]
    apply_batch(studio, "runs", "json", content, preview["token"])
    assert len(studio.runs()) == 2


def test_bom_csv_roundtrip(studio):
    q = studio.store.questions()[0]
    content = f'\ufeffid,question,version\n{q["id"]},Updated,{q["version"]}\n'
    preview = preview_batch(studio, "questions", "csv", content)
    assert not preview["errors"]
    apply_batch(studio, "questions", "csv", content, preview["token"], "overwrite")
    assert studio.store.get_question(q["id"])["question"] == "Updated"


def test_diagnosis_filter_cannot_select_an_older_reviewed_run(studio):
    a, b = experiment(studio), experiment(studio, "B")
    for score, tags in ((1, ["context_truncation"]), (5, [])):
        run = studio.create_run(
            {"experiment_id": a["id"], "question_id": "HIST-EVAL-001", "answer": str(score)}
        )
        studio.store.create_review(
            {
                "response_id": run["response_id"],
                "scores": scores(score),
                "decision": "accept",
                "diagnosis_tags": tags,
            }
        )
    assert studio.compare([a["id"], b["id"]])["experiments"][0]["mean_score"] == 100
    filtered = studio.compare([a["id"], b["id"]], diagnosis="context_truncation")
    assert filtered["experiments"][0]["mean_score"] is None


def test_duplicate_human_review_retry_is_idempotent_even_concurrently(studio):
    response = studio.store.create_response({"question_id": "HIST-EVAL-001", "answer": "A"})
    data = {"response_id": response["id"], "scores": scores(3), "decision": "accept"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(studio.store.create_review, [data, data]))
    assert results[0]["id"] == results[1]["id"]
    assert len(studio.store.reviews()) == 1
    studio.store.create_review({**data, "scores": scores(4)})
    assert len(studio.store.reviews()) == 2
    studio.store.create_review(data)
    assert len(studio.store.reviews()) == 3
    assert studio.store.reviews()[0]["scores"] == scores(3)


def test_migration_schema_failure_rolls_back_all_new_tables(tmp_path):
    store = EvalStore(tmp_path / "migration.sqlite")
    original_connect = store.connect

    @contextmanager
    def failing_connect():
        with original_connect() as db:
            db.set_authorizer(
                lambda op, name, *args: (
                    sqlite3.SQLITE_DENY
                    if op == sqlite3.SQLITE_CREATE_TABLE and name == "evaluation_evidence"
                    else sqlite3.SQLITE_OK
                )
            )
            yield db

    store.connect = failing_connect
    with pytest.raises(sqlite3.DatabaseError):
        StudioV11(store)
    with original_connect() as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='evaluation_experiments'").fetchone()
    store.connect = original_connect
    assert StudioV11(store).experiments() == []


def test_large_imported_id_cannot_poison_question_allocator(studio):
    question = {**studio.store.questions()[0], "id": "HIST-EVAL-9223372036854775807"}
    studio.store.import_questions([question])
    studio.store.delete_question(question["id"])
    assert EvalStore(studio.store.path).create_question()["id"] == "HIST-EVAL-041"


def test_forty_outputs_twenty_reviews_and_multiple_reviews_count_once(studio):
    a, b = experiment(studio), experiment(studio, "B")
    for index in range(1, 41):
        run = studio.create_run(
            {"experiment_id": a["id"], "question_id": f"HIST-EVAL-{index:03}", "answer": "A"}
        )
        if index <= 20:
            for score in (2, 4):
                studio.store.create_review(
                    {"response_id": run["response_id"], "scores": scores(score), "decision": "accept"}
                )
    summary = studio.compare([a["id"], b["id"]])["experiments"][0]
    assert (summary["total_questions"], summary["evaluated_questions"], summary["missing_reviews"]) == (
        40,
        20,
        20,
    )
    assert summary["mean_score"] == 80
    assert summary["latency_ms"] is None


def test_final_insert_failure_rolls_back_entire_batch(studio, monkeypatch):
    exp = experiment(studio)
    content = json.dumps(
        [
            {
                "experiment_id": exp["id"],
                "question_id": f"HIST-EVAL-{i:03}",
                "answer": "A",
                "evidence": [{"chunk_id": "same-id", "excerpt": "captured"}],
            }
            for i in (1, 2)
        ]
    )
    preview = preview_batch(studio, "runs", "json", content)
    insert = studio._insert_run
    calls = 0

    def fail_last(db, item, config):
        nonlocal calls
        calls += 1
        result = insert(db, item, config)
        if calls == 2:
            raise sqlite3.IntegrityError("Simulated final constraint failure")
        return result

    monkeypatch.setattr(studio, "_insert_run", fail_last)
    with pytest.raises(sqlite3.IntegrityError):
        apply_batch(studio, "runs", "json", content, preview["token"])
    with studio.store.connect() as db:
        for table in (
            "evaluation_runs",
            "evaluation_responses",
            "evaluation_evidence",
            "evaluation_import_batches",
        ):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_evidence_chunk_identity_is_scoped_to_run_and_snapshot_is_immutable(studio):
    exp = experiment(studio)
    runs = [
        studio.create_run(
            {
                "experiment_id": exp["id"],
                "question_id": "HIST-EVAL-001",
                "answer": text,
                "evidence": [{"chunk_id": "shared", "excerpt": text}],
            }
        )
        for text in ("Old source", "New source")
    ]
    q = studio.store.get_question("HIST-EVAL-001")
    studio.store.save_question({**q, "reference_answer": "Edited after snapshot"})
    for run in runs:
        detail = studio.evidence(run["id"])
        assert detail["evidence"][0]["excerpt"] == run["answer"]
        assert detail["question"]["reference_answer"] == ""


def test_preview_does_not_echo_rejected_credentials(studio):
    exp = experiment(studio)
    secret = "sk-abcdefghijklmnopqrstuvwx"
    content = json.dumps([{"experiment_id": exp["id"], "question_id": "HIST-EVAL-001", "answer": secret}])
    preview = preview_batch(studio, "runs", "json", content)
    assert preview["errors"]
    assert secret not in json.dumps(preview)


def test_backup_paths_with_fragment_characters_are_literal(tmp_path):
    from app.eval_studio.backup import create_backup, restore_backup

    store = EvalStore(tmp_path / "source#one.sqlite")
    archive = tmp_path / "backup.zip"
    create_backup(store.path, archive)
    target = tmp_path / "restored#two.sqlite"
    restore_backup(archive, target)
    assert len(EvalStore(target).questions()) == 40
