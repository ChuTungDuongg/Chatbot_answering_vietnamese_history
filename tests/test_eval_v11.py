import pytest
from app.eval_studio.store import EvalStore, ConflictError
from app.eval_studio.v11 import StudioV11


@pytest.fixture
def studio(tmp_path):
    old = EvalStore(tmp_path / "studio.sqlite")
    first = old.questions()[0]
    old.save_question({**first, "question": "Câu hỏi gốc"})
    old.snapshot("1.0.0")
    return StudioV11(old)


def experiment(studio, name, model="model-a", snapshot="1.0.0"):
    return studio.create_experiment(
        {"name": name, "snapshot_id": snapshot, "provider": "provider", "model_id": model}
    )


def run(studio, experiment_id, answer="Answer", question_id="HIST-EVAL-001"):
    return studio.create_run({"experiment_id": experiment_id, "question_id": question_id, "answer": answer})


def review(studio, response_id, score):
    return studio.store.create_review(
        {
            "response_id": response_id,
            "scores": dict.fromkeys(["accuracy", "coverage", "reasoning", "evidence", "clarity"], score),
            "decision": "accept",
        }
    )


def test_migration_and_run_preserve_v1(studio):
    assert studio.store.get_question("HIST-EVAL-001")["question"] == "Câu hỏi gốc"
    exp = experiment(studio, "A")
    result = run(studio, exp["id"])
    assert result["snapshot_id"] == "1.0.0"
    assert studio.store.responses()[0]["id"] == result["response_id"]
    assert StudioV11(EvalStore(studio.store.path)).experiments()[0]["id"] == exp["id"]
    with pytest.raises(ValueError):
        run(studio, exp["id"], question_id="missing")


def test_comparison_does_not_score_missing_and_rejects_cross_snapshot(studio):
    a, b = experiment(studio, "A"), experiment(studio, "B", "model-b")
    ra, rb = run(studio, a["id"]), run(studio, b["id"])
    review(studio, ra["response_id"], 5)
    data = studio.compare([a["id"], b["id"]])
    assert data["experiments"][0]["mean_score"] == 100
    assert data["experiments"][1]["mean_score"] is None
    assert data["paired_questions"] == 0
    review(studio, rb["response_id"], 3)
    data = studio.compare([a["id"], b["id"]])
    assert data["paired_questions"] == 1
    assert data["pairwise"][0]["wins"] == 1
    studio.store.snapshot("2.0.0")
    c = experiment(studio, "C", snapshot="2.0.0")
    with pytest.raises(ValueError):
        studio.compare([a["id"], c["id"]])


def test_blind_mapping_not_disclosed_and_locked(studio):
    a, b = experiment(studio, "Terra", "Terra"), experiment(studio, "Luna", "Luna")
    run(studio, a["id"], "Output Terra")
    run(studio, b["id"], "Output Luna")
    session = studio.create_blind("1.0.0", a["id"], b["id"])
    public = studio.get_blind(session["id"])
    text = str(public).lower()
    assert "terra" not in text and "luna" not in text and "provider" not in text
    assert studio.get_blind(session["id"]) == public
    assignment = public["assignments"][0]
    with pytest.raises(ConflictError):
        studio.reveal_blind(session["id"], assignment["id"])
    scores = dict.fromkeys(["accuracy", "coverage", "reasoning", "evidence", "clarity"], 4)
    studio.submit_blind(
        session["id"], assignment["id"], {"scores_a": scores, "scores_b": scores, "preference": "tie"}
    )
    with pytest.raises(ConflictError):
        studio.submit_blind(
            session["id"], assignment["id"], {"scores_a": scores, "scores_b": scores, "preference": "tie"}
        )
    assert studio.reveal_blind(session["id"], assignment["id"])["a"]["model_id"] in ("Terra", "Luna")


def test_atomic_run_import_retry_and_bad_row(studio):
    from app.eval_studio.batch import preview_batch, apply_batch
    import json

    exp = experiment(studio, "A")
    good = {
        "experiment_id": exp["id"],
        "question_id": "HIST-EVAL-001",
        "answer": "A",
        "external_id": "external-1",
    }
    bad = {**good, "question_id": "missing", "external_id": "external-2"}
    content = "\n".join(json.dumps(r) for r in [good, bad])
    preview = preview_batch(studio, "runs", "jsonl", content)
    assert preview["errors"] and studio.runs() == []
    with pytest.raises(ValueError):
        apply_batch(studio, "runs", "jsonl", content, preview["token"])
    assert studio.runs() == []
    content = json.dumps(good)
    preview = preview_batch(studio, "runs", "jsonl", content)
    assert not preview["errors"]
    a = apply_batch(studio, "runs", "jsonl", content, preview["token"])
    b = apply_batch(studio, "runs", "jsonl", content, preview["token"])
    assert a == b and len(studio.runs()) == 1


def test_question_import_preview_detects_revision_change(studio):
    from app.eval_studio.batch import preview_batch, apply_batch
    import json

    q = studio.store.questions()[0]
    content = json.dumps([{**q, "question": "import"}])
    preview = preview_batch(studio, "questions", "json", content)
    studio.store.save_question({**q, "question": "concurrent edit"})
    with pytest.raises(ConflictError):
        apply_batch(studio, "questions", "json", content, preview["token"], "overwrite")
    assert studio.store.get_question(q["id"])["question"] == "concurrent edit"


def test_evaluation_backup_excludes_auth_and_restore_to_new_db(studio, tmp_path):
    from app.eval_studio.backup import create_backup, restore_backup
    import sqlite3

    with studio.store.connect() as db:
        db.execute("CREATE TABLE owner_sessions (id TEXT)")
        db.execute("INSERT INTO owner_sessions VALUES ('secret-session')")
    exp = experiment(studio, "Backup")
    run(studio, exp["id"])
    package = tmp_path / "backup.zip"
    manifest = create_backup(studio.store.path, package)
    assert manifest["counts"]["evaluation_runs"] == 1
    target = tmp_path / "restored.sqlite"
    restore_backup(package, target)
    restored = StudioV11(EvalStore(target))
    assert len(restored.runs()) == 1
    with sqlite3.connect(target) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='owner_sessions'").fetchone()
    with pytest.raises(FileExistsError):
        restore_backup(package, target)
    assert b"secret-session" not in package.read_bytes()


def test_corrupt_and_incompatible_backup_does_not_restore(studio, tmp_path):
    from app.eval_studio.backup import create_backup, restore_backup
    import json, zipfile

    archive = tmp_path / "valid.zip"
    create_backup(studio.store.path, archive)
    with zipfile.ZipFile(archive) as z:
        db, manifest = z.read("evaluation.sqlite"), json.loads(z.read("manifest.json"))
    for kind in ("corrupt", "version"):
        bad = tmp_path / f"{kind}.zip"
        info = dict(manifest)
        if kind == "corrupt":
            info["sha256"] = "0" * 64
        else:
            info["schema_version"] = "99"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("evaluation.sqlite", db)
            z.writestr("manifest.json", json.dumps(info))
        target = tmp_path / f"{kind}.sqlite"
        with pytest.raises(ValueError):
            restore_backup(bad, target)
        assert not target.exists()


def test_invalid_csv_and_duplicate_ids_report_lines(studio):
    from app.eval_studio.batch import preview_batch

    assert preview_batch(studio, "questions", "csv", 'id,question\n"unclosed')["errors"]
    q = studio.store.questions()[0]
    import json

    result = preview_batch(studio, "questions", "json", json.dumps([q, q]))
    assert result["errors"][0]["line"] == 2


def test_blind_requires_owner_and_does_not_reveal_via_api(studio, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from app.eval_studio.api import router
    from app.config import settings

    a, b = experiment(studio, "Secret A", "Terra"), experiment(studio, "Secret B", "Luna")
    run(studio, a["id"], "Terra reply")
    run(studio, b["id"], "Luna reply")
    sid = studio.create_blind("1.0.0", a["id"], b["id"])["id"]
    app = FastAPI()
    app.state.eval_store = studio.store
    app.state.eval_v11 = studio

    @app.middleware("http")
    async def owner(request: Request, call_next):
        request.state.owner_authenticated = request.headers.get("x-owner") == "yes"
        return await call_next(request)

    app.include_router(router)
    monkeypatch.setattr(settings, "eval_studio_enabled", True)
    client = TestClient(app)
    assert client.get(f"/api/v1/eval/blind/{sid}").status_code == 401
    response = client.get(f"/api/v1/eval/blind/{sid}", headers={"x-owner": "yes"})
    assert "Terra" not in response.text and "Luna" not in response.text and a["id"] not in response.text
    aid = response.json()["assignments"][0]["id"]
    assert client.get(f"/api/v1/eval/blind/{sid}/{aid}/reveal", headers={"x-owner": "yes"}).status_code == 409


def test_experiment_lifecycle_does_not_modify_recorded_config(studio):
    exp = experiment(studio, "Before")
    updated = studio.update_experiment(exp["id"], {**exp, "name": "After"})
    assert updated["name"] == "After"
    recorded = run(studio, exp["id"])
    with pytest.raises(ConflictError):
        studio.update_experiment(exp["id"], {**updated, "model_id": "changed"})
    with pytest.raises(ConflictError):
        studio.delete_experiment(exp["id"])
    studio.delete_run(recorded["id"])
    studio.delete_experiment(exp["id"])
    assert studio.experiments() == []


def test_review_diagnosis_aggregation_and_filter(studio):
    a, b = experiment(studio, "A"), experiment(studio, "B")
    ra, rb = run(studio, a["id"]), run(studio, b["id"])
    review(studio, rb["response_id"], 3)
    studio.store.create_review(
        {
            "response_id": ra["response_id"],
            "scores": dict.fromkeys(["accuracy", "coverage", "reasoning", "evidence", "clarity"], 4),
            "decision": "accept",
            "diagnosis_tags": ["context_truncation"],
        }
    )
    result = studio.compare([a["id"], b["id"]], diagnosis="context_truncation")
    assert result["experiments"][0]["diagnosis_counts"]["context_truncation"] == 1
    assert result["paired_questions"] == 0
    assert result["experiments"][1]["mean_score"] is None


def test_update_run_preserves_identity_and_locks_after_review(studio):
    exp = experiment(studio, "A")
    r = run(studio, exp["id"], "Original")
    changed = studio.update_run(
        r["id"], {"experiment_id": exp["id"], "question_id": "HIST-EVAL-001", "answer": "Corrected"}
    )
    assert changed["response_id"] == r["response_id"]
    assert studio.store.responses()[0]["answer"] == "Corrected"
    review(studio, changed["response_id"], 4)
    with pytest.raises(ConflictError):
        studio.update_run(
            r["id"], {"experiment_id": exp["id"], "question_id": "HIST-EVAL-001", "answer": "Too late"}
        )


def test_blind_concurrent_double_submit_only_one_succeeds(studio):
    from concurrent.futures import ThreadPoolExecutor

    a, b = experiment(studio, "A"), experiment(studio, "B")
    run(studio, a["id"])
    run(studio, b["id"])
    session = studio.create_blind("1.0.0", a["id"], b["id"])
    assignment = studio.get_blind(session["id"])["assignments"][0]
    scores = dict.fromkeys(["accuracy", "coverage", "reasoning", "evidence", "clarity"], 3)

    def submit(_):
        try:
            studio.submit_blind(
                session["id"], assignment["id"], {"scores_a": scores, "scores_b": scores, "preference": "tie"}
            )
            return "ok"
        except ConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, range(2))) == ["conflict", "ok"]
    assert len(studio.store.reviews()) == 2


def test_csv_preview_preserves_ground_truth(studio):
    from app.eval_studio.batch import preview_batch, apply_batch

    q = studio.store.questions()[0]
    q = studio.store.save_question({**q, "reference_answer": "Nguồn gốc quan trọng"})
    text = f'id,question,version\n{q["id"]},Câu đã sửa,{q["version"]}\n'
    preview = preview_batch(studio, "questions", "csv", text)
    assert not preview["errors"]
    apply_batch(studio, "questions", "csv", text, preview["token"], "overwrite")
    assert studio.store.get_question(q["id"])["reference_answer"] == "Nguồn gốc quan trọng"


def test_v10_migration_keeps_response_review_and_snapshot(tmp_path):
    old = EvalStore(tmp_path / "old.sqlite")
    q = old.questions()[0]
    response = old.create_response({"question_id": q["id"], "answer": "Legacy answer"})
    saved = old.create_review(
        {
            "response_id": response["id"],
            "scores": dict.fromkeys(["accuracy", "coverage", "reasoning", "evidence", "clarity"], 2),
            "decision": "accept",
        }
    )
    snapshot = old.snapshot("legacy-1")
    StudioV11(old)
    assert old.responses()[0] == response
    assert old.reviews()[0] == saved
    assert old.get_snapshot("legacy-1") == snapshot
    StudioV11(EvalStore(old.path))
    assert len(old.questions()) == 40


def test_restore_into_copy_preserves_chat_and_never_overwrites_source(studio, tmp_path):
    from app.eval_studio.backup import create_backup, restore_to_copy
    import sqlite3

    exp = experiment(studio, "Backup source")
    run(studio, exp["id"])
    backup = tmp_path / "studio.zip"
    create_backup(studio.store.path, backup)
    existing = tmp_path / "existing.sqlite"
    other = EvalStore(existing)
    with other.connect() as db:
        db.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO conversations VALUES ('chat-stays')")
    target = tmp_path / "restored-copy.sqlite"
    with pytest.raises(ValueError):
        restore_to_copy(backup, existing, target)
    assert not target.exists()
    restore_to_copy(backup, existing, target, replace_evaluation=True)
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT id FROM conversations").fetchone()[0] == "chat-stays"
    assert len(StudioV11(EvalStore(target)).runs()) == 1
    with sqlite3.connect(existing) as db:
        assert db.execute("SELECT id FROM conversations").fetchone()[0] == "chat-stays"
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='evaluation_runs'").fetchone()


def test_secrets_rejected_in_experiment_and_backup(studio, tmp_path):
    from app.eval_studio.backup import create_backup
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        studio.create_experiment(
            {"name": "Test", "snapshot_id": "1.0.0", "mcp_config": "api_key=abcdefghijklmnopqrstuvwxyz"}
        )
    q = studio.store.questions()[0]
    studio.store.save_question({**q, "annotator_notes": "Bearer abcdefghijklmnopqrstuvwxyz12345"})
    with pytest.raises(ValueError):
        create_backup(studio.store.path, tmp_path / "secret.zip")
    assert not (tmp_path / "secret.zip").exists()


def test_batch_http_body_limit(tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from app.eval_studio.api import router
    from app.config import settings

    app = FastAPI()
    app.state.eval_store = EvalStore(tmp_path / "http-limit.sqlite")

    @app.middleware("http")
    async def owner(request: Request, call_next):
        request.state.owner_authenticated = True
        return await call_next(request)

    app.include_router(router)
    monkeypatch.setattr(settings, "eval_studio_enabled", True)
    response = TestClient(app).post(
        "/api/v1/eval/batch/preview", json={"kind": "questions", "format": "json", "content": "x" * 2_100_000}
    )
    assert response.status_code == 413


def test_evidence_snapshot_and_failed_run_are_honest(studio):
    exp = experiment(studio, "A")
    captured = studio.create_run(
        {
            "experiment_id": exp["id"],
            "question_id": "HIST-EVAL-001",
            "answer": "A",
            "trace_captured": True,
            "evidence": [
                {
                    "chunk_id": "c1",
                    "excerpt": "Source text",
                    "selected": True,
                    "cited": False,
                    "dense_score": 0.7,
                }
            ],
        }
    )
    detail = studio.evidence(captured["id"])
    assert detail["question"]["question"] == "Câu hỏi gốc"
    assert detail["evidence"][0]["chunk_id"] == "c1"
    assert detail["evidence"][0]["selected"] is True
    assert detail["run"]["latency_ms"] is None
    failed = studio.create_run(
        {
            "experiment_id": exp["id"],
            "question_id": "HIST-EVAL-002",
            "completion_status": "failed",
            "error_reason": "timeout",
        }
    )
    assert failed["response_id"] is None
    assert studio.evidence(failed["id"])["trace_status"] == "not_captured"


def test_batch_api_preview_and_apply(studio, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from app.eval_studio.api import router
    from app.config import settings
    import json

    app = FastAPI()
    app.state.eval_store = studio.store
    app.state.eval_v11 = studio

    @app.middleware("http")
    async def owner(request: Request, call_next):
        request.state.owner_authenticated = True
        return await call_next(request)

    app.include_router(router)
    monkeypatch.setattr(settings, "eval_studio_enabled", True)
    client = TestClient(app)
    exp = experiment(studio, "Batch API")
    payload = {
        "kind": "runs",
        "format": "jsonl",
        "content": json.dumps(
            {"experiment_id": exp["id"], "question_id": "HIST-EVAL-001", "answer": "Manual"}
        ),
    }
    preview = client.post("/api/v1/eval/batch/preview", json=payload)
    assert preview.status_code == 200 and preview.json()["valid"] == 1
    applied = client.post("/api/v1/eval/batch/apply", json={**payload, "token": preview.json()["token"]})
    assert applied.status_code == 200 and applied.json()["imported"] == 1
    assert len(studio.runs()) == 1


def test_trace_accepts_only_safe_structured_metadata(studio):
    from pydantic import ValidationError

    exp = experiment(studio, "Trace")
    item = studio.create_run(
        {
            "experiment_id": exp["id"],
            "question_id": "HIST-EVAL-001",
            "answer": "A",
            "trace": {
                "final_context_chunk_ids": ["c1"],
                "tools": [{"name": "search_history", "status": "ok", "duration_ms": 12}],
            },
        }
    )
    assert studio.evidence(item["id"])["run"]["trace"]["final_context_chunk_ids"] == ["c1"]
    with pytest.raises(ValidationError):
        studio.create_run(
            {
                "experiment_id": exp["id"],
                "question_id": "HIST-EVAL-001",
                "answer": "A",
                "trace": {"private_reasoning": "hidden"},
            }
        )


def test_backup_restore_cli_roundtrip(studio, tmp_path):
    import subprocess, sys

    backup = tmp_path / "from-cli.zip"
    restored = tmp_path / "from-cli.sqlite"
    make = subprocess.run(
        [sys.executable, "-m", "app.eval_studio.backup", "backup", str(studio.store.path), str(backup)],
        capture_output=True,
        text=True,
    )
    assert make.returncode == 0, make.stderr
    restore = subprocess.run(
        [sys.executable, "-m", "app.eval_studio.backup", "restore", str(backup), str(restored)],
        capture_output=True,
        text=True,
    )
    assert restore.returncode == 0, restore.stderr
    assert EvalStore(restored).get_question("HIST-EVAL-001")["question"] == "Câu hỏi gốc"


def test_deleted_question_ids_are_not_reused(tmp_path):
    store = EvalStore(tmp_path / "identifiers.sqlite")
    first = store.create_question()
    store.delete_question(first["id"])
    again = EvalStore(store.path).create_question()
    assert first["id"] != again["id"]
