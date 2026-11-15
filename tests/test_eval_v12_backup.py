import json
import zipfile

from app.eval_studio.backup import create_backup, restore_backup
from app.eval_studio.store import EvalStore
from app.eval_studio.v11 import StudioV11


def test_v12_backup_preserves_jobs_measurements_judge_and_accepts_v11(tmp_path):
    from app.eval_studio.benchmark import BenchmarkStore
    from app.eval_studio.judge import JudgeStore

    store = EvalStore(tmp_path / "source.sqlite")
    question = store.questions()[0]
    store.save_question({**question, "question": "A fixture question"})
    store.snapshot("source")
    studio = StudioV11(store)
    legacy = tmp_path / "legacy.zip"
    create_backup(store.path, legacy)
    # A pre-v1.2 archive remains supported after new migrations are installed.
    with zipfile.ZipFile(legacy) as source:
        data = source.read("evaluation.sqlite")
        manifest = json.loads(source.read("manifest.json"))
    manifest["schema_version"] = "1.1"
    with zipfile.ZipFile(legacy, "w") as source:
        source.writestr("manifest.json", json.dumps(manifest))
        source.writestr("evaluation.sqlite", data)
    restore_backup(legacy, tmp_path / "legacy.sqlite")
    exp = studio.create_experiment({"name": "Backup", "snapshot_id": "source"})
    jobs = BenchmarkStore(studio)
    job = jobs.create(
        {"experiment_id": exp["id"], "question_ids": [question["id"]]},
        {
            "experiment_id": exp["id"],
            "snapshot_id": "source",
            "configuration_hash": "one",
            "prompt_hash": "one",
            "tool_hash": "one",
        },
    )
    jobs.claim(job["id"], "worker")
    item = jobs.next_item(job["id"], "worker")
    run_id = jobs.finish_attempt(
        job["id"], item, "worker", {"answer": "Fixture", "completion_status": "completed"}
    )
    judges = JudgeStore(store)
    profile = judges.create_profile(
        {"name": "Judge", "provider": "openai", "model": "fixture", "version": "one"}
    )
    judges.record_score(
        {
            "profile_id": profile["id"],
            "run_id": run_id,
            "scores": dict.fromkeys(["accuracy", "coverage", "reasoning", "evidence", "clarity"], 3),
        }
    )
    archive = tmp_path / "v12.zip"
    manifest = create_backup(store.path, archive)
    assert manifest["counts"]["evaluation_benchmarks"] == 1
    assert manifest["counts"]["evaluation_judge_scores"] == 1
    restored = tmp_path / "restored.sqlite"
    restore_backup(archive, restored)
    recovered = BenchmarkStore(StudioV11(EvalStore(restored)))
    assert recovered.get(job["id"])["items"][0]["run_id"] == run_id
    assert recovered.measurements(run_id) == jobs.measurements(run_id)
    assert JudgeStore(recovered.store).scores() == judges.scores()
