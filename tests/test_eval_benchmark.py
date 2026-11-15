"""Durable benchmark execution without network or paid model requests."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.eval_studio.store import ConflictError, EvalStore
from app.eval_studio.v11 import StudioV11


def setup(tmp_path, **options):
    from app.eval_studio.benchmark import BenchmarkStore

    store = EvalStore(tmp_path / "benchmark.sqlite")
    for q in store.questions()[:2]:
        store.save_question(
            {**q, "question": f"Question {q['id']}", "reference_answer": "DO NOT SEND TO MODEL"}
        )
    store.snapshot("v12")
    studio = StudioV11(store)
    exp = studio.create_experiment(
        {
            "name": "A",
            "snapshot_id": "v12",
            "provider": "openai",
            "model_id": "gpt-6-luna",
            "chat_mode": "hybrid",
            "final_k": 6,
            "dense_backend": "faiss",
        }
    )
    jobs = BenchmarkStore(studio)
    config = {
        **exp,
        "experiment_id": exp["id"],
        "configuration_hash": "fixture-config",
        "prompt_hash": "fixture-prompt",
        "tool_hash": "fixture-tools",
    }
    job = jobs.create(
        {
            "experiment_id": exp["id"],
            "question_ids": ["HIST-EVAL-001", "HIST-EVAL-002"],
            "requests_per_minute": 600,
            **options,
        },
        config,
    )
    return jobs, job


class Executor:
    def __init__(self, failures=0):
        self.calls = []
        self.failures = failures

    async def execute(self, question_text, configuration, cancel):
        self.calls.append(question_text)
        if self.failures:
            self.failures -= 1
            return {"answer": "", "completion_status": "failed", "error_reason": "rate_limit"}
        return {
            "answer": "A [chunk-1]",
            "completion_status": "completed",
            "latency_ms": 1.25,
            "input_tokens": None,
            "output_tokens": 3,
            "evidence": [
                {"chunk_id": "chunk-1", "excerpt": "captured source", "selected": True, "cited": True}
            ],
            "diagnostics": {
                "tool_calls": [],
                "capture": {"tool_trace": True},
                "planning": {"exit_reason": "evidence_sufficient"},
            },
        }


def test_create_is_inert_and_requires_explicit_start_confirmation(tmp_path):
    from app.eval_studio.benchmark import BenchmarkManager

    jobs, job = setup(tmp_path)
    executor = Executor()
    manager = BenchmarkManager(jobs, executor)
    assert job["status"] == "ready" and executor.calls == []

    async def exercise():
        with pytest.raises(ValueError, match="confirm"):
            await manager.start(job["id"], False)
        await manager.start(job["id"], True)
        await manager.wait(job["id"])

    asyncio.run(exercise())
    result = jobs.get(job["id"])
    assert result["status"] == "completed"
    assert result["counts"]["completed"] == 2
    assert len(jobs.studio.runs()) == 2
    assert all("DO NOT SEND" not in value for value in executor.calls)
    assert (
        jobs.measurements(result["items"][0]["run_id"])["metrics"]["answer_metrics"]["token_f1"] is not None
    )
    assert jobs.studio.runs()[0]["input_tokens"] is None


def test_job_lease_is_atomic_and_recovery_never_starts_itself(tmp_path, monkeypatch):
    jobs, job = setup(tmp_path)

    def claim(worker):
        try:
            jobs.claim(job["id"], worker)
            return "ok"
        except ConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, ["worker-a", "worker-b"])) == ["conflict", "ok"]
    recreated = type(jobs)(StudioV11(EvalStore(jobs.store.path)))
    assert recreated.get(job["id"])["status"] == "running"
    with jobs.store.connect() as db:
        db.execute("UPDATE evaluation_benchmarks SET lease_expires=0 WHERE id=?", (job["id"],))
    recreated.claim(job["id"], "new-worker")
    assert recreated.get(job["id"])["status"] == "running"


def test_retry_retains_every_actual_attempt_and_resume_does_not_duplicate_success(tmp_path):
    from app.eval_studio.benchmark import BenchmarkManager

    jobs, job = setup(tmp_path, max_retries=1)
    executor = Executor(1)
    manager = BenchmarkManager(jobs, executor, retry_delay=0)

    async def exercise():
        await manager.start(job["id"], True)
        await manager.wait(job["id"])
        with pytest.raises(ConflictError):
            await manager.start(job["id"], True)

    asyncio.run(exercise())
    assert len(executor.calls) == 3
    assert len(jobs.studio.runs()) == 3
    assert jobs.get(job["id"])["counts"]["completed"] == 2
    assert sum(r["completion_status"] == "failed" for r in jobs.studio.runs()) == 1


def test_cancel_active_request_and_explicit_resume(tmp_path):
    from app.eval_studio.benchmark import BenchmarkManager

    jobs, job = setup(tmp_path)
    entered = asyncio.Event()

    class Slow(Executor):
        async def execute(self, *args):
            entered.set()
            await asyncio.sleep(30)

    async def exercise():
        manager = BenchmarkManager(jobs, Slow())
        await manager.start(job["id"], True)
        await entered.wait()
        await manager.cancel(job["id"])
        await manager.wait(job["id"])
        assert jobs.get(job["id"])["status"] == "cancelled"
        resumed = BenchmarkManager(jobs, Executor())
        await resumed.start(job["id"], True)
        await resumed.wait(job["id"])

    asyncio.run(exercise())
    assert jobs.get(job["id"])["counts"]["completed"] == 2


def test_recording_attempt_is_atomic_and_idempotent(tmp_path):
    jobs, job = setup(tmp_path)
    jobs.claim(job["id"], "worker")
    item = jobs.next_item(job["id"], "worker")
    result = asyncio.run(Executor().execute("q", {}, None))
    first = jobs.finish_attempt(job["id"], item, "worker", result)
    assert jobs.finish_attempt(job["id"], item, "worker", result) == first
    assert len(jobs.studio.runs()) == 1
    with pytest.raises(ConflictError):
        jobs.studio.update_experiment(job["experiment_id"], {"name": "Changed", "snapshot_id": "v12"})


def test_blank_snapshot_and_config_mismatch_do_not_create_jobs(tmp_path):
    jobs, job = setup(tmp_path)
    with pytest.raises(ValueError, match="question"):
        jobs.create(
            {"experiment_id": job["experiment_id"], "question_ids": ["HIST-EVAL-003"]}, job["configuration"]
        )
    with pytest.raises(ConflictError):
        jobs.create(
            {"experiment_id": job["experiment_id"]}, {**job["configuration"], "snapshot_id": "different"}
        )


def test_cancel_does_not_discard_a_completed_paid_answer(tmp_path):
    from app.eval_studio.benchmark import BenchmarkManager

    jobs, job = setup(tmp_path)

    class FinishThenCancel(Executor):
        async def execute(self, question, config, cancel):
            result = await super().execute(question, config, cancel)
            cancel.set()
            return result

    async def exercise():
        manager = BenchmarkManager(jobs, FinishThenCancel())
        await manager.start(job["id"], True)
        await manager.wait(job["id"])

    asyncio.run(exercise())
    assert jobs.get(job["id"])["counts"]["completed"] == 1
    assert jobs.studio.runs()[0]["answer"] == "A [chunk-1]"


def test_job_creation_rejects_experiment_changed_during_freeze(tmp_path):
    jobs, job = setup(tmp_path)
    exp = jobs.studio.create_experiment({"name": "Mutable", "snapshot_id": "v12", "provider": "openai"})
    frozen = {**job["configuration"], "experiment_id": exp["id"]}
    jobs.studio.update_experiment(exp["id"], {**exp, "provider": "local"})
    with pytest.raises(ConflictError, match="changed"):
        jobs.create({"experiment_id": exp["id"], "question_ids": ["HIST-EVAL-001"]}, frozen, exp)


def test_stale_worker_cannot_retry_a_new_workers_attempt(tmp_path):
    jobs, job = setup(tmp_path)
    jobs.claim(job["id"], "old")
    old = jobs.next_item(job["id"], "old")
    jobs.finish_attempt(job["id"], old, "old", {"completion_status": "failed", "error_reason": "timeout"})
    with jobs.store.connect() as db:
        db.execute("UPDATE evaluation_benchmarks SET lease_expires=0 WHERE id=?", (job["id"],))
    jobs.claim(job["id"], "new")
    current = jobs.next_item(job["id"], "new")
    jobs.finish_attempt(job["id"], current, "new", {"completion_status": "failed", "error_reason": "timeout"})
    with pytest.raises(ConflictError):
        jobs.retry_item(job["id"], old, "old")
    assert jobs.get(job["id"])["items"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "result", [{"completion_status": "pending"}, {"completion_status": "completed", "answer": " "}]
)
def test_invalid_executor_result_cannot_be_recorded_as_a_success(tmp_path, result):
    jobs, job = setup(tmp_path)
    jobs.claim(job["id"], "worker")
    item = jobs.next_item(job["id"], "worker")
    with pytest.raises(ValueError):
        jobs.finish_attempt(job["id"], item, "worker", result)
    assert not jobs.studio.runs()


def test_measurement_failure_rolls_back_response_evidence_and_run(tmp_path, monkeypatch):
    jobs, job = setup(tmp_path)
    jobs.claim(job["id"], "worker")
    item = jobs.next_item(job["id"], "worker")
    result = asyncio.run(Executor().execute("q", {}, None))

    def broken_metric(*args):
        raise ValueError("Invalid captured metric")

    monkeypatch.setattr("app.eval_studio.behavior.evaluate_behavior", broken_metric)
    with pytest.raises(ValueError):
        jobs.finish_attempt(job["id"], item, "worker", result)
    with jobs.store.connect() as db:
        for table in (
            "evaluation_runs",
            "evaluation_responses",
            "evaluation_evidence",
            "evaluation_run_measurements",
        ):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert jobs.get(job["id"])["items"][0]["status"] == "running"


def test_request_timeout_is_retried_without_cancelling_the_job(tmp_path):
    from app.eval_studio.benchmark import BenchmarkManager

    jobs, job = setup(tmp_path, timeout_seconds=0.05, max_retries=1)

    class TimeoutOnce(Executor):
        async def execute(self, *args):
            if not self.calls:
                self.calls.append("timed-out")
                await asyncio.sleep(10)
            return await super().execute(*args)

    async def exercise():
        manager = BenchmarkManager(jobs, TimeoutOnce(), retry_delay=0)
        await manager.start(job["id"], True)
        await manager.wait(job["id"])

    asyncio.run(exercise())
    assert jobs.get(job["id"])["counts"]["completed"] == 2
    failed = next(run for run in jobs.studio.runs() if run["completion_status"] == "failed")
    assert failed["error_reason"] == "timeout" and failed["latency_ms"] > 0
