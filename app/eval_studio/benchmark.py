"""Owner-started, durable sequential benchmarks; never resume paid work at startup."""

import asyncio
import json
import threading
import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .models import now
from .privacy import reject_secrets
from .store import ConflictError, canonical
from .v11 import RunInput


class BenchmarkOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_id: str
    smoke: bool = False
    question_ids: list[str] = Field(default_factory=list, max_length=1000)
    timeout_seconds: float = Field(default=180, ge=0.05, le=1800, allow_inf_nan=False)
    max_retries: int = Field(default=0, ge=0, le=3)
    requests_per_minute: int = Field(default=6, ge=1, le=600)
    prompt_overrides: dict[str, str] = Field(default_factory=dict, max_length=20)
    tool_descriptions: dict[str, str] = Field(default_factory=dict, max_length=100)
    mcp_enabled: bool = False
    allowed_mcp_servers: list[str] = Field(default_factory=list, max_length=20)


def safe_diagnostics(value):
    """Keep capture contracts, never raw arguments, messages or model replay."""
    value = value or {}
    keys = (
        "name",
        "call_id",
        "arguments_valid",
        "argument_fingerprint",
        "status",
        "error_code",
        "duration_ms",
        "result_count",
    )
    result = {
        "tool_calls": [{k: row[k] for k in keys if k in row} for row in value.get("tool_calls", [])[:1000]],
        "planning": {
            k: v
            for k, v in value.get("planning", {}).items()
            if k in ("exit_reason", "action_rounds", "repeated_calls_avoided", "tools_executed")
        },
        "capture": {
            k: v is True
            for k, v in value.get("capture", {}).items()
            if k in ("tool_trace", "retrieved_evidence", "final_context")
        },
        "usage": {k: v for k, v in value.get("usage", {}).items() if v is None or type(v) in (int, float)},
    }
    reject_secrets(result)
    if len(canonical(result)) > 500_000:
        raise ValueError("Captured diagnostics exceed storage limit")
    return result


class BenchmarkStore:
    def __init__(self, studio):
        self.studio, self.store = studio, studio.store
        with self.store.connect() as db:
            db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS evaluation_benchmarks (id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, data TEXT NOT NULL, lease_owner TEXT, lease_expires REAL, FOREIGN KEY(experiment_id) REFERENCES evaluation_experiments(id));
                CREATE TABLE IF NOT EXISTS evaluation_benchmark_items (id TEXT PRIMARY KEY, benchmark_id TEXT NOT NULL, question_id TEXT NOT NULL, data TEXT NOT NULL, FOREIGN KEY(benchmark_id) REFERENCES evaluation_benchmarks(id), FOREIGN KEY(question_id) REFERENCES evaluation_questions(id), UNIQUE(benchmark_id,question_id));
                CREATE TABLE IF NOT EXISTS evaluation_run_measurements (run_id TEXT PRIMARY KEY, benchmark_id TEXT NOT NULL, data TEXT NOT NULL, FOREIGN KEY(run_id) REFERENCES evaluation_runs(id), FOREIGN KEY(benchmark_id) REFERENCES evaluation_benchmarks(id));
                CREATE INDEX IF NOT EXISTS evaluation_benchmarks_experiment_idx ON evaluation_benchmarks(experiment_id);
                CREATE INDEX IF NOT EXISTS evaluation_measurements_benchmark_idx ON evaluation_run_measurements(benchmark_id);
            """)

    def create(self, data, configuration, expected_experiment=None):
        options = BenchmarkOptions.model_validate(data).model_dump()
        reject_secrets(options)
        reject_secrets(configuration)
        job_id = str(uuid4())
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            exp_row = db.execute(
                "SELECT data FROM evaluation_experiments WHERE id=?", (options["experiment_id"],)
            ).fetchone()
            if not exp_row:
                raise KeyError(options["experiment_id"])
            exp = json.loads(exp_row[0])
            if expected_experiment is not None and exp != expected_experiment:
                raise ConflictError("Experiment changed while freezing configuration; configure again")
            if (
                configuration["experiment_id"] != exp["id"]
                or configuration["snapshot_id"] != exp["snapshot_id"]
            ):
                raise ConflictError("Experiment configuration changed; configure again")
            # One experiment denotes one immutable execution configuration.
            for (saved,) in db.execute(
                "SELECT data FROM evaluation_benchmarks WHERE experiment_id=?", (exp["id"],)
            ):
                if json.loads(saved)["configuration_hash"] != configuration["configuration_hash"]:
                    raise ConflictError(
                        "Create a separate experiment for a different prompt/tool configuration"
                    )
            snapshot = json.loads(
                db.execute(
                    "SELECT data FROM dataset_versions WHERE version=?", (exp["snapshot_id"],)
                ).fetchone()[0]
            )
            ids = options["question_ids"]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate question IDs")
            cases = [
                q
                for q in snapshot["cases"]
                if (not ids or q["id"] in ids) and (not options["smoke"] or q["smoke"])
            ]
            if ids and {q["id"] for q in cases} != set(ids):
                raise ValueError("Selected question is absent from the snapshot/filter")
            if not cases or any(not 2 <= len(q["question"].strip()) <= 1000 for q in cases):
                raise ValueError(
                    "Every selected question must contain 2–1000 characters; fill blank slots before running"
                )
            for q in cases:
                if not db.execute("SELECT 1 FROM evaluation_questions WHERE id=?", (q["id"],)).fetchone():
                    raise ValueError(
                        "Snapshot question missing from working dataset; restore its ID before execution"
                    )
            record = {
                "id": job_id,
                "experiment_id": exp["id"],
                "snapshot_id": exp["snapshot_id"],
                "status": "ready",
                "options": options,
                "configuration": configuration,
                "configuration_hash": configuration["configuration_hash"],
                "prompt_hash": configuration["prompt_hash"],
                "tool_hash": configuration["tool_hash"],
                "cancel_requested": False,
                "created_at": now(),
                "updated_at": now(),
            }
            db.execute(
                "INSERT INTO evaluation_benchmarks VALUES (?,?,?,NULL,NULL)",
                (job_id, exp["id"], canonical(record)),
            )
            for q in cases:
                item = {
                    "id": str(uuid4()),
                    "question_id": q["id"],
                    "status": "pending",
                    "attempts": 0,
                    "run_id": None,
                    "error_code": None,
                    "updated_at": now(),
                }
                db.execute(
                    "INSERT INTO evaluation_benchmark_items VALUES (?,?,?,?)",
                    (item["id"], job_id, q["id"], canonical(item)),
                )
        return self.get(job_id)

    def _job(self, db, job_id):
        row = db.execute(
            "SELECT data,lease_owner,lease_expires FROM evaluation_benchmarks WHERE id=?", (job_id,)
        ).fetchone()
        if not row:
            raise KeyError(job_id)
        return json.loads(row[0]), row[1], row[2]

    def _save(self, db, job):
        job["updated_at"] = now()
        db.execute("UPDATE evaluation_benchmarks SET data=? WHERE id=?", (canonical(job), job["id"]))

    def get(self, job_id):
        with self.store.connect() as db:
            db.execute("BEGIN")
            job, _, expiry = self._job(db, job_id)
            items = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM evaluation_benchmark_items WHERE benchmark_id=? ORDER BY question_id",
                    (job_id,),
                )
            ]
        counts = {
            key: sum(i["status"] == key for i in items)
            for key in ("pending", "running", "completed", "failed", "cancelled")
        }
        return {**job, "lease_expires_at": expiry, "counts": {"total": len(items), **counts}, "items": items}

    def list(self):
        with self.store.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM evaluation_benchmarks ORDER BY rowid DESC")]
        return [self.get(id) for id in ids]

    def claim(self, job_id, worker):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job, owner, expiry = self._job(db, job_id)
            if owner and expiry and expiry > time.time():
                raise ConflictError("Benchmark already has an active worker")
            if job["status"] == "completed":
                raise ConflictError("Benchmark is complete; create a new benchmark for another replicate")
            # Interrupted in-flight requests may already have been billed. Do not
            # replay automatically; this transition only follows explicit resume.
            for row in db.execute(
                "SELECT id,data FROM evaluation_benchmark_items WHERE benchmark_id=?", (job_id,)
            ).fetchall():
                item = json.loads(row[1])
                if item["status"] in ("running", "cancelled", "failed"):
                    if item["status"] == "running":
                        item["error_code"] = "interrupted_previous_attempt"
                        item.setdefault("interruptions", []).append(
                            {
                                "attempt": item["attempts"],
                                "recorded_at": now(),
                                "reason": "Worker stopped without a confirmed result; an upstream request may have been billed",
                            }
                        )
                    item["status"] = "pending"
                    db.execute(
                        "UPDATE evaluation_benchmark_items SET data=? WHERE id=?", (canonical(item), row[0])
                    )
            job.update(status="running", cancel_requested=False)
            self._save(db, job)
            db.execute(
                "UPDATE evaluation_benchmarks SET lease_owner=?,lease_expires=? WHERE id=?",
                (worker, time.time() + 30, job_id),
            )
        return self.get(job_id)

    def heartbeat(self, job_id, worker):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job, owner, _ = self._job(db, job_id)
            if owner != worker:
                raise ConflictError("Benchmark worker lease was lost")
            db.execute(
                "UPDATE evaluation_benchmarks SET lease_expires=? WHERE id=?", (time.time() + 30, job_id)
            )
            return job["cancel_requested"]

    def request_cancel(self, job_id):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job, owner, expiry = self._job(db, job_id)
            if job["status"] == "completed":
                raise ConflictError("Completed benchmarks cannot be cancelled")
            job["cancel_requested"] = True
            if not owner or not expiry or expiry <= time.time():
                job["status"] = "cancelled"
            self._save(db, job)
        return self.get(job_id)

    def next_item(self, job_id, worker):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job, owner, expiry = self._job(db, job_id)
            if owner != worker or expiry <= time.time():
                raise ConflictError("Benchmark worker lease was lost")
            if job["cancel_requested"]:
                return None
            for row in db.execute(
                "SELECT id,data FROM evaluation_benchmark_items WHERE benchmark_id=? ORDER BY question_id",
                (job_id,),
            ):
                item = json.loads(row[1])
                if item["status"] == "pending":
                    item.update(status="running", attempts=item["attempts"] + 1, updated_at=now())
                    db.execute(
                        "UPDATE evaluation_benchmark_items SET data=? WHERE id=?", (canonical(item), row[0])
                    )
                    return item
        return None

    def finish_attempt(self, job_id, item, worker, result):
        from .behavior import evaluate_behavior

        diagnostics = safe_diagnostics(result.get("diagnostics"))
        fields = {k: v for k, v in result.items() if k in RunInput.model_fields}
        fields.update(
            experiment_id=self.get(job_id)["experiment_id"],
            question_id=item["question_id"],
            external_id=f"benchmark:{job_id}:{item['id']}:{item['attempts']}",
        )
        if fields.get("completion_status") == "cancelled":
            fields.update(completion_status="failed", error_reason="cancelled")
        clean = RunInput.model_validate(fields).model_dump()
        if clean["completion_status"] not in ("completed", "failed"):
            raise ValueError("completion_status must be completed or failed")
        if clean["completion_status"] == "completed" and not clean["answer"].strip():
            raise ValueError("Completed run requires answer")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job, owner, expiry = self._job(db, job_id)
            existing = db.execute(
                "SELECT id FROM evaluation_runs WHERE experiment_id=? AND external_id=?",
                (job["experiment_id"], clean["external_id"]),
            ).fetchone()
            if existing:
                return existing[0]
            if owner != worker or expiry <= time.time():
                raise ConflictError("Benchmark worker lease was lost")
            current = json.loads(
                db.execute(
                    "SELECT data FROM evaluation_benchmark_items WHERE id=? AND benchmark_id=?",
                    (item["id"], job_id),
                ).fetchone()[0]
            )
            if current["attempts"] != item["attempts"] or current["status"] != "running":
                raise ConflictError("Benchmark attempt was superseded")
            snapshot = json.loads(
                db.execute(
                    "SELECT data FROM dataset_versions WHERE version=?", (job["snapshot_id"],)
                ).fetchone()[0]
            )
            question = next(q for q in snapshot["cases"] if q["id"] == item["question_id"])
            exp = json.loads(
                db.execute(
                    "SELECT data FROM evaluation_experiments WHERE id=?", (job["experiment_id"],)
                ).fetchone()[0]
            )
            run = self.studio._insert_run(db, clean, exp)
            metrics = (
                evaluate_behavior(question, run, clean["evidence"], diagnostics)
                if clean["completion_status"] == "completed"
                else None
            )
            measurement = {
                "run_id": run["id"],
                "benchmark_id": job_id,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "created_at": now(),
            }
            db.execute(
                "INSERT INTO evaluation_run_measurements VALUES (?,?,?)",
                (run["id"], job_id, canonical(measurement)),
            )
            current.update(
                status=(
                    "completed"
                    if clean["completion_status"] == "completed"
                    else "cancelled" if clean["error_reason"] == "cancelled" else "failed"
                ),
                run_id=run["id"],
                error_code=clean["error_reason"],
                updated_at=now(),
            )
            db.execute(
                "UPDATE evaluation_benchmark_items SET data=? WHERE id=?", (canonical(current), item["id"])
            )
        return run["id"]

    def retry_item(self, job_id, attempt, worker):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job, owner, expires = self._job(db, job_id)
            if owner != worker or not expires or expires <= time.time() or job["cancel_requested"]:
                raise ConflictError("Benchmark lease expired or cancellation requested")
            row = db.execute(
                "SELECT data FROM evaluation_benchmark_items WHERE id=? AND benchmark_id=?",
                (attempt["id"], job_id),
            ).fetchone()
            if not row:
                raise KeyError(attempt["id"])
            item = json.loads(row[0])
            if item["status"] != "failed" or item["attempts"] != attempt["attempts"]:
                raise ConflictError("Only the current failed attempt can be retried")
            item["status"] = "pending"
            db.execute(
                "UPDATE evaluation_benchmark_items SET data=? WHERE id=?", (canonical(item), attempt["id"])
            )

    def release(self, job_id, worker, status, error=None):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job, owner, _ = self._job(db, job_id)
            if owner != worker:
                return
            job.update(status=status, error_code=error)
            self._save(db, job)
            db.execute(
                "UPDATE evaluation_benchmarks SET lease_owner=NULL,lease_expires=NULL WHERE id=?", (job_id,)
            )

    def measurements(self, run_id):
        with self.store.connect() as db:
            row = db.execute(
                "SELECT data FROM evaluation_run_measurements WHERE run_id=?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None


class BenchmarkManager:
    def __init__(self, jobs, executor, persist=lambda: None, retry_delay=1):
        self.jobs, self.executor, self.persist = jobs, executor, persist
        self.retry_delay = retry_delay
        self.tasks, self.cancels = {}, {}
        self.worker = str(uuid4())

    async def _db(self, method, *args):
        return await asyncio.to_thread(method, *args)

    async def start(self, job_id, confirm_live=False):
        if not confirm_live:
            raise ValueError("Explicit confirm_live is required before starting or resuming live execution")
        if job_id in self.tasks and not self.tasks[job_id].done():
            raise ConflictError("Benchmark is already running")
        job = await self._db(self.jobs.claim, job_id, self.worker)
        await asyncio.to_thread(self.persist)
        self.cancels[job_id] = threading.Event()
        self.tasks[job_id] = asyncio.create_task(self._run(job_id))
        return job

    async def cancel(self, job_id):
        result = await self._db(self.jobs.request_cancel, job_id)
        if job_id in self.cancels:
            self.cancels[job_id].set()
        await asyncio.to_thread(self.persist)
        return result

    async def wait(self, job_id):
        if job_id in self.tasks:
            await self.tasks[job_id]

    async def _run(self, job_id):
        cancel = self.cancels[job_id]
        status, error = "completed", None
        pulse = None
        try:
            job = await self._db(self.jobs.get, job_id)
            snapshot = await self._db(self.jobs.store.get_snapshot, job["snapshot_id"])
            cases = {q["id"]: q for q in snapshot["cases"]}
            options = job["options"]
            last_started = None
            retries = {}

            async def heartbeat():
                while not cancel.is_set():
                    try:
                        stop = await self._db(self.jobs.heartbeat, job_id, self.worker)
                    except Exception:
                        cancel.set()
                        return
                    if stop:
                        cancel.set()
                        return
                    await asyncio.sleep(1)

            pulse = asyncio.create_task(heartbeat())
            while not cancel.is_set():
                item = await self._db(self.jobs.next_item, job_id, self.worker)
                if not item:
                    break
                if last_started is not None:
                    delay = max(0, 60 / options["requests_per_minute"] - (time.monotonic() - last_started))
                    if delay:
                        await self._sleep(delay, cancel)
                if cancel.is_set():
                    break
                last_started = time.monotonic()
                call = asyncio.create_task(
                    self.executor.execute(
                        cases[item["question_id"]]["question"], job["configuration"], cancel
                    )
                )
                try:
                    deadline = time.monotonic() + options["timeout_seconds"]
                    while not call.done() and not cancel.is_set() and time.monotonic() < deadline:
                        await asyncio.wait({call}, timeout=min(0.1, max(0, deadline - time.monotonic())))
                    if call.done():
                        result = call.result()
                    else:
                        call.cancel()
                        await asyncio.gather(call, return_exceptions=True)
                        result = {
                            "completion_status": "failed",
                            "error_reason": "cancelled" if cancel.is_set() else "timeout",
                            "latency_ms": (time.monotonic() - last_started) * 1000,
                        }
                except asyncio.CancelledError:
                    call.cancel()
                    await asyncio.gather(call, return_exceptions=True)
                    raise
                except Exception:
                    result = {"completion_status": "failed", "error_reason": "execution_error"}
                await self._db(self.jobs.finish_attempt, job_id, item, self.worker, result)
                await asyncio.to_thread(self.persist)
                retryable = result.get("completion_status") != "completed" and result.get("error_reason") in (
                    "rate_limit",
                    "request_limit",
                    "timeout",
                    "upstream_error",
                    "interrupted",
                    "execution_error",
                )
                attempt = retries.get(item["id"], 0)
                if retryable and attempt < options["max_retries"] and not cancel.is_set():
                    retries[item["id"]] = attempt + 1
                    await self._sleep(min(30, self.retry_delay * 2**attempt), cancel)
                    if not cancel.is_set():
                        await self._db(self.jobs.retry_item, job_id, item, self.worker)
            if cancel.is_set():
                status = "cancelled"
        except asyncio.CancelledError:
            cancel.set()
            status, error = "paused", "worker_stopped"
        except Exception:
            status, error = "paused", "worker_error"
        finally:
            if pulse:
                pulse.cancel()
                await asyncio.gather(pulse, return_exceptions=True)
            await self._db(self.jobs.release, job_id, self.worker, status, error)
            await asyncio.to_thread(self.persist)

    async def _sleep(self, seconds, cancel):
        until = time.monotonic() + seconds
        while not cancel.is_set() and time.monotonic() < until:
            await asyncio.sleep(min(0.1, max(0, until - time.monotonic())))

    async def shutdown(self):
        for cancel in self.cancels.values():
            cancel.set()
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
