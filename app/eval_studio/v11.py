"""Additive experiment and human evaluation storage, independent of inference."""

import json
import re
import secrets
from collections import Counter
from itertools import combinations
from statistics import mean
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from urllib.parse import urlsplit
from .models import Review, Scores, now
from .store import ConflictError, canonical
from .privacy import reject_secrets


class ExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=300)
    snapshot_id: str
    description: str = ""
    provider: str | None = None
    model_id: str | None = None
    chat_mode: str | None = None
    reasoning_effort: str | None = None
    planning_reasoning_effort: str | None = None
    dense_backend: str | None = None
    final_k: int | None = Field(default=None, ge=1)
    tool_allowlist: list[str] | None = None
    mcp_config: str | None = None
    prompt_hash: str | None = None
    git_commit: str | None = None
    corpus_fingerprint: str | None = None
    runtime_environment: str | None = None
    status: str = "active"

    @model_validator(mode="after")
    def no_credentials(self):
        reject_secrets(self.model_dump())
        return self


class EvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    chunk_id: str | None = None
    title: str | None = None
    source_kind: str | None = None
    source_url: str | None = None
    excerpt: str | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    final_rank: int | None = None
    selected: bool | None = None
    cited: bool | None = None

    @field_validator("source_url")
    @classmethod
    def safe_source_url(cls, value):
        if value:
            parts = urlsplit(value)
            if (
                parts.scheme not in ("http", "https")
                or not parts.hostname
                or parts.username
                or parts.password
            ):
                raise ValueError("Source URL must be an HTTP(S) URL without credentials")
        return value


class ToolTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str
    status: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    error_code: str | None = None


class TraceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    retrieval_backend: str | None = None
    final_context_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    tools: list[ToolTrace] = Field(default_factory=list)


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    experiment_id: str
    question_id: str
    answer: str = ""
    external_id: str | None = Field(default=None, max_length=200)
    completion_status: str = "completed"
    error_reason: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    evidence: list[EvidenceInput] = Field(default_factory=list)
    trace_captured: bool = False
    trace: TraceInput | None = None

    @model_validator(mode="after")
    def no_credentials(self):
        reject_secrets(self.model_dump())
        return self


class BlindSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scores_a: Scores
    scores_b: Scores
    preference: str
    notes: str = ""
    reviewer: str = ""


class StudioV11:
    def __init__(self, store):
        self.store = store
        with store.connect() as db:
            db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS evaluation_experiments (id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, data TEXT NOT NULL, FOREIGN KEY(snapshot_id) REFERENCES dataset_versions(version));
                CREATE TABLE IF NOT EXISTS evaluation_runs (id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, question_id TEXT NOT NULL, response_id TEXT, external_id TEXT, data TEXT NOT NULL, FOREIGN KEY(experiment_id) REFERENCES evaluation_experiments(id), FOREIGN KEY(response_id) REFERENCES evaluation_responses(id), UNIQUE(experiment_id, external_id));
                CREATE TABLE IF NOT EXISTS evaluation_evidence (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, data TEXT NOT NULL, FOREIGN KEY(run_id) REFERENCES evaluation_runs(id));
                CREATE TABLE IF NOT EXISTS evaluation_blind_sessions (id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, experiment_a TEXT NOT NULL, experiment_b TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(snapshot_id) REFERENCES dataset_versions(version), FOREIGN KEY(experiment_a) REFERENCES evaluation_experiments(id), FOREIGN KEY(experiment_b) REFERENCES evaluation_experiments(id));
                CREATE TABLE IF NOT EXISTS evaluation_blind_assignments (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, question_id TEXT NOT NULL, run_a TEXT NOT NULL, run_b TEXT NOT NULL, FOREIGN KEY(session_id) REFERENCES evaluation_blind_sessions(id), FOREIGN KEY(run_a) REFERENCES evaluation_runs(id), FOREIGN KEY(run_b) REFERENCES evaluation_runs(id), UNIQUE(session_id, question_id));
                CREATE TABLE IF NOT EXISTS evaluation_blind_reviews (assignment_id TEXT PRIMARY KEY, data TEXT NOT NULL, FOREIGN KEY(assignment_id) REFERENCES evaluation_blind_assignments(id));
                CREATE TABLE IF NOT EXISTS evaluation_import_batches (id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, result TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS evaluation_experiments_snapshot_idx ON evaluation_experiments(snapshot_id);
                CREATE INDEX IF NOT EXISTS evaluation_runs_question_idx ON evaluation_runs(question_id);
                CREATE INDEX IF NOT EXISTS evaluation_runs_response_idx ON evaluation_runs(response_id);
                CREATE INDEX IF NOT EXISTS evaluation_evidence_run_idx ON evaluation_evidence(run_id);
                CREATE INDEX IF NOT EXISTS evaluation_blind_assignments_runs_idx ON evaluation_blind_assignments(run_a,run_b);
            """)

    def experiments(self):
        with self.store.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute("SELECT data FROM evaluation_experiments ORDER BY rowid DESC")
            ]

    def get_experiment(self, id):
        with self.store.connect() as db:
            row = db.execute("SELECT data FROM evaluation_experiments WHERE id=?", (id,)).fetchone()
        if not row:
            raise KeyError(id)
        return json.loads(row[0])

    def create_experiment(self, data):
        item = ExperimentInput.model_validate(data).model_dump()
        self.store.get_snapshot(item["snapshot_id"])
        item.update(id=str(uuid4()), created_at=now())
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO evaluation_experiments VALUES (?, ?, ?)",
                (item["id"], item["snapshot_id"], canonical(item)),
            )
        return item

    def update_experiment(self, id, data):
        clean = {k: v for k, v in data.items() if k not in ("id", "created_at", "updated_at")}
        item = ExperimentInput.model_validate(clean).model_dump()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM evaluation_experiments WHERE id=?", (id,)).fetchone()
            if not row:
                raise KeyError(id)
            old = json.loads(row[0])
            if not db.execute(
                "SELECT 1 FROM dataset_versions WHERE version=?", (item["snapshot_id"],)
            ).fetchone():
                raise KeyError(item["snapshot_id"])
            if db.execute("SELECT 1 FROM evaluation_runs WHERE experiment_id=?", (id,)).fetchone():
                raise ConflictError("Experiment configuration is locked after the first run")
            item.update(id=id, created_at=old["created_at"], updated_at=now())
            db.execute(
                "UPDATE evaluation_experiments SET snapshot_id=?, data=? WHERE id=?",
                (item["snapshot_id"], canonical(item), id),
            )
        return item

    def delete_experiment(self, id):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT 1 FROM evaluation_runs WHERE experiment_id=?", (id,)).fetchone()
                or db.execute(
                    "SELECT 1 FROM evaluation_blind_sessions WHERE experiment_a=? OR experiment_b=?", (id, id)
                ).fetchone()
            ):
                raise ConflictError("Experiment has recorded runs or blind sessions")
            if not db.execute("DELETE FROM evaluation_experiments WHERE id=?", (id,)).rowcount:
                raise KeyError(id)

    def update_run(self, id, data):
        item = RunInput.model_validate(data).model_dump()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data,response_id FROM evaluation_runs WHERE id=?", (id,)).fetchone()
            if not row:
                raise KeyError(id)
            old = json.loads(row[0])
            if (
                item["experiment_id"],
                item["question_id"],
                item["completion_status"],
                item["external_id"],
            ) != (old["experiment_id"], old["question_id"], old["completion_status"], old["external_id"]):
                raise ConflictError("Run identity and status cannot be changed")
            if db.execute(
                "SELECT 1 FROM evaluation_blind_assignments WHERE run_a=? OR run_b=?", (id, id)
            ).fetchone() or (
                row[1] and db.execute("SELECT 1 FROM human_reviews WHERE response_id=?", (row[1],)).fetchone()
            ):
                raise ConflictError("Run has reviews or blind assignments")
            if item["completion_status"] == "completed" and not item["answer"].strip():
                raise ValueError("Completed run requires answer")
            if row[1]:
                response = json.loads(
                    db.execute("SELECT data FROM evaluation_responses WHERE id=?", (row[1],)).fetchone()[0]
                )
                response["answer"] = item["answer"]
                db.execute("UPDATE evaluation_responses SET data=? WHERE id=?", (canonical(response), row[1]))
            evidence = item.pop("evidence")
            item.update(
                id=id,
                response_id=row[1],
                snapshot_id=old["snapshot_id"],
                created_at=old["created_at"],
                updated_at=now(),
            )
            db.execute("UPDATE evaluation_runs SET data=? WHERE id=?", (canonical(item), id))
            db.execute("DELETE FROM evaluation_evidence WHERE run_id=?", (id,))
            for chunk in evidence:
                db.execute(
                    "INSERT INTO evaluation_evidence VALUES (?, ?, ?)", (str(uuid4()), id, canonical(chunk))
                )
        return item

    def delete_run(self, id):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT response_id FROM evaluation_runs WHERE id=?", (id,)).fetchone()
            if not row:
                raise KeyError(id)
            if db.execute(
                "SELECT 1 FROM evaluation_blind_assignments WHERE run_a=? OR run_b=?", (id, id)
            ).fetchone() or (
                row[0] and db.execute("SELECT 1 FROM human_reviews WHERE response_id=?", (row[0],)).fetchone()
            ):
                raise ConflictError("Run has reviews or blind assignments")
            db.execute("DELETE FROM evaluation_evidence WHERE run_id=?", (id,))
            db.execute("DELETE FROM evaluation_runs WHERE id=?", (id,))
            if row[0]:
                db.execute("DELETE FROM evaluation_responses WHERE id=?", (row[0],))

    def create_run(self, data):
        item = RunInput.model_validate(data).model_dump()
        if item["completion_status"] not in ("completed", "failed"):
            raise ValueError("completion_status must be completed or failed")
        if item["completion_status"] == "completed" and not item["answer"].strip():
            raise ValueError("Completed run requires answer")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT data FROM evaluation_experiments WHERE id=?", (item["experiment_id"],)
            ).fetchone()
            if not row:
                raise KeyError(item["experiment_id"])
            exp = json.loads(row[0])
            snapshot = json.loads(
                db.execute(
                    "SELECT data FROM dataset_versions WHERE version=?", (exp["snapshot_id"],)
                ).fetchone()[0]
            )
            if item["question_id"] not in {q["id"] for q in snapshot["cases"]}:
                raise ValueError("Question is not in the experiment snapshot")
            result = self._insert_run(db, item, exp)
        return result

    def _insert_run(self, db, item, exp):
        if item.get("external_id"):
            found = db.execute(
                "SELECT data FROM evaluation_runs WHERE experiment_id=? AND external_id=?",
                (item["experiment_id"], item["external_id"]),
            ).fetchone()
            if found:
                old = json.loads(found[0])
                comparable = {k: old.get(k) for k in item if k != "evidence"}
                if comparable != {k: v for k, v in item.items() if k != "evidence"}:
                    raise ConflictError("External ID already used with different content")
                evidence = [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT data FROM evaluation_evidence WHERE run_id=? ORDER BY rowid", (old["id"],)
                    )
                ]
                if evidence != item["evidence"]:
                    raise ConflictError("External ID already used with different evidence")
                return old
        if not db.execute("SELECT 1 FROM evaluation_questions WHERE id=?", (item["question_id"],)).fetchone():
            raise ValueError("Snapshot question is missing from the working dataset")
        run_id = str(uuid4())
        response_id = str(uuid4()) if item["completion_status"] == "completed" else None
        if response_id:
            response = {
                "id": response_id,
                "question_id": item["question_id"],
                "answer": item["answer"],
                "provider": exp["provider"] or "",
                "model": exp["model_id"] or "",
                "chat_mode": exp["chat_mode"] or "",
                "reasoning_effort": exp["reasoning_effort"] or "",
                "retrieval_config": "",
                "retrieved_evidence": "",
                "created_at": now(),
            }
            db.execute(
                "INSERT INTO evaluation_responses VALUES (?, ?, ?)",
                (response_id, item["question_id"], canonical(response)),
            )
        result = {
            **item,
            "id": run_id,
            "response_id": response_id,
            "snapshot_id": exp["snapshot_id"],
            "created_at": now(),
        }
        evidence = result.pop("evidence")
        db.execute(
            "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, exp["id"], item["question_id"], response_id, item["external_id"], canonical(result)),
        )
        for chunk in evidence:
            db.execute(
                "INSERT INTO evaluation_evidence VALUES (?, ?, ?)", (str(uuid4()), run_id, canonical(chunk))
            )
        return result

    def runs(self, experiment_id=None):
        with self.store.connect() as db:
            rows = (
                db.execute(
                    "SELECT data FROM evaluation_runs WHERE experiment_id=? ORDER BY rowid DESC",
                    (experiment_id,),
                )
                if experiment_id
                else db.execute("SELECT data FROM evaluation_runs ORDER BY rowid DESC")
            )
            return [json.loads(r[0]) for r in rows]

    def evidence(self, run_id):
        with self.store.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT data FROM evaluation_runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                raise KeyError(run_id)
            run = json.loads(row[0])
            chunks = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM evaluation_evidence WHERE run_id=? ORDER BY rowid", (run_id,)
                )
            ]
            snapshot = json.loads(
                db.execute(
                    "SELECT data FROM dataset_versions WHERE version=?", (run["snapshot_id"],)
                ).fetchone()[0]
            )
            reviews = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM human_reviews WHERE response_id=? ORDER BY rowid DESC",
                    (run["response_id"],),
                )
            ]
        question = next(q for q in snapshot["cases"] if q["id"] == run["question_id"])
        return {
            "run": run,
            "question": question,
            "evidence": chunks,
            "reviews": reviews,
            "trace_status": (
                "captured" if run.get("trace") is not None or run["trace_captured"] else "not_captured"
            ),
        }

    def compare(self, ids, category=None, difficulty=None, smoke=None, diagnosis=None):
        if len(set(ids)) != len(ids) or len(ids) < 2:
            raise ValueError("Select at least two distinct experiments")
        with self.store.connect() as db:
            db.execute("BEGIN")
            experiments = []
            for id in ids:
                row = db.execute("SELECT data FROM evaluation_experiments WHERE id=?", (id,)).fetchone()
                if not row:
                    raise KeyError(id)
                experiments.append(json.loads(row[0]))
            if len({e["snapshot_id"] for e in experiments}) != 1:
                raise ValueError("Experiments use different immutable snapshots")
            snapshot = json.loads(
                db.execute(
                    "SELECT data FROM dataset_versions WHERE version=?", (experiments[0]["snapshot_id"],)
                ).fetchone()[0]
            )
            reviews = [
                json.loads(r[0]) for r in db.execute("SELECT data FROM human_reviews ORDER BY rowid DESC")
            ]
            allruns = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM evaluation_runs WHERE experiment_id IN ("
                    + ",".join("?" for _ in ids)
                    + ") ORDER BY rowid DESC",
                    ids,
                )
            ]
        questions = [
            q
            for q in snapshot["cases"]
            if (not category or q["category"] == category)
            and (not difficulty or q["difficulty"] == difficulty)
            and (smoke is None or q["smoke"] == smoke)
        ]
        qids = {q["id"] for q in questions}
        latest_review = {}
        for r in reviews:
            latest_review.setdefault(r["response_id"], r)
        per = {}
        summaries = []
        for exp in experiments:
            runs = [r for r in allruns if r["experiment_id"] == exp["id"] and r["question_id"] in qids]
            chosen = {}
            for r in runs:
                if r["completion_status"] == "completed" and r["response_id"] in latest_review:
                    chosen.setdefault(r["question_id"], r)
            if diagnosis:
                chosen = {
                    qid: r
                    for qid, r in chosen.items()
                    if diagnosis in latest_review[r["response_id"]].get("diagnosis_tags", [])
                }
            per[exp["id"]] = chosen
            graded = [latest_review[r["response_id"]] for r in chosen.values()]

            def avg(values):
                return round(mean(values), 2) if values else None

            summaries.append(
                {
                    **exp,
                    "total_questions": len(questions),
                    "evaluated_questions": len(chosen),
                    "missing_reviews": len(questions) - len(chosen),
                    "failed_runs": sum(r["completion_status"] == "failed" for r in runs),
                    "mean_score": avg([r["weighted_score"] for r in graded]),
                    "criteria": {
                        k: avg([r["scores"][k] for r in graded])
                        for k in ("accuracy", "coverage", "reasoning", "evidence", "clarity")
                    },
                    "distribution": {
                        str(i): sum(min(4, int(r["weighted_score"] // 20)) == i for r in graded)
                        for i in range(5)
                    },
                    "diagnosis_counts": dict(Counter(t for r in graded for t in r.get("diagnosis_tags", []))),
                    "metadata_missing": [
                        k
                        for k in (
                            "provider",
                            "model_id",
                            "chat_mode",
                            "dense_backend",
                            "prompt_hash",
                            "corpus_fingerprint",
                        )
                        if not exp.get(k)
                    ],
                    "latency_ms": avg([r["latency_ms"] for r in runs if r["latency_ms"] is not None]),
                    "input_tokens": avg([r["input_tokens"] for r in runs if r["input_tokens"] is not None]),
                    "output_tokens": avg(
                        [r["output_tokens"] for r in runs if r["output_tokens"] is not None]
                    ),
                    "tool_calls": avg([r["tool_calls"] for r in runs if r["tool_calls"] is not None]),
                    "cost": avg([r["cost"] for r in runs if r["cost"] is not None]),
                }
            )
        pairwise = []
        for a, b in combinations(ids, 2):
            shared = set(per[a]) & set(per[b])
            wins = ties = losses = 0
            for qid in shared:
                left = latest_review[per[a][qid]["response_id"]]["weighted_score"]
                right = latest_review[per[b][qid]["response_id"]]["weighted_score"]
                wins += left > right
                losses += left < right
                ties += left == right
            pairwise.append(
                {"a": a, "b": b, "paired": len(shared), "wins": wins, "ties": ties, "losses": losses}
            )
        rows = []
        for q in questions:
            cells = {}
            for exp in experiments:
                run = per[exp["id"]].get(q["id"])
                cells[exp["id"]] = (
                    {"run_id": run["id"], "review": latest_review[run["response_id"]]} if run else None
                )
            rows.append({"question": q, "cells": cells})
        configuration_differences = [
            k
            for k in (
                "provider",
                "model_id",
                "chat_mode",
                "dense_backend",
                "final_k",
                "prompt_hash",
                "corpus_fingerprint",
                "reasoning_effort",
            )
            if len({str(e.get(k)) for e in experiments}) > 1
        ]
        return {
            "snapshot_id": experiments[0]["snapshot_id"],
            "comparable": True,
            "configuration_differences": configuration_differences,
            "experiments": summaries,
            "pairwise": pairwise,
            "paired_questions": len(set.intersection(*(set(per[id]) for id in ids))),
            "rows": rows,
        }

    def create_blind(self, snapshot_id, experiment_a, experiment_b):
        a, b = self.get_experiment(experiment_a), self.get_experiment(experiment_b)
        if a["id"] == b["id"] or a["snapshot_id"] != snapshot_id or b["snapshot_id"] != snapshot_id:
            raise ValueError("Experiments must be distinct and use the selected snapshot")
        ar, br = {}, {}
        for r in self.runs():
            if r["completion_status"] != "completed":
                continue
            if r["experiment_id"] == a["id"]:
                ar.setdefault(r["question_id"], r)
            if r["experiment_id"] == b["id"]:
                br.setdefault(r["question_id"], r)
        shared = sorted(set(ar) & set(br))
        if not shared:
            raise ValueError("No completed responses share a question")
        sid = str(uuid4())
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO evaluation_blind_sessions VALUES (?, ?, ?, ?, ?)",
                (sid, snapshot_id, a["id"], b["id"], now()),
            )
            for qid in shared:
                left, right = ar[qid]["id"], br[qid]["id"]
                if secrets.randbelow(2):
                    left, right = right, left
                db.execute(
                    "INSERT INTO evaluation_blind_assignments VALUES (?, ?, ?, ?, ?)",
                    (str(uuid4()), sid, qid, left, right),
                )
        return {"id": sid, "count": len(shared)}

    def blind_sessions(self):
        with self.store.connect() as db:
            return [
                {"id": r[0], "snapshot_id": r[1], "created_at": r[2]}
                for r in db.execute(
                    "SELECT id,snapshot_id,created_at FROM evaluation_blind_sessions ORDER BY rowid DESC"
                )
            ]

    def get_blind(self, session_id):
        with self.store.connect() as db:
            session = db.execute(
                "SELECT snapshot_id,experiment_a,experiment_b FROM evaluation_blind_sessions WHERE id=?",
                (session_id,),
            ).fetchone()
            if not session:
                raise KeyError(session_id)
            assignments = db.execute(
                "SELECT id,question_id,run_a,run_b FROM evaluation_blind_assignments WHERE session_id=? ORDER BY question_id",
                (session_id,),
            ).fetchall()
            done = {r[0] for r in db.execute("SELECT assignment_id FROM evaluation_blind_reviews")}
        exp = [self.get_experiment(session[1]), self.get_experiment(session[2])]
        identifiers = {s for e in exp for s in (e["model_id"], e["provider"], e["name"]) if s and len(s) >= 3}
        identifiers.update(["Luna", "Terra", "Qwen"])

        def scrub(text):
            for name in sorted(identifiers, key=len, reverse=True):
                text = re.sub(re.escape(name), "[ẩn danh]", text, flags=re.IGNORECASE)
            return text

        runs = {r["id"]: r for r in self.runs()}
        snapshot = self.store.get_snapshot(session[0])
        cases = {q["id"]: q for q in snapshot["cases"]}
        return {
            "id": session_id,
            "snapshot_id": session[0],
            "assignments": [
                {
                    "id": aid,
                    "question_id": qid,
                    "question": scrub(cases[qid]["question"]),
                    "reference_answer": scrub(cases[qid]["reference_answer"]),
                    "answer_a": scrub(runs[ra]["answer"]),
                    "answer_b": scrub(runs[rb]["answer"]),
                    "reviewed": aid in done,
                }
                for aid, qid, ra, rb in assignments
            ],
        }

    def submit_blind(self, session_id, assignment_id, data):
        item = BlindSubmission.model_validate(data)
        if item.preference not in ("a", "b", "tie"):
            raise ValueError("Invalid preference")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT run_a,run_b FROM evaluation_blind_assignments WHERE id=? AND session_id=?",
                (assignment_id, session_id),
            ).fetchone()
            if not row:
                raise KeyError(assignment_id)
            if db.execute(
                "SELECT 1 FROM evaluation_blind_reviews WHERE assignment_id=?", (assignment_id,)
            ).fetchone():
                raise ConflictError("Blind review already submitted")
            for run_id, scores in zip(row, (item.scores_a, item.scores_b)):
                response_id = db.execute(
                    "SELECT response_id FROM evaluation_runs WHERE id=?", (run_id,)
                ).fetchone()[0]
                review = Review(
                    response_id=response_id,
                    scores=scores,
                    decision="needs_review",
                    notes=item.notes,
                    reviewer=item.reviewer,
                )
                record = {
                    **review.model_dump(),
                    "id": str(uuid4()),
                    "created_at": now(),
                    "weighted_score": review.weighted_score(),
                }
                db.execute(
                    "INSERT INTO human_reviews VALUES (?, ?, ?)",
                    (record["id"], response_id, canonical(record)),
                )
            db.execute(
                "INSERT INTO evaluation_blind_reviews VALUES (?, ?)",
                (assignment_id, canonical({**item.model_dump(), "created_at": now()})),
            )
        return {"reviewed": True}

    def reveal_blind(self, session_id, assignment_id):
        with self.store.connect() as db:
            row = db.execute(
                "SELECT a.run_a,a.run_b,r.data FROM evaluation_blind_assignments a LEFT JOIN evaluation_blind_reviews r ON r.assignment_id=a.id WHERE a.id=? AND a.session_id=?",
                (assignment_id, session_id),
            ).fetchone()
        if not row:
            raise KeyError(assignment_id)
        if not row[2]:
            raise ConflictError("Review both responses before revealing identities")
        runs = {r["id"]: r for r in self.runs()}
        return {
            "a": self.get_experiment(runs[row[0]]["experiment_id"]),
            "b": self.get_experiment(runs[row[1]]["experiment_id"]),
            "review": json.loads(row[2]),
        }
