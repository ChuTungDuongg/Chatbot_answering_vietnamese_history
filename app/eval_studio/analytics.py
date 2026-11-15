"""Paired run measurements; heuristic, human and judge results stay distinct."""

import json
from itertools import combinations
from statistics import mean

METRICS = {
    "tools.success_rate": "higher",
    "tools.argument_validity_rate": "higher",
    "tools.repeated_call_rate": "lower",
    "citation_metrics.citation_validity_rate": "higher",
    "grounding_metrics.required_fact_phrase_recall": "higher",
    "answer_metrics.token_f1": "higher",
    "checks.required_tools": "higher",
    "checks.forbidden_tools": "higher",
    "checks.any_tool_groups": "higher",
    "checks.allowed_tools": "higher",
    "checks.expected_exit_reason": "higher",
    "checks.output_format": "higher",
    "checks.evidence_ids": "higher",
    "checks.evidence_terms": "higher",
    "checks.citations": "higher",
    "human_score": "higher",
    "judge_score": "higher",
    "latency_ms": "lower",
    "input_tokens": "lower",
    "output_tokens": "lower",
    "tool_calls": "lower",
    "cost": "lower",
}


def at(value, path):
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value if type(value) in (int, float, bool) else None


def average(values):
    known = [v for v in values if v is not None]
    return round(mean(known), 6) if known else None


def compare_benchmarks(jobs, experiment_ids, benchmark_ids=None, judge_profile_id=None):
    if len(experiment_ids) < 2 or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Select at least two distinct experiments")
    with jobs.store.connect() as db:
        db.execute("BEGIN")
        all_jobs = [
            json.loads(r[0]) for r in db.execute("SELECT data FROM evaluation_benchmarks ORDER BY rowid DESC")
        ]
        selected = {}
        if benchmark_ids is not None:
            if len(benchmark_ids) != len(set(benchmark_ids)):
                raise ValueError("Duplicate benchmark IDs")
            available = {j["id"]: j for j in all_jobs}
            for id in benchmark_ids:
                if id not in available:
                    raise KeyError(id)
                job = available[id]
                if job["experiment_id"] not in experiment_ids or job["experiment_id"] in selected:
                    raise ValueError("Choose exactly one benchmark per experiment")
                selected[job["experiment_id"]] = job
        else:
            for job in all_jobs:
                if job["experiment_id"] in experiment_ids:
                    selected.setdefault(job["experiment_id"], job)
        if set(selected) != set(experiment_ids):
            raise ValueError("Each selected experiment needs a benchmark")
        if len({j["snapshot_id"] for j in selected.values()}) != 1:
            raise ValueError("Benchmarks use different immutable snapshots")
        version = selected[experiment_ids[0]]["snapshot_id"]
        snapshot = json.loads(
            db.execute("SELECT data FROM dataset_versions WHERE version=?", (version,)).fetchone()[0]
        )
        experiments = {
            r[0]: json.loads(r[1]) for r in db.execute("SELECT id,data FROM evaluation_experiments")
        }
        items = {
            j["id"]: [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM evaluation_benchmark_items WHERE benchmark_id=?", (j["id"],)
                )
            ]
            for j in selected.values()
        }
        measurements = {
            r[0]: json.loads(r[1]) for r in db.execute("SELECT run_id,data FROM evaluation_run_measurements")
        }
        runs = {r[0]: json.loads(r[1]) for r in db.execute("SELECT id,data FROM evaluation_runs")}
        reviews = {}
        for response_id, record in db.execute(
            "SELECT response_id,data FROM human_reviews ORDER BY rowid DESC"
        ):
            reviews.setdefault(response_id, json.loads(record))
        judge_scores = {}
        if judge_profile_id:
            if (
                not db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='evaluation_judge_profiles'"
                ).fetchone()
                or not db.execute(
                    "SELECT 1 FROM evaluation_judge_profiles WHERE id=?", (judge_profile_id,)
                ).fetchone()
            ):
                raise KeyError(judge_profile_id)
            for run_id, record in db.execute(
                "SELECT run_id,data FROM evaluation_judge_scores WHERE profile_id=? ORDER BY rowid DESC",
                (judge_profile_id,),
            ):
                judge_scores.setdefault(run_id, json.loads(record))
    summaries, cells = [], {}
    for id in experiment_ids:
        job = selected[id]
        per_question = {}
        for item in items[job["id"]]:
            run = runs.get(item["run_id"])
            if item["status"] != "completed" or not run:
                continue
            measurement = measurements.get(run["id"], {})
            human = reviews.get(run["response_id"])
            judge = judge_scores.get(run["id"])
            metrics = measurement.get("metrics") or {}
            values = {path: at(metrics, path) for path in METRICS}
            values.update(
                {k: run.get(k) for k in ("latency_ms", "input_tokens", "output_tokens", "tool_calls", "cost")}
            )
            values.update(
                human_score=human["weighted_score"] if human else None,
                judge_score=judge["weighted_score"] if judge else None,
            )
            per_question[item["question_id"]] = {
                "run_id": run["id"],
                "metrics": metrics,
                "values": values,
                "human_review": human,
                "judge_score": judge,
            }
        cells[id] = per_question
        attempts = [
            runs[run_id]
            for run_id, m in measurements.items()
            if m["benchmark_id"] == job["id"] and run_id in runs
        ]

        def avg_metric(key):
            return average([c["values"].get(key) for c in per_question.values()])

        summaries.append(
            {
                "id": id,
                "name": experiments[id]["name"],
                "benchmark_id": job["id"],
                "status": job["status"],
                "configuration": job["configuration"],
                "configuration_hash": job["configuration_hash"],
                "total_questions": len(items[job["id"]]),
                "evaluated_questions": len(per_question),
                "failed_questions": sum(i["status"] == "failed" for i in items[job["id"]]),
                "failed_runs": sum(r["completion_status"] != "completed" for r in attempts),
                "missing_measurements": sum(not c["metrics"] for c in per_question.values()),
                "metrics": {path: avg_metric(path) for path in METRICS if "." in path},
                **{
                    k: avg_metric(k)
                    for k in ("latency_ms", "input_tokens", "output_tokens", "tool_calls", "cost")
                },
                "quality": {"human_mean": avg_metric("human_score"), "judge_mean": avg_metric("judge_score")},
                "attempt_totals": {
                    k: (
                        sum(r[k] for r in attempts)
                        if attempts and all(r.get(k) is not None for r in attempts)
                        else None
                    )
                    for k in ("input_tokens", "output_tokens", "cost")
                },
            }
        )
    pairwise = []
    for a, b in combinations(experiment_ids, 2):
        shared = set(cells[a]) & set(cells[b])
        for metric, direction in METRICS.items():
            pairs = [
                (cells[a][qid]["values"][metric], cells[b][qid]["values"][metric])
                for qid in shared
                if cells[a][qid]["values"][metric] is not None and cells[b][qid]["values"][metric] is not None
            ]
            deltas = [float(right) - float(left) for left, right in pairs]
            wins = sum(left > right if direction == "higher" else left < right for left, right in pairs)
            ties = sum(left == right for left, right in pairs)
            pairwise.append(
                {
                    "a": a,
                    "b": b,
                    "metric": metric,
                    "paired": len(pairs),
                    "wins": wins,
                    "ties": ties,
                    "losses": len(pairs) - wins - ties,
                    "mean_delta": average(deltas),
                    "direction": direction,
                }
            )
    qids = {i["question_id"] for values in items.values() for i in values}
    rows = [
        {"question": q, "cells": {id: cells[id].get(q["id"]) for id in experiment_ids}}
        for q in snapshot["cases"]
        if q["id"] in qids
    ]
    config_keys = (
        "provider",
        "model_id",
        "chat_mode",
        "reasoning_effort",
        "planning_reasoning_effort",
        "dense_backend",
        "final_k",
        "prompt_hash",
        "tool_hash",
        "tool_allowlist",
        "runtime_provenance",
    )
    differences = [
        k
        for k in config_keys
        if len({json.dumps(selected[id]["configuration"].get(k), sort_keys=True) for id in experiment_ids})
        > 1
    ]
    return {
        "snapshot_id": version,
        "comparable": True,
        "configuration_differences": differences,
        "experiments": summaries,
        "pairwise": pairwise,
        "metric_directions": METRICS,
        "judge_profile_id": judge_profile_id,
        "rows": rows,
        "policy": "Latest selected benchmark; one completed final attempt per question. Missing observations stay null. Positive mean_delta is B minus A; metric direction determines regression. Heuristics are not historical correctness.",
    }
