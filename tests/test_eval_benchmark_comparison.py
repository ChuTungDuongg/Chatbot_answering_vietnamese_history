import asyncio

import pytest

from app.eval_studio.store import EvalStore
from app.eval_studio.v11 import StudioV11


def test_automated_comparison_pairs_snapshot_questions_and_keeps_human_separate(tmp_path):
    from app.eval_studio.analytics import compare_benchmarks
    from app.eval_studio.benchmark import BenchmarkStore

    store = EvalStore(tmp_path / "compare.sqlite")
    q = store.questions()[0]
    store.save_question(
        {
            **q,
            "question": "A real fixture question",
            "expected_behavior": {"required_tools": ["search_history"]},
        }
    )
    store.snapshot("one")
    studio = StudioV11(store)
    jobs = BenchmarkStore(studio)
    exps = [studio.create_experiment({"name": name, "snapshot_id": "one"}) for name in ("A", "B")]
    for index, exp in enumerate(exps):
        config = {
            "experiment_id": exp["id"],
            "snapshot_id": "one",
            "configuration_hash": f"config{index}",
            "prompt_hash": f"prompt{index}",
            "tool_hash": "tools",
        }
        job = jobs.create({"experiment_id": exp["id"], "question_ids": [q["id"]]}, config)
        jobs.claim(job["id"], "worker")
        item = jobs.next_item(job["id"], "worker")
        diagnostics = {
            "capture": {"tool_trace": True},
            "tool_calls": [
                {
                    "name": "search_history",
                    "status": "completed" if index == 0 else "failed",
                    "arguments_valid": True,
                    "argument_fingerprint": "x",
                }
            ],
        }
        jobs.finish_attempt(
            job["id"],
            item,
            "worker",
            {"answer": "answer", "completion_status": "completed", "diagnostics": diagnostics},
        )
        jobs.release(job["id"], "worker", "completed")
    result = compare_benchmarks(jobs, [e["id"] for e in exps])
    assert result["snapshot_id"] == "one"
    assert result["experiments"][0]["quality"]["human_mean"] is None
    assert result["experiments"][0]["latency_ms"] is None
    pair = next(p for p in result["pairwise"] if p["metric"] == "tools.success_rate")
    assert (pair["paired"], pair["wins"], pair["losses"]) == (1, 1, 0)
    assert "prompt_hash" in result["configuration_differences"]
    store.snapshot("two")
    other = studio.create_experiment({"name": "Different", "snapshot_id": "two"})
    config = {
        "experiment_id": other["id"],
        "snapshot_id": "two",
        "configuration_hash": "other",
        "prompt_hash": "other",
        "tool_hash": "tools",
    }
    jobs.create({"experiment_id": other["id"], "question_ids": [q["id"]]}, config)
    with pytest.raises(ValueError, match="snapshot"):
        compare_benchmarks(jobs, [exps[0]["id"], other["id"]])
