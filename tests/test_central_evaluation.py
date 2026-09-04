import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from evaluation.io import load_run, read_jsonl
from evaluation.metrics import calculate_metrics
from evaluation.recording import from_result
from evaluation.runners.compare import compare_runs, write_reports
from evaluation.runners.run import main, run_questions
from evaluation.schema import Annotation, EvaluationRecord, Question, RunMetadata

ROOT = Path(__file__).resolve().parents[1]


def metadata(variant="base", **updates):
    values = dict(run_id=variant, variant=variant, timestamp=datetime.now(timezone.utc), git_commit="fixture-commit",
        model_id="Qwen/Qwen3-8B", model_revision="fixture-revision", adapter_enabled=variant == "adapted",
        adapter_path="/artifacts/adapters/central-v2" if variant == "adapted" else None,
        adapter_sha256="fixture-adapter-hash" if variant == "adapted" else None,
        dataset_version="fixture", dataset_sha256="fixture-dataset-hash", retrieval_index_sha256="fixture-index",
        prompt_sha256="fixture-prompts", generation_settings={"max_new_tokens": 100}, retrieval_settings={"top_k": 3},
        tools=["search_history"], context_budgets={"max_chars": 2000}, host_config={"mode": "central"},
        seed=42, hardware_class="fixture-cpu", environment={"python": "fixture"})
    return RunMetadata(**{**values, **updates})


def record(variant="base", question_id="q1", **signals):
    return EvaluationRecord(run_id=variant, variant=variant, question_id=question_id,
        question="Fixture question " + question_id, category="fixture", signals=signals,
        adapter_loaded=variant == "adapted", adapter_configured=variant == "adapted")


def value(groups, group, name):
    return groups[group][name]["value"]


def test_schema_fixture_validation_and_invalid_data(tmp_path):
    rows = read_jsonl(ROOT / "evaluation/datasets/fixtures/questions.jsonl", Question)
    assert len(rows) == 2
    with pytest.raises(ValidationError):
        Question(id="q", question="", category="fixture")
    with pytest.raises(ValidationError):
        metadata("base", adapter_path="/bad")
    with pytest.raises(ValidationError):
        metadata("adapted", adapter_sha256=None)
    path = tmp_path / "duplicate.jsonl"
    path.write_text(rows[0].model_dump_json() + "\n" + rows[0].model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_jsonl(path, Question)


def test_missing_metrics_are_null_in_every_group_and_empty_counts_not_perfect():
    groups = calculate_metrics([record()])
    assert len(groups) == 7
    assert all(metric["value"] is None for group in groups.values() for metric in group.values())
    zero = calculate_metrics([record(valid_citations=0, total_citations=0)])
    assert value(zero, "citations", "citation_validity_rate") is None


def test_known_metrics_keep_denominators_and_conditional_eligibility():
    rows = [record(success=True, repair=True, valid_citations=2, invalid_citations=1, total_citations=3,
                   tool_calls=2, successful_tools=1, checked_tools=2, latency_ms=100, cold_start=True, legacy_calls=0),
            record(question_id="q2", success=False, repair=False, valid_citations=1, invalid_citations=0,
                   total_citations=1, latency_ms=200, cold_start=False, legacy_calls=0),
            record(question_id="q3")]
    groups = calculate_metrics(rows)
    assert value(groups, "reliability", "final_answer_success_rate") == .5
    assert groups["reliability"]["final_answer_success_rate"]["observed_records"] == 2
    assert value(groups, "reliability", "repair_success_rate") == 1
    assert value(groups, "citations", "citation_validity_rate") == .75
    assert value(groups, "behavior", "tool_call_success_rate") == .5
    assert value(groups, "behavior", "legacy_role_agent_call_count") == 0
    assert value(groups, "efficiency", "median_latency_ms") == 150
    assert value(groups, "efficiency", "p95_latency_ms") == 200
    assert value(groups, "efficiency", "cold_mean_latency_ms") == 100
    assert value(groups, "efficiency", "warm_mean_latency_ms") == 200


def test_annotations_are_required_for_truth_and_false_positive_metrics():
    row = record(tool_calls=2, viewpoint_flagged=True, partial=True)
    assert value(calculate_metrics([row]), "viewpoint", "viewpoint_false_positive_rate") is None
    row.annotation = Annotation(reviewer="fixture reviewer", method="human", rubric_version="v1",
        historical_correctness=.5, unnecessary_tool_calls=1, viewpoint_should_flag=False, partial_answer_correct=True)
    groups = calculate_metrics([row])
    assert value(groups, "viewpoint", "viewpoint_false_positive_rate") == 1
    assert value(groups, "behavior", "unnecessary_tool_call_rate") == .5
    assert value(groups, "answer_quality", "historical_correctness") == .5
    assert value(groups, "answer_quality", "partial_answer_correctness") == 1


def test_pairing_deltas_use_common_eligible_questions_and_zero_base_is_not_infinite():
    base = [record(success=False, latency_ms=200), record(question_id="q2", success=True)]
    adapted = [record("adapted", success=True, latency_ms=100), record("adapted", "q2", latency_ms=10)]
    report = compare_runs(metadata(), base, metadata("adapted"), adapted)
    success = report["groups"]["reliability"]["final_answer_success_rate"]
    assert success["paired_observations"] == 1
    assert success["absolute_delta"] == 1
    assert success["relative_delta"] is None
    assert report["groups"]["efficiency"]["mean_latency_ms"]["absolute_delta"] == -100


@pytest.mark.parametrize("field,changed", [("git_commit", "other"), ("seed", 43), ("retrieval_index_sha256", "other"),
    ("tools", ["other"]), ("prompt_sha256", "other"), ("host_config", {"mode": "different"}),
    ("dataset_sha256", "other"), ("generation_settings", {"max_new_tokens": 999}),
    ("graph_topology_fingerprint", "other")])
def test_fairness_rejects_mismatched_host_inputs(field, changed):
    with pytest.raises(ValueError, match="unfair"):
        compare_runs(metadata(), [record()], metadata("adapted", **{field: changed}), [record("adapted")])


def test_pairing_rejects_missing_questions_silent_base_and_unknown_hardware():
    with pytest.raises(ValueError, match="same nonempty"):
        compare_runs(metadata(), [record()], metadata("adapted"), [record("adapted", "q2")])
    adapted = record("adapted", success=True)
    adapted.adapter_loaded = False
    with pytest.raises(ValueError, match="adapter state"):
        compare_runs(metadata(), [record()], metadata("adapted"), [adapted])
    with pytest.raises(ValueError, match="hardware"):
        compare_runs(metadata(), [record()], metadata("adapted", hardware_class=None), [record("adapted")])
    report = compare_runs(metadata(), [record(latency_ms=1)], metadata("adapted", hardware_class=None),
                          [record("adapted", latency_ms=2)], compare_latency=False)
    assert report["groups"]["efficiency"]["mean_latency_ms"]["absolute_delta"] is None


def test_trace_recording_preserves_raw_metadata_and_unknown_denominators():
    raw = {"status": "ok", "answer": "Fixture answer [1]", "final_failure_reason": None,
        "analysis": {"actors": ["Alpha", "Beta"]}, "source_chunks": [{"text": "fixture", "target_consistent": True}],
        "central_debug": {"tools": [{"name": "search_history", "error": None}], "central_adapter_configured": False,
            "central_adapter_loaded": False, "required_actor_coverage": ["Alpha", "Beta"], "selected_actor_coverage": {"Alpha": ["s1"], "Beta": []},
            "uncited_factual_paragraphs": 1, "citation_paragraph_classifications": [{"kind": "new_factual_claim", "paragraph": 2}]},
        "performance_debug": {"central_model_calls": 1, "central_tool_calls": 1, "central_input_tokens": 50,
            "central_output_tokens": 10, "research_generation_calls": 0, "evidence_generation_calls": 0, "history_generation_calls": 0}}
    row = from_result(Question(id="q1", question="fixture?", category="fixture"), metadata(), raw)
    assert row.raw_result == raw
    assert row.signals["actor_coverage"] == .5
    assert row.signals["one_generation_success"] is True
    assert value(calculate_metrics([row]), "citations", "uncited_factual_paragraph_rate") is None


def test_fake_runner_roundtrip_reports_and_error_records(tmp_path):
    class FakeCentral:
        async def run(self, *, question, **kwargs):
            assert kwargs["history"] == []
            if "two" in question:
                raise RuntimeError("fixture failure")
            return {"answer": "fixture", "status": "ok", "final_failure_reason": None,
                    "central_debug": {"central_adapter_loaded": False, "central_adapter_configured": False}}
    questions = [Question(id="one", question="one?", category="fixture"), Question(id="two", question="two?", category="fixture")]
    output = tmp_path / "base"
    asyncio.run(run_questions(FakeCentral(), questions, metadata(), output))
    meta, rows = load_run(output)
    assert len(rows) == 2 and rows[1].status == "runner_error"
    assert rows[1].signals.get("input_tokens") is None
    adapted = [row.model_copy(update={"run_id": "adapted", "variant": "adapted", "adapter_configured": True, "adapter_loaded": True}) for row in rows]
    report = compare_runs(meta, rows, metadata("adapted"), adapted)
    write_reports(report, tmp_path / "reports")
    assert (tmp_path / "reports/comparison.json").is_file()
    assert "N/A" in (tmp_path / "reports/comparison.md").read_text()
    assert "base_pass" in (tmp_path / "reports/per_question.csv").read_text()
    with pytest.raises(FileExistsError):
        asyncio.run(run_questions(FakeCentral(), questions, meta, output))


def test_default_cli_is_validation_only_without_importing_production_or_models():
    script = "from evaluation.runners.run import main; import sys; main(['--variant','base']); assert 'app.main' not in sys.modules; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"executed": false' in result.stdout


def test_generated_outputs_ignored_and_placeholders_source_fixtures_trackable():
    for path in ("evaluation/logs/future/records.jsonl", "evaluation/reports/future/comparison.json", "training/central/data/generated/train.jsonl"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0
    for path in ("evaluation/logs/.gitkeep", "evaluation/reports/.gitkeep", "evaluation/datasets/fixtures/questions.jsonl",
                 "evaluation/metrics/specs.py", "training/central/normalization/hermes.py"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 1
