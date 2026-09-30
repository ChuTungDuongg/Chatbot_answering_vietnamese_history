"""Checks for deploy dependencies and the committed six-way summary."""

import csv
import importlib.metadata
import json
from pathlib import Path

from evaluation.six_way import SYSTEM_NAMES, write_summary_csv


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_peft_dependency_is_installed_without_training_stack():
    lines = {line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()}
    assert "peft==0.20.0" in lines
    assert importlib.metadata.version("peft") == "0.20.0"
    assert not any(line.startswith(("trl", "datasets", "bitsandbytes")) for line in lines)


def test_six_way_csv_surfaces_committed_metrics_without_inference(tmp_path):
    root = ROOT / "artifacts/evaluation/reports/six_way_best_b4_ga4_e2"
    summaries = {mode: json.loads((root / mode / "metrics.json").read_text(encoding="utf-8"))
                 for mode in SYSTEM_NAMES}
    output = tmp_path / "summary.csv"
    write_summary_csv(output, summaries)
    assert output.read_text(encoding="utf-8") == (root / "summary.csv").read_text(encoding="utf-8")
    with output.open(newline="", encoding="utf-8") as stream:
        rows = {row["system"]: row for row in csv.DictReader(stream)}
    assert rows["vanilla_faiss"]["answer_citation_coverage"] == "0.4"
    assert rows["sft_faiss"]["answer_citation_coverage"] == "0.0"
    assert rows["sft_faiss"]["citation_precision"] == ""
    assert rows["sft_faiss"]["chunk_hit_rate@10"] == rows["vanilla_faiss"]["chunk_hit_rate@10"]
