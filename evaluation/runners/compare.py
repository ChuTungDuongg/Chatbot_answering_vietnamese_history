"""Paired, offline comparison from saved JSONL; never loads a model."""
import argparse
import csv
from pathlib import Path

from evaluation.io import load_run, write_json
from evaluation.metrics import calculate_metrics

FAIR_FIELDS = ("git_commit", "model_id", "model_revision", "dataset_version", "dataset_sha256",
               "retrieval_index_sha256", "prompt_sha256", "generation_settings", "retrieval_settings",
               "tools", "context_budgets", "host_config", "seed", "environment")


def compare_runs(base_meta, base, adapted_meta, adapted, *, compare_latency=True):
    if base_meta.variant != "base" or adapted_meta.variant != "adapted":
        raise ValueError("expected BASE then ADAPTED runs")
    mismatches = [name for name in FAIR_FIELDS if getattr(base_meta, name) != getattr(adapted_meta, name)]
    if mismatches:
        raise ValueError("unfair paired configuration: " + ", ".join(mismatches))
    if compare_latency and (not base_meta.hardware_class or base_meta.hardware_class != adapted_meta.hardware_class):
        raise ValueError("latency comparison requires matching known hardware_class; use --without-latency")
    left, right = {row.question_id: row for row in base}, {row.question_id: row for row in adapted}
    if len(left) != len(base) or len(right) != len(adapted) or not left or set(left) != set(right):
        raise ValueError("paired runs require the same nonempty question IDs without duplicates")
    for rows, meta in ((base, base_meta), (adapted, adapted_meta)):
        for row in rows:
            if row.run_id != meta.run_id or row.variant != meta.variant:
                raise ValueError("record run/variant mismatch")
            # Load failures remain in the comparison. Successful answers must
            # prove the requested variant actually ran, never silently use BASE.
            if row.signals.get("success") and (row.adapter_configured is not meta.adapter_enabled or row.adapter_loaded is not meta.adapter_enabled):
                raise ValueError("successful record has missing or mismatched adapter state")
            resolved = row.raw_result.get("central_debug", {}).get("central_model_snapshot_resolved")
            if resolved and Path(resolved).name != meta.model_revision:
                raise ValueError("runtime model revision differs from run metadata")
    for key in left:
        if (left[key].question, left[key].category) != (right[key].question, right[key].category):
            raise ValueError(f"question/category differs for {key}")
    # Cache per-question eligibility; delta aggregates use the SAME eligible pairs.
    scores = {key: (calculate_metrics([left[key]]), calculate_metrics([right[key]])) for key in left}
    groups = {}
    for group, metrics in calculate_metrics(base).items():
        groups[group] = {}
        for name, definition in metrics.items():
            eligible = [key for key, pair in scores.items() if all(side[group][name]["value"] is not None for side in pair)]
            if not compare_latency and "latency_ms" in name:
                eligible = []
            b = calculate_metrics([left[key] for key in eligible])[group][name]
            a = calculate_metrics([right[key] for key in eligible])[group][name]
            bv, av = b["value"], a["value"]
            delta = av - bv if av is not None and bv is not None else None
            groups[group][name] = {"base_value": bv, "adapted_value": av, "absolute_delta": delta,
                "relative_delta": delta / abs(bv) if delta is not None and bv != 0 else None,
                "paired_observations": len(eligible), "base_denominator": b["denominator"], "adapted_denominator": a["denominator"],
                "direction": definition["direction"], "definition": definition["definition"]}
    per_question = [{"question_id": key, "base_pass": left[key].signals.get("success"),
        "adapted_pass": right[key].signals.get("success"), "base_status": left[key].status, "adapted_status": right[key].status,
        "preference": right[key].annotation.preference if right[key].annotation else None} for key in left]
    return {"base_run_id": base_meta.run_id, "adapted_run_id": adapted_meta.run_id,
            "paired_questions": len(left), "latency_comparable": compare_latency, "groups": groups,
            "per_question": per_question, "base_observed_metrics": calculate_metrics(base),
            "adapted_observed_metrics": calculate_metrics(adapted)}


def write_reports(report, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "comparison.json", report)
    lines = ["# Paired Central comparison", "", f"BASE: `{report['base_run_id']}`; ADAPTED: `{report['adapted_run_id']}`.",
             "", "Deltas use common eligible question pairs. Unknown fields are N/A. Host heuristics do not establish historical truth.", ""]
    show = lambda value: "N/A" if value is None else f"{value:.4g}"
    for group, metrics in report["groups"].items():
        lines += [f"## {group}", "", "| Metric | Base | Adapted | Delta | Relative delta | Pairs |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {name} | {show(row['base_value'])} | {show(row['adapted_value'])} | {show(row['absolute_delta'])} | {show(row['relative_delta'])} | {row['paired_observations']} |"
                  for name, row in metrics.items()]
        lines.append("")
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    with (root / "per_question.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["per_question"][0]))
        writer.writeheader()
        writer.writerows(report["per_question"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-latency", action="store_true")
    args = parser.parse_args(argv)
    base_meta, base = load_run(args.base)
    adapted_meta, adapted = load_run(args.adapted)
    write_reports(compare_runs(base_meta, base, adapted_meta, adapted, compare_latency=not args.without_latency), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
