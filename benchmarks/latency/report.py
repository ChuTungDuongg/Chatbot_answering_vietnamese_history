"""Human and machine readable benchmark output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def render_markdown(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines = ["# HTTP SSE latency baseline", "", f"Run ID: `{metadata['run_id']}`  ",
             f"Mode: `{metadata['mode']}`  ", f"Concurrency: `{metadata['concurrency']}`", "",
             "Metrics use client receipt times or explicit server telemetry. N/A means unobserved.", ""]
    for phase, group in summary["groups"].items():
        lines += [f"## {phase.title()}", "",
                  f"Requests: {group['count']}; success: {_fmt(group['success_rate'])}; error: {_fmt(group['error_rate'])}", "",
                  "| Metric | Observed | Mean | Std | p50 | p95 | p99 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, values in group["metrics"].items():
            lines.append("| " + name + " | " + " | ".join(_fmt(values[key]) for key in
                         ("count", "mean", "std", "p50", "p95", "p99")) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"
