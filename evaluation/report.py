"""Aggregate deterministic per-question scores without a composite accuracy."""
from __future__ import annotations

import math
import statistics
from typing import Any

from evaluation.metrics.answer import score_answer
from evaluation.metrics.citations import score_citations
from evaluation.metrics.grounding import score_grounding
from evaluation.metrics.retrieval import score_retrieval


def _means(rows: list[dict[str, Any]], template: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    keys = sorted({key for row in [template, *rows] for key, value in row.items()
                   if isinstance(value, (float, int)) or value is None})
    result: dict[str, dict[str, float | int | None]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (float, int))
                  and not isinstance(row[key], bool) and math.isfinite(row[key])]
        result[key] = {"value": statistics.mean(values) if values else None,
                       "observed": len(values)}
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("success") is True]
    count = len(records)
    retrieval_template = score_retrieval([], relevant_chunk_ids=None, relevant_source_ids=None)
    return {
        "schema_version": 1,
        "prediction_count": count,
        "success_count": len(successful),
        "success_rate": len(successful) / count if count else None,
        "retrieval": {
            level: _means([record["retrieval"][level] for record in successful], retrieval_template[level])
            for level in ("chunk", "source")
        },
        "answer": _means([record["answer_metrics"] for record in successful], score_answer("", None)),
        "grounding": _means([record["grounding"] for record in successful], score_grounding("", "", [])),
        "citations": _means([record["citations"] for record in successful], score_citations("", [])),
    }


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def render_markdown(summary: dict[str, Any], *, dataset: str, predictions: str) -> str:
    lines = ["# Baseline evaluation", "", f"Dataset: `{dataset}`  ",
             f"Predictions: `{predictions}`  ",
             f"Model: `{summary['provenance']['model_id'] or 'N/A'}`  ",
             f"Predictions: {summary['prediction_count']}; success rate: {_fmt(summary['success_rate'])}", "",
             f"Dataset questions: {summary['dataset_question_count']}; predicted questions: {summary['predicted_question_count']}",
             f"Missing question IDs: {', '.join(summary['missing_question_ids']) or 'none'}", "",
             "N/A means the required label or observation was unavailable. Surface similarity does not establish historical truth.", ""]
    for title, metrics in (("Retrieval: chunks", summary["retrieval"]["chunk"]),
                           ("Retrieval: sources", summary["retrieval"]["source"]),
                           ("Answer similarity", summary["answer"]),
                           ("Grounding checks", summary["grounding"]),
                           ("Citations", summary["citations"])):
        lines += [f"## {title}", "", "| Metric | Mean | Observed |", "|---|---:|---:|"]
        for name, item in metrics.items():
            lines.append(f"| {name} | {_fmt(item['value'])} | {item['observed']} |")
        lines.append("")
    return "\n".join(lines) + "\n"
