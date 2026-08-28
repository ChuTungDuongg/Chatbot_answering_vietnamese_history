from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from training.common.datasets import first_user_assistant
from training.common.jsonl import read_jsonl
from training.history_answerer.evaluate import parse_source_ids


CONTEXT_ID_RE = re.compile(r"(?m)^\s*\[([^\]\r\n]+)\]")
EMBEDDING_FIELD_RE = re.compile(r'(?i)["\']?(?:embedding|embeddings|dense_vector)["\']?\s*[:=]')
VECTOR_RE = re.compile(r"\[(?:\s*-?\d+(?:\.\d+)?\s*,){15,}\s*-?\d+(?:\.\d+)?\s*\]")


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    ids: Counter[str] = Counter()
    types: Counter[str] = Counter()
    invalid_source_ids = 0
    hallucinated_gold_citations = 0
    embedding_leakage = 0
    citation_aware = 0
    grounded = 0

    for index, row in enumerate(rows, 1):
        label = f"row {index}"
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            errors.append(f"{label}: missing id/group key")
        ids[row_id] += 1
        row_type = str(row.get("type") or "unknown")
        types[row_type] += 1
        try:
            user_text, assistant_text = first_user_assistant(row)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        context_ids = set(CONTEXT_ID_RE.findall(user_text))
        gold_ids = set(parse_source_ids(assistant_text))
        citation_aware += int("Nguồn được dùng:" in assistant_text and "Trả lời:" in assistant_text)
        grounded += int(bool(gold_ids) and gold_ids <= context_ids)
        unknown = sorted(gold_ids - context_ids)
        if unknown:
            invalid_source_ids += len(unknown)
            hallucinated_gold_citations += 1
            errors.append(f"{label}: gold cites IDs absent from the input evidence: {unknown}")
        if EMBEDDING_FIELD_RE.search(user_text) or VECTOR_RE.search(user_text):
            embedding_leakage += 1
            errors.append(f"{label}: embedding/vector data leaked into the model prompt")
        if not assistant_text.strip():
            errors.append(f"{label}: empty assistant target")

    if not rows:
        errors.append("dataset is empty")
    return {
        "valid": not errors,
        "rows": len(rows),
        "unique_groups": len([value for value in ids if value]),
        "type_distribution": dict(types),
        "grounded_rows": grounded,
        "citation_aware_rows": citation_aware,
        "invalid_source_ids": invalid_source_ids,
        "hallucinated_gold_citations": hallucinated_gold_citations,
        "embedding_leakage": embedding_leakage,
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate grounded History Answerer Phase-6 data.")
    parser.add_argument("--dataset", default="datasets/history_answerer/train.jsonl")
    parser.add_argument("--report", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_rows(read_jsonl(args.dataset))
    except (OSError, ValueError) as exc:
        report = {"valid": False, "errors": [str(exc)], "warnings": []}
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
