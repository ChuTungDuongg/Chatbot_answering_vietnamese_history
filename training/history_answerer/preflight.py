from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from app.agents.common.model_registry import SHARED_BASE_MODEL_ID
from training.common.datasets import first_user_assistant, split_rows
from training.common.jsonl import read_jsonl
from training.history_answerer.config import Phase6Config
from training.history_answerer.loss import build_rag_training_example_with_stats
from training.history_answerer.validate_dataset import validate_rows


def _summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values) if values else 0,
        "mean": statistics.mean(values) if values else 0.0,
        "max": max(values) if values else 0,
    }


def tokenization_report(
    rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int, split_name: str
) -> dict[str, Any]:
    raw_prompt: list[int] = []
    final_prompt: list[int] = []
    assistant: list[int] = []
    sequence: list[int] = []
    overlength = capped = zero_supervised = 0
    errors: list[str] = []
    for index, row in enumerate(rows, 1):
        try:
            user_text, assistant_text = first_user_assistant(row)
            feature, stats = build_rag_training_example_with_stats(
                tokenizer, user_text, assistant_text, max_length=max_length
            )
            supervised = sum(label != -100 for label in feature["labels"])
            zero_supervised += int(supervised == 0)
            raw_prompt.append(stats.prompt_tokens)
            final_prompt.append(stats.prompt_tokens_kept)
            assistant.append(stats.assistant_tokens)
            sequence.append(stats.sequence_tokens)
            overlength += int(stats.truncated)
            capped += int(stats.prompt_truncated)
            if stats.assistant_truncated:
                errors.append(f"{split_name} row {index}: assistant target was truncated")
        except (TypeError, ValueError) as exc:
            errors.append(f"{split_name} row {index}: {exc}")
    if zero_supervised:
        errors.append(f"{split_name}: {zero_supervised} rows have zero supervised tokens")
    return {
        "rows": len(rows),
        "raw_prompt_tokens": _summary(raw_prompt),
        "final_prompt_tokens": _summary(final_prompt),
        "assistant_tokens": _summary(assistant),
        "sequence_tokens": _summary(sequence),
        "overlength_rows": overlength,
        "context_capped_rows": capped,
        "zero_supervised_rows": zero_supervised,
        "errors": errors,
        "valid": not errors,
    }


def run_preflight(dataset: str | Path, tokenizer: Any, *, max_length: int = 4096) -> dict[str, Any]:
    rows = read_jsonl(dataset)
    validation = validate_rows(rows)
    cfg = Phase6Config()
    splits = split_rows(
        rows,
        seed=42,
        group_key="group_id",
        stratify_key="type",
        train_ratio=cfg.train_ratio,
        eval_ratio=cfg.eval_ratio,
    )
    reports = {
        name: tokenization_report(getattr(splits, name), tokenizer, max_length=max_length, split_name=name)
        for name in ("train", "eval", "test")
    }
    groups = {
        name: {str(row.get("group_id") or row["id"]) for row in getattr(splits, name)}
        for name in ("train", "eval", "test")
    }
    overlap = {
        "train_eval": len(groups["train"] & groups["eval"]),
        "train_test": len(groups["train"] & groups["test"]),
        "eval_test": len(groups["eval"] & groups["test"]),
    }
    split_types = {
        name: {
            row_type: sum(row.get("type") == row_type for row in getattr(splits, name))
            for row_type in sorted({str(row.get("type")) for row in getattr(splits, name)})
        }
        for name in reports
    }
    all_types = sorted({str(row.get("type")) for row in rows})
    groups_per_type = {
        row_type: len({str(row.get("group_id") or row["id"]) for row in rows if row.get("type") == row_type})
        for row_type in all_types
    }
    split_missing_types = {
        name: [row_type for row_type in all_types if not split_types[name].get(row_type)]
        for name in reports
    }
    classes_unable_in_both_holdouts = [
        row_type for row_type, count in groups_per_type.items() if count < 2
    ]
    errors = list(validation.get("errors", []))
    errors.extend(error for report in reports.values() for error in report["errors"])
    if any(overlap.values()):
        errors.append(f"group leakage: {overlap}")
    avoidable_missing = {
        name: [
            row_type
            for row_type in missing
            if row_type not in classes_unable_in_both_holdouts
        ]
        for name, missing in split_missing_types.items()
        if name in {"eval", "test"}
    }
    if any(avoidable_missing.values()):
        errors.append(f"avoidable History behavior coverage gaps: {avoidable_missing}")
    return {
        "valid": not errors,
        "dataset": str(Path(dataset)),
        "model_id": SHARED_BASE_MODEL_ID,
        "max_length": max_length,
        "validation": validation,
        "splits": reports,
        "split_rows": {name: len(getattr(splits, name)) for name in reports},
        "split_groups": {name: len(value) for name, value in groups.items()},
        "split_type_distribution": split_types,
        "split_missing_types": split_missing_types,
        "groups_per_type": groups_per_type,
        "classes_unable_in_both_holdouts": classes_unable_in_both_holdouts,
        "group_overlap": overlap,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="History tokenization preflight; never loads model weights.")
    parser.add_argument("--dataset", default="datasets/history_answerer/train.jsonl")
    parser.add_argument("--tokenizer-id", default=SHARED_BASE_MODEL_ID)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--report", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    report = run_preflight(args.dataset, tokenizer, max_length=args.max_length)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
