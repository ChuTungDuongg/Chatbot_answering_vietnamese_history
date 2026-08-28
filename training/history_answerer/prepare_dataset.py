from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common.datasets import load_messages
from training.common.jsonl import write_jsonl
from training.history_answerer.validate_dataset import validate_rows


def build_rows(input_path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in load_messages(input_path):
        if not validate_rows([row])["valid"]:
            continue
        item = dict(row)
        item["group_id"] = f"history-answer-{row.get('id')}"
        item["original_sample_id"] = row.get("id")
        item["source_dataset"] = "vn_history_phase6"
        rows.append(item)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build validated canonical Phase-6 History Answerer chat data.")
    parser.add_argument("--input", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output", default="datasets/history_answerer/train.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_rows = load_messages(args.input)
    rows = build_rows(args.input)
    count = write_jsonl(args.output, rows)
    report = validate_rows(rows)
    print(json.dumps({
        "source_rows": len(source_rows),
        "rows": count,
        "excluded_invalid_rows": len(source_rows) - count,
        "validation": report,
    }, ensure_ascii=False, sort_keys=True))
    if not report["valid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



