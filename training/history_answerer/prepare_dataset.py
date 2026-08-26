from __future__ import annotations

import argparse
from pathlib import Path

from training.common.datasets import first_user_assistant, load_messages
from training.common.jsonl import write_jsonl


def build_rows(input_path: str | Path) -> list[dict[str, object]]:
    rows = []
    for row in load_messages(input_path):
        user_text, assistant_text = first_user_assistant(row)
        rows.append(
            {
                "id": row.get("id"),
                "type": row.get("type"),
                "user": user_text,
                "assistant": assistant_text,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize Phase 6 RAG-SFT chat data.")
    parser.add_argument("--input", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output", default="artifacts/training/history_answerer/messages_normalized.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    count = write_jsonl(args.output, build_rows(args.input))
    print(f"Wrote {count} normalized rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



