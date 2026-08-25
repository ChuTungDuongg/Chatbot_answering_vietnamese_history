from __future__ import annotations

import argparse
from pathlib import Path

from training.common.jsonl import read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2/3 corpus builder from chunk JSONL packs.")
    parser.add_argument("--input-dir", default="training/Dataset/Chunk_id")
    parser.add_argument("--output", default="artifacts/corpus/vn_history_rag_chunks.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = []
    for path in sorted(Path(args.input_dir).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    seen = set()
    unique = []
    for row in rows:
        chunk_id = str(row.get("chunk_id", "")).strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(row)
    print(f"Wrote {write_jsonl(args.output, unique)} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



