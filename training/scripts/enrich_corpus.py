from __future__ import annotations

import argparse
import re

from training.common.jsonl import read_jsonl, write_jsonl


YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def enrich(row):
    text = str(row.get("text", ""))
    years = sorted({int(match.group(0)) for match in YEAR_RE.finditer(text)})
    out = dict(row)
    out.setdefault("metadata", {})
    out["metadata"] = {**out.get("metadata", {}), "years": years, "char_len": len(text)}
    out.setdefault("history_score", 1.0)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 8 lightweight corpus metadata enrichment.")
    parser.add_argument("--input", default="artifacts/corpus/vn_history_rag_chunks.jsonl")
    parser.add_argument("--output", default="artifacts/corpus/vn_history_rag_chunks_enriched.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Wrote {write_jsonl(args.output, [enrich(row) for row in read_jsonl(args.input)])} enriched chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



