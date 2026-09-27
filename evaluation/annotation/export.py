"""Export only human-accepted, validated records as Question schema v1 JSONL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.validate_dataset import validate_records, workspace_records
from evaluation.annotation.workspace import Workspace, dumps


def export(workspace: Workspace, lookup: CorpusLookup, output: Path, *,
           tier: str, split: str | None = None) -> int:
    if output.exists():
        raise FileExistsError(output)
    records = workspace_records(workspace, tier)
    if split:
        records = [record for record in records if record.get("benchmark_split") == split]
    if not records:
        raise ValueError("No accepted records for this tier and split")
    result = validate_records(records, lookup, tier=tier)
    if result["errors"]:
        raise RuntimeError("Gold validation failed:\n" + "\n".join(result["errors"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace"))
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--tier", choices=("retrieval", "deep"), required=True)
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = Workspace(args.workspace)
    lookup = CorpusLookup(args.corpus, args.workspace / "corpus_lookup.sqlite3")
    try:
        count = export(workspace, lookup, args.output, tier=args.tier, split=args.split)
        print(f"exported {count} reviewed {args.tier} questions to {args.output}")
        return 0
    finally:
        workspace.close()
        lookup.close()


if __name__ == "__main__":
    raise SystemExit(main())
