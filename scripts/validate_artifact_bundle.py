from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artifact_contract import validate_artifact_lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a canonical deployment bundle locally without GPU or Modal."
    )
    parser.add_argument(
        "bundle",
        nargs="?",
        default="artifacts/vn_history_deployment",
        help="Canonical bundle produced by training.scripts.export_artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.bundle).expanduser().resolve()
    try:
        lock = validate_artifact_lock(root)
    except Exception as exc:
        print(f"ARTIFACT_BUNDLE_INVALID root={root}\n{exc}", file=sys.stderr)
        return 1
    summary = {
        "valid": True,
        "root": str(root),
        "deployment_id": lock["deployment_id"],
        "roles": sorted(lock.get("roles") or {}),
        "central": lock.get("central") is not None,
        "corpus_count": lock["corpus"]["count"],
        "faiss_count": lock["faiss"]["ntotal"],
        "bm25_count": lock["bm25"]["count"],
    }
    print("ARTIFACT_BUNDLE_VALID " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
