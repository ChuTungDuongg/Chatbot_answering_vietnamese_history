from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.common.hf_cache import hf_cache_status, seed_hf_cache
from app.agents.common.model_registry import CENTRAL_BASE_MODEL_ID, SHARED_BASE_MODEL_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed or validate the Hugging Face inference cache.")
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="HF model ID to check/seed. Repeatable. Defaults to Qwen3-8B Central.",
    )
    parser.add_argument("--cache-dir", default=None, help="Hub cache directory. Production uses /hf-cache/hub.")
    parser.add_argument("--include-shared-4b", action="store_true", help="Also include the shared 4B role base.")
    parser.add_argument("--validate-only", action="store_true", help="Do not download; only report cache hit/miss.")
    parser.add_argument("--local-files-only", action="store_true", help="Pass local_files_only=True to the seeding call.")
    return parser


def _models(args: argparse.Namespace) -> list[str]:
    models = list(args.model_id or [CENTRAL_BASE_MODEL_ID])
    if args.include_shared_4b:
        models.append(SHARED_BASE_MODEL_ID)
    return list(dict.fromkeys(models))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports: list[dict[str, Any]] = []
    ok = True
    for model_id in _models(args):
        if args.validate_only:
            report = hf_cache_status(model_id, cache_dir=args.cache_dir)
        else:
            report = seed_hf_cache(
                model_id,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            )
        reports.append(report)
        ok = ok and bool(report.get("cache_hit"))
    print(json.dumps({"ok": ok, "models": reports}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
