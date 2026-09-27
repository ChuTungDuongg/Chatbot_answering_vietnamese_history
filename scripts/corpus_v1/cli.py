"""CLI for inspecting, building, sampling, and auditing Corpus V1."""

import argparse
import json
from pathlib import Path

from scripts.corpus_v1.audit import audit
from scripts.corpus_v1.pipeline import build
from scripts.corpus_v1.source import inspect


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Offline Corpus V1 builder")
    actions = command.add_subparsers(dest="action", required=True)
    for name in ("inspect-source", "build"):
        item = actions.add_parser(name)
        item.add_argument("--dataset-id", required=True)
        item.add_argument("--dataset-config")
        item.add_argument("--dataset-revision")
        item.add_argument("--split", default="train")
        item.add_argument("--cache-dir")
        for field in ("title", "text", "id", "url"):
            item.add_argument(f"--{field}-field")
        item.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
        if name == "inspect-source":
            item.add_argument("--examples", type=int, default=3)
        else:
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--resume", action="store_true")
            item.add_argument("--chunk-tokens", type=int, default=384)
            item.add_argument("--chunk-overlap", type=int, default=48)
            item.add_argument("--tokenizer-id", default="intfloat/multilingual-e5-base")
    item = actions.add_parser("audit")
    item.add_argument("--corpus", type=Path, required=True)
    item = actions.add_parser("sample")
    item.add_argument("--corpus", type=Path, required=True)
    item.add_argument("--decision", choices=("kept", "review", "dropped"), default="review")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action in ("inspect-source", "build"):
        cfg = {"dataset_id": args.dataset_id, "dataset_config": args.dataset_config,
               "dataset_revision": args.dataset_revision, "split": args.split,
               "cache_dir": args.cache_dir, "streaming": args.streaming,
               "fields": {name: getattr(args, name + "_field") for name in ("title", "text", "id", "url")}}
        if args.action == "inspect-source":
            result = inspect(cfg, args.examples)
        else:
            cfg.update(chunk_tokens=args.chunk_tokens, chunk_overlap=args.chunk_overlap,
                       tokenizer_id=args.tokenizer_id)
            result = build(cfg, args.output, resume=args.resume)
    elif args.action == "audit":
        result = audit(args.corpus)
    else:
        root = args.corpus if args.corpus.is_dir() else args.corpus.parent
        path = root / "samples" / (args.decision + ".jsonl")
        result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
