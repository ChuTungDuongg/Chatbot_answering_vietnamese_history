"""CLI for inspecting, building, sampling, and auditing Corpus V1."""

import argparse
import json
from pathlib import Path
import sys

from scripts.corpus_v1.audit import audit
from scripts.corpus_v1.pipeline import build
from scripts.corpus_v1.source import inspect, preset


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Offline Corpus V1 builder")
    actions = command.add_subparsers(dest="action", required=True)
    for name in ("inspect-source", "build"):
        item = actions.add_parser(name)
        item.add_argument("--preset", choices=("uvw-2026",))
        item.add_argument("--dataset-id")
        item.add_argument("--dataset-config")
        item.add_argument("--dataset-revision")
        item.add_argument("--split")
        item.add_argument("--cache-dir")
        for field in ("title", "text", "id", "url"):
            item.add_argument(f"--{field}-field")
        item.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=None)
        if name == "inspect-source":
            item.add_argument("--examples", type=int, default=3)
        else:
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--resume", action="store_true")
            item.add_argument("--chunk-tokens", type=int, default=384)
            item.add_argument("--chunk-overlap", type=int, default=48)
            item.add_argument("--tokenizer-id", default="intfloat/multilingual-e5-base")
            item.add_argument("--max-records-per-split", type=int)
            item.add_argument("--shard-size", type=int, default=10000)
            item.add_argument("--scratch-dir", type=Path)
            item.add_argument("--offline-fixture-dir", type=Path,
                              help="Test-only local JSONL source; requires a pilot limit")
    item = actions.add_parser("audit")
    item.add_argument("--corpus", type=Path, required=True)
    item = actions.add_parser("sample")
    item.add_argument("--corpus", type=Path, required=True)
    item.add_argument("--decision", choices=("kept", "review", "dropped"), default="review")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action in ("inspect-source", "build"):
        cfg = preset(args.preset) if args.preset else {}
        dataset_id = args.dataset_id or cfg.get("dataset_id")
        if not dataset_id:
            raise ValueError("Pass --preset or --dataset-id")
        cfg.update(dataset_id=dataset_id,
                   dataset_config=args.dataset_config or cfg.get("dataset_config"),
                   requested_revision=args.dataset_revision or cfg.get("requested_revision") or "main",
                   split=args.split or cfg.get("split") or "train",
                   cache_dir=args.cache_dir,
                   streaming=True if args.streaming is None else args.streaming)
        fields = dict(cfg.get("fields", {}))
        fields.update({name: getattr(args, name + "_field") for name in ("title", "text", "id", "url")
                       if getattr(args, name + "_field") is not None})
        cfg["fields"] = fields
        if args.action == "inspect-source":
            result = inspect(cfg, args.examples)
        else:
            cfg.update(chunk_tokens=args.chunk_tokens, chunk_overlap=args.chunk_overlap,
                       tokenizer_id=args.tokenizer_id,
                       max_records_per_split=args.max_records_per_split,
                       shard_size=args.shard_size,
                       scratch_dir=str(args.scratch_dir) if args.scratch_dir else None)
            if args.offline_fixture_dir:
                if not args.max_records_per_split:
                    raise ValueError("Offline fixture builds require --max-records-per-split")
                fixture = args.offline_fixture_dir.resolve()
                info = json.loads((fixture / "metadata.json").read_text(encoding="utf-8"))
                cfg["offline_fixture_dir"] = str(fixture)

                def source_factory(split):
                    with (fixture / f"{split}.jsonl").open(encoding="utf-8") as stream:
                        for line in stream:
                            yield json.loads(line)

                result = build(cfg, args.output, resume=args.resume, source_info=info,
                               source_factory=source_factory,
                               token_counter=lambda value: len(value.split()))
            else:
                result = build(cfg, args.output, resume=args.resume)
    elif args.action == "audit":
        result = audit(args.corpus)
    else:
        root = args.corpus if args.corpus.is_dir() else args.corpus.parent
        path = root / "samples" / (args.decision + ".jsonl")
        result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
