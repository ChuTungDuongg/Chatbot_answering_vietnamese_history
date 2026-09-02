"""Explicit local-file preparation; never downloads or generates model answers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.central.normalization.hermes import normalize_hermes_function_calling
from training.central.normalization.viquad import normalize_uit_viquad2
from training.central.normalization.validation import require_v2_trajectory
from training.central.mixing.mix import DEFAULT_CONFIG, load_mix_config, mix_v2
from training.trajectory_dataset.dedup import deduplicate
from training.trajectory_dataset.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl
from training.trajectory_dataset.split import split_trajectories


def normalize_rows(rows, *, source, source_file="func-calling.json", split="train"):
    normalized, rejected = [], []
    for index, row in enumerate(rows):
        try:
            if source == "hermes":
                item = normalize_hermes_function_calling(row, index=index, split=split, source_file=source_file)
            elif source == "viquad":
                item = normalize_uit_viquad2(row, index=index, split=split)
            else:
                raise ValueError("source must be hermes or viquad")
            normalized.append(require_v2_trajectory(item))
        except ValueError as exc:
            rejected.append({"index": index, "id": row.get("id"), "reason": str(exc)})
    return normalized, rejected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize")
    normalize.add_argument("--source", required=True, choices=("hermes", "viquad"))
    normalize.add_argument("--input", type=Path, required=True, help="Local JSONL export; no Hub download")
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--source-file", default="func-calling.json")
    normalize.add_argument("--split", default="train", choices=("train", "validation", "test"))
    mix = commands.add_parser("mix")
    mix.add_argument("--hermes", type=Path, required=True)
    mix.add_argument("--viquad", type=Path, required=True)
    mix.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mix.add_argument("--output-dir", type=Path, required=True)
    mix.add_argument("--seed", type=int, default=42)
    mix.add_argument("--max-total", type=int)
    args = parser.parse_args(argv)
    if args.command == "normalize":
        rows, rejected = normalize_rows(read_jsonl(args.input), source=args.source, source_file=args.source_file, split=args.split)
        atomic_write_jsonl(args.output, rows)
        atomic_write_json(args.output.with_suffix(".rejections.json"), rejected)
        print(json.dumps({"accepted": len(rows), "rejected": len(rejected)}))
    else:
        pools = {"hermes_function_calling": read_jsonl(args.hermes), "uit_viquad2_grounded": read_jsonl(args.viquad)}
        # Deduplicate BEFORE mixing; split whole source groups to prevent leakage.
        unique = deduplicate([row for rows in pools.values() for row in rows])
        rows = unique.rows
        mixed = mix_v2({name: [row for row in rows if row["source_dataset"] == name] for name in pools},
                       config_path=args.config, seed=args.seed, max_total=args.max_total)
        splits = split_trajectories(mixed, seed=args.seed)
        for name in ("train", "validation", "test"):
            atomic_write_jsonl(args.output_dir / f"{name}.jsonl", getattr(splits, name))
        atomic_write_json(args.output_dir / "preparation.json", {"seed": args.seed, "mix_config": load_mix_config(args.config),
                          "samples": len(mixed), "splits": {name: len(getattr(splits, name)) for name in ("train", "validation", "test")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
