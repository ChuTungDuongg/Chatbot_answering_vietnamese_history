from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from training.common.jsonl import read_jsonl, write_jsonl
from training.research_agent.build_history_trajectories import build_no_tool_samples, build_trajectory_samples
from training.research_agent.converters import convert_agentinstruct, convert_xlam


HF_DATASETS = {
    "xlam": "Salesforce/xlam-function-calling-60k",
    "agentinstruct": "zai-org/AgentInstruct",
}


def _load_huggingface(source: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements-training.txt to load Hugging Face datasets.") from exc
    try:
        return load_dataset(HF_DATASETS[source], split=split)
    except Exception as exc:
        if source == "xlam":
            raise RuntimeError(
                "Could not load gated xLAM. Accept the dataset terms on Hugging Face, then run "
                "`huggingface-cli login` (or `hf auth login`) and retry."
            ) from exc
        raise RuntimeError(f"Could not load {HF_DATASETS[source]} split {split!r}: {exc}") from exc


def convert_rows(source: str, rows: Iterable[dict[str, Any]], *, source_split: str = "os") -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    stats = {"converted_count": 0, "skipped_count": 0, "parse_error_count": 0, "invalid_tool_count": 0}
    for index, row in enumerate(rows):
        try:
            if source == "xlam":
                converted = [convert_xlam(dict(row), row_index=index)]
            elif source == "agentinstruct":
                converted = convert_agentinstruct(dict(row), source_split=source_split)
            elif source == "history":
                converted = build_trajectory_samples(dict(row))
            elif source == "normalized-jsonl":
                converted = [dict(row)]
            else:
                raise ValueError(f"unsupported source: {source}")
        except LookupError as exc:
            stats["invalid_tool_count"] += 1
            stats["skipped_count"] += 1
            print(f"skip row {index}: {exc}")
            continue
        except (ValueError, TypeError, NotImplementedError) as exc:
            stats["parse_error_count"] += 1
            stats["skipped_count"] += 1
            print(f"skip row {index}: {exc}")
            continue
        output.extend(converted)
        stats["converted_count"] += 1
    return output, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert one explicit source schema into validated research-policy rows.")
    parser.add_argument("--source", choices=("xlam", "agentinstruct", "history", "no-tool", "normalized-jsonl"), required=True)
    parser.add_argument("--input", help="Local JSONL. Omit for the official Hugging Face xLAM/AgentInstruct loader.")
    parser.add_argument("--split", default=None, help="HF split/environment (xLAM: train; AgentInstruct: os by default).")
    parser.add_argument("--output", default="artifacts/training/research_agent/normalized.jsonl")
    parser.add_argument("--report", default=None, help="Optional conversion statistics JSON path.")
    parser.add_argument("--max-source-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source == "no-tool":
        converted = build_no_tool_samples()
        stats = {"converted_count": len(converted), "skipped_count": 0, "parse_error_count": 0, "invalid_tool_count": 0}
    else:
        split = args.split or ("train" if args.source == "xlam" else "os")
        if args.input:
            source_rows: Iterable[dict[str, Any]] = read_jsonl(args.input)
        elif args.source in HF_DATASETS:
            source_rows = _load_huggingface(args.source, split)
        else:
            raise SystemExit(f"--input is required for source {args.source}")
        if args.max_source_rows is not None:
            source_rows = list(source_rows)[: max(args.max_source_rows, 0)]
        converted, stats = convert_rows(args.source, source_rows, source_split=split)
    if not converted:
        print(json.dumps(stats, sort_keys=True))
        return 2
    count = write_jsonl(Path(args.output), converted)
    report_path = Path(args.report) if args.report else Path(f"{args.output}.report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({**stats, "output_rows": count}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**stats, "output_rows": count, "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
