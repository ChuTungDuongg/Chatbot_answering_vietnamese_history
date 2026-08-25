from __future__ import annotations

import argparse

from training.common.jsonl import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate research-agent trajectories.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preds = read_jsonl(args.predictions)
    gold = read_jsonl(args.gold)
    exact = 0
    for pred, ref in zip(preds, gold):
        pred_tools = [call.get("name") for call in pred.get("trajectory", {}).get("tool_calls", [])]
        ref_tools = [call.get("name") for call in ref.get("trajectory", {}).get("tool_calls", [])]
        exact += int(pred_tools == ref_tools)
    print({"tool_sequence_exact": exact / max(len(gold), 1), "count": len(gold)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



