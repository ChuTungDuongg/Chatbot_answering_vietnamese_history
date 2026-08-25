from __future__ import annotations

import argparse

from training.common.jsonl import read_jsonl


def selected_ids(row):
    output = row.get("output", row)
    if output.get("selected_ids") is not None:
        return set(output.get("selected_ids", []))
    return {str(item.get("evidence_id")) for item in output.get("selected_evidence", [])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate evidence-agent selected chunk IDs.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preds = read_jsonl(args.predictions)
    gold = read_jsonl(args.gold)
    f1_total = 0.0
    for pred, ref in zip(preds, gold):
        p = selected_ids(pred)
        r = selected_ids(ref)
        precision = len(p & r) / max(len(p), 1)
        recall = len(p & r) / max(len(r), 1)
        f1_total += 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    print({"selected_id_f1": f1_total / max(len(gold), 1), "count": len(gold)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


