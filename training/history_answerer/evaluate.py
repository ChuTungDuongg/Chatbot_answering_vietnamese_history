from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.common.jsonl import read_jsonl
from training.common.datasets import first_user_assistant


SOURCE_RE = re.compile(r"Nguồn được dùng\s*:\s*\[(.*?)\]", re.S)
CONTEXT_ID_RE = re.compile(r"(?m)^\[([^\]]+)\]")


def parse_source_ids(text: str) -> list[str]:
    match = SOURCE_RE.search(text or "")
    if not match:
        return []
    return [item.strip().strip("'\"") for item in match.group(1).split(",") if item.strip()]


def rouge_l(pred: str, gold: str) -> float:
    a = pred.split()
    b = gold.split()
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if token_a == token_b else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    return 2 * lcs / (len(a) + len(b))


def evaluate_predictions(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, float]:
    source_exact = 0
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0
    format_ok = 0
    answer_non_empty = 0
    source_ids_exist = 0
    insufficient_empty = 0
    insufficient_count = 0
    rouge_total = 0.0
    n = len(gold_rows) or 1
    for idx, gold in enumerate(gold_rows):
        pred = pred_rows[idx] if idx < len(pred_rows) else {}
        pred_text = str(pred.get("assistant") or pred.get("answer") or pred.get("prediction") or "")
        if gold.get("messages"):
            user_text, gold_text = first_user_assistant(gold)
        else:
            user_text = str(gold.get("user") or gold.get("question") or "")
            gold_text = str(gold.get("assistant") or gold.get("answer") or "")
        pred_ids = set(parse_source_ids(pred_text))
        gold_ids = set(parse_source_ids(gold_text))
        context_ids = set(CONTEXT_ID_RE.findall(user_text))
        if pred_ids == gold_ids:
            source_exact += 1
        if "Nguồn được dùng:" in pred_text and "Trả lời:" in pred_text:
            format_ok += 1
        answer_body = pred_text.split("Trả lời:", 1)[-1].strip() if "Trả lời:" in pred_text else pred_text.strip()
        answer_non_empty += int(bool(answer_body))
        source_ids_exist += int(pred_ids <= context_ids)
        if gold.get("type") == "insufficient_context":
            insufficient_count += 1
            insufficient_empty += int(not pred_ids)
        precision = len(pred_ids & gold_ids) / max(len(pred_ids), 1)
        recall = len(pred_ids & gold_ids) / max(len(gold_ids), 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precision_total += precision
        recall_total += recall
        f1_total += f1
        rouge_total += rouge_l(pred_text, gold_text)
    metrics = {
        "source_exact": source_exact / n,
        "source_precision": precision_total / n,
        "source_recall": recall_total / n,
        "source_f1": f1_total / n,
        "format_ok": format_ok / n,
        "answer_non_empty": answer_non_empty / n,
        "source_ids_exist": source_ids_exist / n,
        "insufficient_empty_rate": insufficient_empty / max(insufficient_count, 1),
        "rouge_l": rouge_total / n,
    }
    metrics["generation_composite"] = (
        metrics["source_f1"]
        + metrics["format_ok"]
        + metrics["answer_non_empty"]
        + metrics["source_ids_exist"]
        + metrics["rouge_l"]
    ) / 5
    loss_values = [float(row["eval_loss"]) for row in pred_rows if row.get("eval_loss") is not None]
    test_values = [float(row["test_loss"]) for row in pred_rows if row.get("test_loss") is not None]
    metrics["eval_loss"] = sum(loss_values) / len(loss_values) if loss_values else float("nan")
    metrics["test_loss"] = sum(test_values) / len(test_values) if test_values else float("nan")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate history-answerer predictions.")
    parser.add_argument("--gold", default="artifacts/training/history_answerer/messages_normalized.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate_predictions(read_jsonl(args.gold), read_jsonl(args.predictions))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


