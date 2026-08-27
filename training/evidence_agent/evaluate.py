from __future__ import annotations

import argparse
import json
from typing import Any

from app.agents.schemas import EvidenceModelOutput
from training.common.jsonl import read_jsonl


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value: Any = row
    for key in ("prediction", "output", "target"):
        if key in row:
            value = row[key]
            break
    if value is row and isinstance(row.get("messages"), list):
        value = next(
            (item.get("content") for item in reversed(row["messages"]) if item.get("role") == "assistant"),
            None,
        )
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("prediction is not a JSON object")
    return value


def evaluate_rows(predictions: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    count = min(len(predictions), len(gold))
    parsed = schema_valid = status_correct = 0
    precision_sum = recall_sum = f1_sum = 0.0
    invented = predicted_selected = 0
    duplicate_cases = duplicate_success = 0
    conflict_correct = 0
    missing_proxy_correct = 0
    compressed_nonempty = compressed_total = 0
    original_chars = compressed_chars = 0

    for pred_row, gold_row in zip(predictions[:count], gold[:count]):
        gold_output = EvidenceModelOutput.model_validate(_payload(gold_row))
        candidates = {str(item["evidence_id"]): item for item in gold_row.get("evidence", [])}
        try:
            pred_payload = _payload(pred_row)
            parsed += 1
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        try:
            prediction = EvidenceModelOutput.model_validate(pred_payload)
            schema_valid += 1
        except Exception:
            continue

        status_correct += int(prediction.status == gold_output.status)
        pred_ids = {item.evidence_id for item in prediction.selected_evidence}
        gold_ids = {item.evidence_id for item in gold_output.selected_evidence}
        if not pred_ids and not gold_ids:
            precision = recall = f1 = 1.0
        else:
            precision = len(pred_ids & gold_ids) / max(len(pred_ids), 1)
            recall = len(pred_ids & gold_ids) / max(len(gold_ids), 1)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        predicted_selected += len(pred_ids)
        invented += len(pred_ids - set(candidates))

        if gold_row.get("behavior") == "duplicate":
            duplicate_cases += 1
            duplicate_ids = {
                evidence_id
                for evidence_id, item in candidates.items()
                if str(item.get("source_type") or "").startswith("synthetic_duplicate")
            }
            duplicate_success += int(bool(pred_ids & gold_ids) and not (pred_ids & duplicate_ids))
        gold_is_conflict = gold_output.status == "conflicting"
        pred_is_conflict = prediction.status == "conflicting"
        conflict_correct += int(gold_is_conflict == pred_is_conflict)
        missing_proxy_correct += int(bool(prediction.missing_information) == bool(gold_output.missing_information))

        for item in prediction.selected_evidence:
            compressed_total += 1
            compressed_nonempty += int(bool(item.compressed_text.strip()))
            compressed_chars += len(item.compressed_text)
            if item.evidence_id in candidates:
                original_chars += len(str(candidates[item.evidence_id].get("text") or ""))

    ratio = lambda numerator, denominator: numerator / max(denominator, 1)
    return {
        "count": count,
        "json_parse_rate": ratio(parsed, count),
        "runtime_schema_validity": ratio(schema_valid, count),
        "status_accuracy": ratio(status_correct, count),
        "selected_evidence_precision": ratio(precision_sum, count),
        "selected_evidence_recall": ratio(recall_sum, count),
        "selected_evidence_f1": ratio(f1_sum, count),
        "invented_id_rate": ratio(invented, predicted_selected),
        "duplicate_reduction_rate": ratio(duplicate_success, duplicate_cases),
        "conflict_detection_accuracy": ratio(conflict_correct, count),
        "missing_information_accuracy_proxy": ratio(missing_proxy_correct, count),
        "compressed_text_nonempty_rate": ratio(compressed_nonempty, compressed_total),
        "compression_ratio": ratio(compressed_chars, original_chars),
        "semantic_grounding_metric": None,
        "semantic_grounding_note": "No reliable automated semantic grounding metric is claimed; use human/execution evaluation.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Evidence Agent predictions against the production runtime schema.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_rows(read_jsonl(args.predictions), read_jsonl(args.gold))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
