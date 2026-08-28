from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from app.agents.policy_schema import ResearchPolicyState, ToolBatchDecision, ToolDecision, validate_training_decision
from training.common.jsonl import read_jsonl


def _payload(row: dict[str, Any]) -> Any:
    for key in ("prediction", "training_target", "target"):
        if key in row:
            value = row[key]
            if isinstance(value, str):
                return json.loads(value)
            return value
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return json.loads(message.get("content") or "")
    raise ValueError("missing decision payload")


def evaluate_rows(predictions: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    count = min(len(predictions), len(gold))
    totals = defaultdict(int)
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sequences: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"pred": [], "gold": []})
    for pred_row, gold_row in zip(predictions[:count], gold[:count]):
        trajectory_class = str(gold_row.get("trajectory_class") or "unknown")
        bucket = by_class[trajectory_class]
        bucket["count"] += 1
        try:
            state = ResearchPolicyState.model_validate(gold_row["training_prompt"])
            tools = {tool.name for tool in state.tools}
            pred = validate_training_decision(_payload(pred_row))
            totals["parsed"] += 1
            bucket["parsed"] += 1
        except Exception:
            continue
        gold_decision = validate_training_decision(_payload(gold_row))
        pred_action = pred.action
        gold_action = gold_decision.action
        if pred_action == gold_action:
            totals["action_correct"] += 1
            bucket["action_correct"] += 1

        pred_calls = pred.tool_calls if isinstance(pred, ToolBatchDecision) else ([pred] if isinstance(pred, ToolDecision) else [])
        gold_calls = gold_decision.tool_calls if isinstance(gold_decision, ToolBatchDecision) else ([gold_decision] if isinstance(gold_decision, ToolDecision) else [])
        if gold_calls:
            totals["tool_rows"] += 1
            bucket["tool_rows"] += 1
            if [call.tool_name for call in pred_calls] == [call.tool_name for call in gold_calls]:
                totals["tool_correct"] += 1
                bucket["tool_correct"] += 1
            if all(isinstance(call.arguments, dict) for call in pred_calls):
                totals["arguments_valid"] += 1
                bucket["arguments_valid"] += 1
        if any(call.tool_name not in tools for call in pred_calls):
            totals["unknown_tool"] += 1
            bucket["unknown_tool"] += 1
        if pred_action == "finish" and gold_action == "finish":
            totals["finish_correct"] += int(pred.model_dump() == gold_decision.model_dump())
        if trajectory_class in {"no_tool", "no_tool_needed"}:
            totals["no_tool_rows"] += 1
            totals["no_tool_correct"] += int(pred_action == "finish")
        predicted_no_tool = pred_action == "finish" and state.step == 1 and not state.observations
        gold_no_tool = trajectory_class in {"no_tool", "no_tool_needed"}
        totals["no_tool_predicted"] += int(predicted_no_tool)
        totals["no_tool_tp"] += int(predicted_no_tool and gold_no_tool)
        if any(call.tool_name == "search_history" for call in gold_calls):
            totals["history_search_rows"] += 1
            totals["history_search_correct"] += int(
                any(call.tool_name == "search_history" for call in pred_calls)
            )
        metadata = gold_row.get("metadata") or {}
        if metadata.get("boundary_category"):
            totals["boundary_rows"] += 1
            totals["boundary_correct"] += int(
                any(call.tool_name == "search_history" for call in pred_calls)
            )
        if metadata.get("no_tool_category") in {"capability", "usage_help", "ui_help"}:
            totals["meta_rows"] += 1
            totals["meta_correct"] += int(pred_action == "finish")
        if trajectory_class == "false_premise" and state.step == 1:
            totals["false_premise_initial"] += 1
            totals["false_premise_search_correct"] += int(
                any(call.tool_name == "search_history" for call in pred_calls)
            )
        if any(call.tool_name == "inspect_evidence" for call in gold_calls):
            totals["search_to_inspect_rows"] += 1
            totals["search_to_inspect_correct"] += int(
                any(call.tool_name == "inspect_evidence" for call in pred_calls)
            )
        if gold_action == "finish" and any(obs.tool == "inspect_evidence" for obs in state.observations):
            totals["inspect_to_finish_rows"] += 1
            totals["inspect_to_finish_correct"] += int(pred_action == "finish")
        if trajectory_class in {"local_only", "no_tool", "no_tool_needed"} and any(call.tool_name == "search_web" for call in pred_calls):
            totals["unnecessary_web"] += 1
        if any(call.tool_name == "search_web" for call in pred_calls) and state.limits.web_searches_left <= 0:
            totals["budget_violation"] += 1

        trajectory_id = str(gold_row.get("trajectory_id") or gold_row.get("id"))
        sequences[trajectory_id]["pred"].extend(call.tool_name for call in pred_calls)
        sequences[trajectory_id]["gold"].extend(call.tool_name for call in gold_calls)
        if pred_action == "finish":
            sequences[trajectory_id]["pred"].append("finish")
        if gold_action == "finish":
            sequences[trajectory_id]["gold"].append("finish")

    trajectory_success = sum(value["pred"] == value["gold"] for value in sequences.values())
    ratio = lambda numerator, denominator: numerator / max(denominator, 1)
    no_tool_precision = ratio(totals["no_tool_tp"], totals["no_tool_predicted"])
    no_tool_recall = ratio(totals["no_tool_tp"], totals["no_tool_rows"])
    report = {
        "count": count,
        "action_accuracy": ratio(totals["action_correct"], count),
        "decision_parse_rate": ratio(totals["parsed"], count),
        "tool_selection_accuracy": ratio(totals["tool_correct"], totals["tool_rows"]),
        "argument_validity_rate": ratio(totals["arguments_valid"], totals["tool_rows"]),
        "unknown_tool_rate": ratio(totals["unknown_tool"], count),
        "finish_accuracy": ratio(totals["finish_correct"], sum(1 for row in gold[:count] if _payload(row).get("action") == "finish")),
        "no_tool_accuracy": ratio(totals["no_tool_correct"], totals["no_tool_rows"]),
        "no_tool_precision": no_tool_precision,
        "no_tool_recall": no_tool_recall,
        "no_tool_f1": ratio(2 * no_tool_precision * no_tool_recall, no_tool_precision + no_tool_recall),
        "history_search_recall": ratio(totals["history_search_correct"], totals["history_search_rows"]),
        "conversational_prefix_history_search_accuracy": ratio(totals["boundary_correct"], totals["boundary_rows"]),
        "meta_request_no_tool_accuracy": ratio(totals["meta_correct"], totals["meta_rows"]),
        "false_premise_initial_search_accuracy": ratio(
            totals["false_premise_search_correct"], totals["false_premise_initial"]
        ),
        "search_to_inspect_accuracy": ratio(
            totals["search_to_inspect_correct"], totals["search_to_inspect_rows"]
        ),
        "inspect_to_finish_accuracy": ratio(
            totals["inspect_to_finish_correct"], totals["inspect_to_finish_rows"]
        ),
        "schema_valid_rate": ratio(totals["parsed"], count),
        "unnecessary_web_search_rate": ratio(totals["unnecessary_web"], count),
        "budget_violation_rate": ratio(totals["budget_violation"], count),
        "trajectory_success_rate": ratio(trajectory_success, len(sequences)),
        "tool_sequence_exact": ratio(trajectory_success, len(sequences)),
        "trajectory_success_is_offline_proxy": True,
        "by_trajectory_class": {
            name: {
                "count": values["count"],
                "decision_parse_rate": ratio(values["parsed"], values["count"]),
                "tool_selection_accuracy": ratio(values["tool_correct"], values["tool_rows"]),
                "unknown_tool_rate": ratio(values["unknown_tool"], values["count"]),
            }
            for name, values in by_class.items()
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Research Agent policy evaluation (trajectory success is a proxy).")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(evaluate_rows(read_jsonl(args.predictions), read_jsonl(args.gold)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
