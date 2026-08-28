from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.agents.policy_schema import (
    FinishDecision,
    ResearchPolicyState,
    ToolDecision,
    validate_training_decision,
)
from training.common.jsonl import read_jsonl, write_jsonl
from training.research_agent.evaluate import evaluate_rows


SANITY_CATEGORIES = (
    "no_tool",
    "factual_history_search",
    "conversational_prefix_history_search",
    "search_to_inspect",
    "inspect_to_finish",
)


def inference_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Return the exact system/user messages seen during canonical SFT."""
    validate_canonical_sanity_row(row)
    return copy.deepcopy(row["messages"][:-1])


def _target(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("training_target")
    if not isinstance(value, dict):
        raise ValueError(f"row {row.get('id')} is missing training_target")
    return value


def _category(row: dict[str, Any]) -> str | None:
    target = _target(row)
    action = target.get("action")
    tool_name = target.get("tool_name")
    state = row.get("training_prompt") or {}
    observations = state.get("observations") or []
    metadata = row.get("metadata") or {}
    step = row.get("step")

    if row.get("trajectory_class") in {"no_tool", "no_tool_needed"} and step == 1 and action == "finish":
        return "no_tool"
    if metadata.get("boundary_category") and step == 1 and tool_name == "search_history":
        return "conversational_prefix_history_search"
    if (
        row.get("source_dataset") == "vn_history_phase6"
        and row.get("trajectory_class") == "local_only"
        and step == 1
        and tool_name == "search_history"
        and not metadata.get("boundary_category")
    ):
        return "factual_history_search"
    if (
        row.get("source_dataset") == "vn_history_phase6"
        and row.get("synthetic") is False
        and step == 2
        and tool_name == "inspect_evidence"
    ):
        return "search_to_inspect"
    if action == "finish" and any(obs.get("tool") == "inspect_evidence" for obs in observations):
        return "inspect_to_finish"
    return None


def _diverse_take(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Prefer distinct source groups before taking another row from a group."""
    ordered = sorted(rows, key=lambda row: str(row.get("id") or ""))
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for row in ordered:
        group = str(row.get("group_id") or row.get("trajectory_id") or row.get("id"))
        if group in seen_groups:
            continue
        selected.append(row)
        seen_groups.add(group)
        if len(selected) == count:
            return selected
    selected_ids = {str(row.get("id")) for row in selected}
    for row in ordered:
        if str(row.get("id")) in selected_ids:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    return selected


def validate_canonical_sanity_row(row: dict[str, Any]) -> None:
    """Fail if a replay case diverges from canonical/runtime policy schema."""
    category = str(row.get("sanity_category") or _category(row) or "")
    if category not in SANITY_CATEGORIES:
        raise ValueError(f"row {row.get('id')} is not a supported Research sanity case")
    messages = row.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"row {row.get('id')} must preserve canonical system/user/assistant messages")
    state = ResearchPolicyState.model_validate(row.get("training_prompt"))
    serialized_state = json.loads(str(messages[1].get("content") or ""))
    if serialized_state != state.model_dump(exclude_none=True):
        raise ValueError(f"row {row.get('id')} user message differs from training_prompt")
    target = validate_training_decision(_target(row), tool_names={tool.name for tool in state.tools})
    serialized_target = json.loads(str(messages[2].get("content") or ""))
    if serialized_target != target.model_dump():
        raise ValueError(f"row {row.get('id')} assistant message differs from training_target")

    if category == "search_to_inspect":
        if row.get("source_dataset") != "vn_history_phase6" or row.get("synthetic") is not False:
            raise ValueError("search_to_inspect sanity rows must be real canonical Phase-6 rows")
        if state.step != 2 or len(state.observations) != 1 or state.observations[0].tool != "search_history":
            raise ValueError("search_to_inspect row must contain the real post-search state")
        observation = state.observations[0]
        if observation.result_count is None or not observation.evidence_ids:
            raise ValueError("post-search observation requires result_count and observed evidence IDs")
        if state.evidence_ids != observation.evidence_ids:
            raise ValueError("step-2 evidence_ids must preserve the canonical search observation")
        if not isinstance(target, ToolDecision) or target.tool_name != "inspect_evidence":
            raise ValueError("step-2 gold target must inspect evidence")
        target_ids = target.arguments.get("ids")
        if not isinstance(target_ids, list) or not target_ids:
            raise ValueError("step-2 inspect target requires non-empty evidence IDs")
        if not set(map(str, target_ids)).issubset(set(state.evidence_ids)):
            raise ValueError("step-2 inspect target IDs must come from the search observation")

    if category == "inspect_to_finish":
        if not isinstance(target, FinishDecision):
            raise ValueError("inspect_to_finish gold target must finish")
        if not any(obs.tool == "inspect_evidence" and obs.evidence_ids for obs in state.observations):
            raise ValueError("inspect_to_finish row requires a real inspect observation")


def build_sanity_suite(
    rows: list[dict[str, Any]], *, policy_cases: int = 5, state_cases: int = 20
) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = _category(row)
        if category:
            candidates[category].append(row)

    requested = {
        "no_tool": policy_cases,
        "factual_history_search": policy_cases,
        "conversational_prefix_history_search": policy_cases,
        "search_to_inspect": state_cases,
        "inspect_to_finish": state_cases,
    }
    suite: list[dict[str, Any]] = []
    for category in SANITY_CATEGORIES:
        selected = _diverse_take(candidates[category], requested[category])
        if len(selected) != requested[category]:
            raise ValueError(
                f"sanity category {category} has {len(selected)} canonical rows; "
                f"requires {requested[category]}"
            )
        for source in selected:
            row = copy.deepcopy(source)
            row["sanity_category"] = category
            validate_canonical_sanity_row(row)
            suite.append(row)
    return suite


def evaluate_sanity_predictions(
    predictions: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(predictions) != len(gold):
        raise ValueError(f"prediction count {len(predictions)} != sanity gold count {len(gold)}")
    grouped_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction, row in zip(predictions, gold):
        validate_canonical_sanity_row(row)
        category = str(row["sanity_category"])
        grouped_predictions[category].append(prediction)
        grouped_gold[category].append(row)
    return {
        "suite_counts": {category: len(grouped_gold[category]) for category in SANITY_CATEGORIES},
        "overall": evaluate_rows(predictions, gold),
        "by_sanity_category": {
            category: evaluate_rows(grouped_predictions[category], grouped_gold[category])
            for category in SANITY_CATEGORIES
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/evaluate canonical Research policy sanity replay; never loads model weights."
    )
    parser.add_argument("--dataset", default="datasets/research_agent/history_trajectories.jsonl")
    parser.add_argument("--output-gold", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--policy-cases", type=int, default=5)
    parser.add_argument("--state-cases", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.policy_cases <= 0 or args.state_cases <= 0:
        raise ValueError("case counts must be positive")
    suite = build_sanity_suite(
        read_jsonl(args.dataset),
        policy_cases=args.policy_cases,
        state_cases=args.state_cases,
    )
    if args.output_gold:
        write_jsonl(args.output_gold, suite)
    summary: dict[str, Any] = {
        "dataset": str(Path(args.dataset)),
        "rows": len(suite),
        "categories": {
            category: sum(row["sanity_category"] == category for row in suite)
            for category in SANITY_CATEGORIES
        },
        "search_to_inspect_source": "canonical vn_history_phase6 step-2 replay",
        "synthetic_step2_rows": sum(
            bool(row.get("synthetic"))
            for row in suite
            if row["sanity_category"] == "search_to_inspect"
        ),
    }
    if args.predictions:
        summary["evaluation"] = evaluate_sanity_predictions(read_jsonl(args.predictions), suite)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
