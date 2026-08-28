from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.agents.policy_schema import (
    GENERIC_TOOL_USE_SYSTEM,
    RESEARCH_AGENT_SYSTEM,
    ResearchPolicyState,
    ToolBatchDecision,
    ToolDecision,
    validate_training_decision,
)
from training.common.jsonl import read_jsonl
from training.common.datasets import split_rows


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = schema.get("required")
    properties = schema.get("properties")
    if isinstance(required, list):
        for name in required:
            if name not in arguments:
                errors.append(f"missing required argument {name!r}")
    elif isinstance(schema, dict):
        # xLAM's official schema stores required on each parameter.
        for name, definition in schema.items():
            if isinstance(definition, dict) and definition.get("required") is True and name not in arguments:
                errors.append(f"missing required argument {name!r}")
    if isinstance(properties, dict):
        unknown = sorted(set(arguments) - set(properties))
        if schema.get("additionalProperties") is False and unknown:
            errors.append(f"unknown arguments: {', '.join(unknown)}")
        definitions = properties
    else:
        definitions = schema
    type_map = {
        "string": str,
        "integer": int,
        "int": int,
        "number": (int, float),
        "float": (int, float),
        "array": list,
        "list": list,
        "object": dict,
        "dict": dict,
        "boolean": bool,
        "bool": bool,
    }
    if isinstance(definitions, dict):
        for name, value in arguments.items():
            definition = definitions.get(name)
            expected = definition.get("type") if isinstance(definition, dict) else None
            python_type = type_map.get(expected)
            if python_type and (not isinstance(value, python_type) or expected in {"integer", "int"} and isinstance(value, bool)):
                errors.append(f"argument {name!r} must have type {expected}")
    return errors


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    trajectory_steps: dict[str, list[int]] = defaultdict(list)
    trajectory_records: dict[str, list[tuple[int, ResearchPolicyState, Any]]] = defaultdict(list)
    trajectory_identity: dict[str, tuple[str, str, str]] = {}
    group_trajectories: dict[str, set[str]] = defaultdict(set)
    classes: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    no_tool_categories: Counter[str] = Counter()
    boundary_categories: Counter[str] = Counter()
    no_tool_rows = 0
    meta_without_history_lookup = 0

    for index, row in enumerate(rows, 1):
        label = f"row {index}"
        row_id = str(row.get("id") or "")
        if not row_id:
            errors.append(f"{label}: missing id")
        elif row_id in ids:
            errors.append(f"{label}: duplicate id {row_id!r}")
        if row_id:
            ids.add(row_id)
        group_id = str(row.get("group_id") or "")
        trajectory_id = str(row.get("trajectory_id") or "")
        step = row.get("step")
        if not group_id:
            errors.append(f"{label}: missing group_id")
        if not trajectory_id:
            errors.append(f"{label}: missing trajectory_id")
        if not isinstance(step, int) or step < 1:
            errors.append(f"{label}: invalid step")
        elif trajectory_id:
            trajectory_steps[trajectory_id].append(step)

        messages = row.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages if isinstance(item, dict)] != ["system", "user", "assistant"]:
            errors.append(f"{label}: messages must be system/user/assistant")
            continue
        expected_system = GENERIC_TOOL_USE_SYSTEM if row.get("stage") in {"generic_tool_use", "multi_step_agent"} else RESEARCH_AGENT_SYSTEM
        if messages[0].get("content") != expected_system:
            errors.append(f"{label}: system prompt does not match shared constant")
        try:
            state_payload = json.loads(messages[1].get("content") or "")
            state = ResearchPolicyState.model_validate(state_payload)
        except Exception as exc:
            errors.append(f"{label}: invalid policy state: {exc}")
            continue
        if "Tài liệu tham khảo:" in state.question:
            errors.append(f"{label}: Phase-6 reference block leaked into question")
        if state.step != step:
            errors.append(f"{label}: top-level and state steps differ")
        if len(state.observations) > state.step - 1:
            errors.append(f"{label}: future observations present")
        if row.get("training_prompt") != state.model_dump(exclude_none=True):
            errors.append(f"{label}: training_prompt differs from serialized runtime state")
        try:
            target_payload = json.loads(messages[2].get("content") or "")
            decision = validate_training_decision(target_payload, tool_names={tool.name for tool in state.tools})
        except Exception as exc:
            errors.append(f"{label}: invalid decision: {exc}")
            continue
        if row.get("stage") == "history_policy" and isinstance(decision, ToolBatchDecision):
            errors.append(f"{label}: runtime history policy cannot emit tool_batch")
        if row.get("training_target") != decision.model_dump():
            errors.append(f"{label}: training_target differs from assistant decision")
        trajectory_class = str(row.get("trajectory_class") or "unknown")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if trajectory_class == "no_tool":
            no_tool_rows += 1
            category = str(metadata.get("no_tool_category") or "other")
            no_tool_categories[category] += 1
            if decision.action != "finish":
                errors.append(f"{label}: no-tool supervision must finish without retrieval")
            if category in {"capability", "usage_help", "ui_help"}:
                meta_without_history_lookup += 1
        boundary_category = str(metadata.get("boundary_category") or "")
        if boundary_category:
            boundary_categories[boundary_category] += 1
            if not isinstance(decision, ToolDecision) or decision.tool_name != "search_history":
                errors.append(f"{label}: conversational-prefix history boundary must search_history")
        calls = decision.tool_calls if isinstance(decision, ToolBatchDecision) else ([decision] if isinstance(decision, ToolDecision) else [])
        schemas = {tool.name: tool.input_schema for tool in state.tools}
        for call in calls:
            for issue in _validate_arguments(call.arguments, schemas.get(call.tool_name, {})):
                errors.append(f"{label}: {call.tool_name}: {issue}")
            if call.tool_name == "inspect_evidence":
                requested_ids = call.arguments.get("ids") or []
                if not set(requested_ids).issubset(set(state.evidence_ids)):
                    errors.append(f"{label}: inspect_evidence references IDs not yet observed")

        if trajectory_id and isinstance(step, int):
            trajectory_records[trajectory_id].append((step, state, decision))
            identity = (group_id, str(row.get("original_sample_id") or ""), state.question)
            previous_identity = trajectory_identity.setdefault(trajectory_id, identity)
            if previous_identity != identity:
                errors.append(f"{label}: duplicate trajectory_id {trajectory_id!r} reused for distinct source trajectories")
            if group_id:
                group_trajectories[group_id].add(trajectory_id)

        classes[trajectory_class] += 1
        sources[str(row.get("source_dataset") or row.get("source") or "unknown")] += 1

    for trajectory_id, steps in trajectory_steps.items():
        ordered = sorted(steps)
        if len(ordered) != len(set(ordered)):
            errors.append(f"duplicate trajectory_id {trajectory_id!r}: one or more steps are repeated")
        if ordered and ordered != list(range(1, max(ordered) + 1)):
            errors.append(f"trajectory {trajectory_id!r}: non-contiguous ordering {ordered}")
    for trajectory_id, records in trajectory_records.items():
        ordered_records = sorted(records, key=lambda item: item[0])
        for (_, previous_state, previous_decision), (_, current_state, _) in zip(ordered_records, ordered_records[1:]):
            previous_observations = [item.model_dump(exclude_none=True) for item in previous_state.observations]
            current_observations = [item.model_dump(exclude_none=True) for item in current_state.observations]
            if current_observations[: len(previous_observations)] != previous_observations:
                errors.append(f"trajectory {trajectory_id!r}: observations are not a causal prefix")
            if isinstance(previous_decision, ToolDecision):
                if len(current_observations) != len(previous_observations) + 1:
                    errors.append(f"trajectory {trajectory_id!r}: tool action lacks exactly one next observation")
                elif current_observations[-1].get("tool") != previous_decision.tool_name:
                    errors.append(f"trajectory {trajectory_id!r}: observation tool does not match previous action")
    if not rows:
        errors.append("dataset is empty")
    if len(classes) < 2:
        warnings.append("dataset contains fewer than two trajectory classes")
    unique_groups = len(group_trajectories)
    unique_trajectories = len(trajectory_identity)
    split_no_tool_distribution = {"train": 0, "eval": 0, "test": 0}
    split_class_distribution: dict[str, dict[str, int]] = {"train": {}, "eval": {}, "test": {}}
    split_row_counts = {"train": 0, "eval": 0, "test": 0}
    split_groups_count = {"train": 0, "eval": 0, "test": 0}
    split_group_overlap = {"train_eval": 0, "train_test": 0, "eval_test": 0}
    near_duplicate_no_tool_cross_split = 0
    no_tool_group_overlap = 0
    if rows and all(row.get("group_id") for row in rows):
        splits = split_rows(
            rows,
            seed=42,
            train_ratio=0.88,
            eval_ratio=0.06,
            group_key="group_id",
            stratify_key="trajectory_class",
        )
        split_groups = {
            name: {str(row["group_id"]) for row in getattr(splits, name)}
            for name in ("train", "eval", "test")
        }
        split_group_overlap = {
            "train_eval": len(split_groups["train"] & split_groups["eval"]),
            "train_test": len(split_groups["train"] & split_groups["test"]),
            "eval_test": len(split_groups["eval"] & split_groups["test"]),
        }
        split_row_counts = {name: len(getattr(splits, name)) for name in ("train", "eval", "test")}
        split_groups_count = {name: len(split_groups[name]) for name in ("train", "eval", "test")}
        split_class_distribution = {
            name: dict(Counter(str(row.get("trajectory_class") or "unknown") for row in getattr(splits, name)))
            for name in ("train", "eval", "test")
        }
        no_tool_group_overlap = sum(split_group_overlap.values())
        split_no_tool_distribution = {
            name: sum(row.get("trajectory_class") == "no_tool" for row in getattr(splits, name))
            for name in ("train", "eval", "test")
        }
        semantic_groups = {
            name: {
                str((row.get("metadata") or {}).get("semantic_group"))
                for row in getattr(splits, name)
                if row.get("trajectory_class") == "no_tool" and (row.get("metadata") or {}).get("semantic_group")
            }
            for name in ("train", "eval", "test")
        }
        near_duplicate_no_tool_cross_split = (
            len(semantic_groups["train"] & semantic_groups["eval"])
            + len(semantic_groups["train"] & semantic_groups["test"])
            + len(semantic_groups["eval"] & semantic_groups["test"])
        )
        if any(split_group_overlap.values()):
            errors.append(f"group leakage across Research splits: {split_group_overlap}")
        if near_duplicate_no_tool_cross_split:
            errors.append("near-duplicate no-tool semantic families cross splits")
        if no_tool_rows and any(value == 0 for value in split_no_tool_distribution.values()):
            errors.append(f"no-tool supervision missing from a split: {split_no_tool_distribution}")
    return {
        "valid": not errors,
        "rows": len(rows),
        "unique_ids": len(ids),
        "unique_trajectory_ids": unique_trajectories,
        "unique_groups": unique_groups,
        "avg_rows_per_group": len(rows) / max(unique_groups, 1),
        "avg_trajectories_per_group": unique_trajectories / max(unique_groups, 1),
        "class_distribution": dict(classes),
        "source_distribution": dict(sources),
        "no_tool_rows": no_tool_rows,
        "no_tool_percentage": 100.0 * no_tool_rows / max(len(rows), 1),
        "no_tool_categories": dict(no_tool_categories),
        "history_with_conversational_prefix": sum(boundary_categories.values()),
        "boundary_categories": dict(boundary_categories),
        "meta_without_history_lookup": meta_without_history_lookup,
        "split_no_tool_distribution": split_no_tool_distribution,
        "split_rows": split_row_counts,
        "split_groups": split_groups_count,
        "split_class_distribution": split_class_distribution,
        "split_group_overlap": split_group_overlap,
        "near_duplicate_no_tool_cross_split": near_duplicate_no_tool_cross_split,
        "no_tool_group_overlap": no_tool_group_overlap,
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Research Agent JSONL before training.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = read_jsonl(args.dataset)
        report = validate_rows(rows)
    except (OSError, ValueError) as exc:
        report = {"valid": False, "errors": [str(exc)], "warnings": []}
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
