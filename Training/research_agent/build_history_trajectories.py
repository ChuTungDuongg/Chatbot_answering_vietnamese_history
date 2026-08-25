from __future__ import annotations

import argparse
import json
from typing import Any

from training.common.datasets import first_user_assistant, load_messages
from training.common.jsonl import write_jsonl
from training.history_answerer.evaluate import parse_source_ids


def build_trajectory(row: dict[str, Any], index: int = 0) -> dict[str, Any]:
    user_text, assistant_text = first_user_assistant(row)
    source_ids = parse_source_ids(assistant_text)
    sample_type = str(row.get("type", ""))
    if sample_type == "false_premise":
        trajectory_class = "no_tool_needed"
        tool_calls = []
    elif sample_type == "insufficient_context":
        trajectory_class = "local_then_web" if index % 2 == 0 else "insufficient"
        tool_calls = [
            {"name": "search_history", "arguments": {"query": user_text, "top_k": 8}},
            {"name": "search_web", "arguments": {"query": user_text, "top_k": 5}},
        ]
    elif sample_type == "noisy_context" and index % 5 == 0:
        trajectory_class = "conflicting_sources"
        tool_calls = [
            {"name": "search_history", "arguments": {"query": user_text, "top_k": 8}},
            {"name": "inspect_evidence", "arguments": {"ids": source_ids[:3]}},
            {"name": "search_history", "arguments": {"query": f"đối chiếu {user_text}", "top_k": 8}},
        ]
    elif sample_type == "noisy_context" and index % 3 == 0:
        trajectory_class = "multi_hop"
        tool_calls = [
            {"name": "search_history", "arguments": {"query": user_text, "top_k": 8}},
            {"name": "retrieve_evidence", "arguments": {"query": user_text, "top_k": 6}},
        ]
    else:
        trajectory_class = "local_only"
        tool_calls = [{"name": "search_history", "arguments": {"query": user_text, "top_k": 8}}]
    if tool_calls:
        action_index = index % len(tool_calls)
        action = tool_calls[action_index]
        training_target = {
            "action": "tool",
            "tool_name": action["name"],
            "arguments": action["arguments"],
        }
        observations = [
            {"tool": call["name"], "result": "observation_available"}
            for call in tool_calls[:action_index]
        ]
    else:
        training_target = {
            "action": "finish",
            "sufficient": sample_type == "false_premise",
            "missing_information": [],
        }
        observations = []
    training_prompt = {
        "question": user_text,
        "trajectory_class": trajectory_class,
        "observations": observations,
        "evidence_ids": source_ids,
        "allowed_tools": [
            "search_history",
            "search_web",
            "fetch_web_page",
            "retrieve_evidence",
            "inspect_evidence",
        ],
    }
    return {
        "id": row.get("id"),
        "source": "vn_history_phase6",
        "messages": [
            {"role": "user", "content": json.dumps(training_prompt, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(training_target, ensure_ascii=False)},
        ],
        "training_prompt": training_prompt,
        "training_target": training_target,
        "trajectory": {
            "question": user_text,
            "trajectory_class": trajectory_class,
            "tool_calls": tool_calls,
            "evidence_ids": source_ids,
            "stop_reason": "sufficient_evidence" if source_ids else "insufficient_evidence",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create local VN-history tool-use trajectories for the research agent.")
    parser.add_argument("--input", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output", default="artifacts/training/research_agent/history_trajectories.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [build_trajectory(row, index) for index, row in enumerate(load_messages(args.input))]
    print(f"Wrote {write_jsonl(args.output, rows)} trajectories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



