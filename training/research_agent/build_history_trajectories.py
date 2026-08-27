from __future__ import annotations

import argparse
import hashlib
import re
from typing import Any

from app.agents.policy_schema import (
    PolicyLimits,
    ResearchPolicyState,
    ToolObservation,
    default_research_tool_definitions,
    policy_messages,
)
from training.common.datasets import first_user_assistant, load_messages
from training.common.jsonl import write_jsonl
from training.history_answerer.evaluate import parse_source_ids


QUESTION_BLOCK_RE = re.compile(
    r"^\s*Câu hỏi\s*:\s*(?P<question>.*?)(?:\n\s*Tài liệu tham khảo\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
REFERENCE_ID_RE = re.compile(r"(?m)^\s*\[([^\]\r\n]+)\]")


def extract_question_only(user_text: str) -> str:
    match = QUESTION_BLOCK_RE.search(user_text)
    if match:
        question = match.group("question").strip()
    else:
        question = re.split(r"\n\s*Tài liệu tham khảo\s*:", user_text, maxsplit=1, flags=re.IGNORECASE)[0]
        question = re.sub(r"^\s*Câu hỏi\s*:\s*", "", question, flags=re.IGNORECASE).strip()
    if not question:
        raise ValueError("empty question after removing reference material")
    return question


def extract_reference_ids(user_text: str) -> list[str]:
    parts = re.split(r"\n\s*Tài liệu tham khảo\s*:\s*", user_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return []
    return list(dict.fromkeys(match.strip() for match in REFERENCE_ID_RE.findall(parts[1]) if match.strip()))


def canonical_question_hash(question: str) -> str:
    canonical = " ".join(question.casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _state(*, question: str, step: int, trajectory_class: str,
           observations: list[ToolObservation], evidence_ids: list[str]) -> ResearchPolicyState:
    return ResearchPolicyState(
        question=question,
        retrieval_question=question,
        step=step,
        limits=PolicyLimits(max_steps=4, web_searches_left=0, page_fetches_left=0),
        tools=default_research_tool_definitions(),
        observations=observations,
        evidence_ids=evidence_ids,
        trajectory_class=trajectory_class,
    )


def _sample(*, sample_id: str, original_id: str, group_id: str, trajectory_id: str,
            state: ResearchPolicyState, target: dict[str, Any], trajectory_class: str,
            step: int, gold_source_ids: list[str], source_dataset: str = "vn_history_phase6") -> dict[str, Any]:
    del gold_source_ids  # Gold IDs may appear only in a post-observation action, never as hidden row metadata.
    return {
        "id": sample_id,
        "source_dataset": source_dataset,
        "source": "history_policy",
        "stage": "history_policy",
        "original_sample_id": original_id,
        "group_id": group_id,
        "trajectory_id": trajectory_id,
        "trajectory_class": trajectory_class,
        "step": step,
        "grounded": True,
        "synthetic": False,
        "messages": policy_messages(state, target),
        "training_prompt": state.model_dump(exclude_none=True),
        "training_target": target,
    }


def build_trajectory_samples(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Unroll a recorded Phase-6 retrieval into causal state -> action rows."""
    user_text, assistant_text = first_user_assistant(row)
    question = extract_question_only(user_text)
    retrieved_ids = extract_reference_ids(user_text)
    gold_ids = list(dict.fromkeys(parse_source_ids(assistant_text)))
    if not set(gold_ids).issubset(set(retrieved_ids)):
        raise ValueError(f"gold source IDs are absent from recorded references for row {row.get('id')}")

    original_id = str(row.get("id") or canonical_question_hash(question))
    group_id = str(row.get("original_sample_id") or canonical_question_hash(question))
    trajectory_id = f"history-{original_id}"
    sample_type = str(row.get("type") or "")
    trajectory_class = "false_premise" if sample_type == "false_premise" else (
        "local_only" if retrieved_ids else "insufficient"
    )

    initial = _state(question=question, step=1, trajectory_class=trajectory_class,
                     observations=[], evidence_ids=[])
    samples = [_sample(
        sample_id=f"{trajectory_id}-step-001", original_id=original_id, group_id=group_id,
        trajectory_id=trajectory_id, state=initial,
        target={"action": "tool", "tool_name": "search_history", "arguments": {"query": question, "top_k": 8}},
        trajectory_class=trajectory_class, step=1, gold_source_ids=gold_ids,
    )]

    search_observation = ToolObservation(tool="search_history", result_count=len(retrieved_ids), evidence_ids=retrieved_ids)
    state_after_search = _state(question=question, step=2, trajectory_class=trajectory_class,
                                observations=[search_observation], evidence_ids=retrieved_ids)
    if gold_ids:
        samples.append(_sample(
            sample_id=f"{trajectory_id}-step-002", original_id=original_id, group_id=group_id,
            trajectory_id=trajectory_id, state=state_after_search,
            target={"action": "tool", "tool_name": "inspect_evidence", "arguments": {"ids": gold_ids}},
            trajectory_class=trajectory_class, step=2, gold_source_ids=gold_ids,
        ))
        inspect_observation = ToolObservation(tool="inspect_evidence", result_count=len(gold_ids), evidence_ids=gold_ids)
        final_state = _state(question=question, step=3, trajectory_class=trajectory_class,
                             observations=[search_observation, inspect_observation], evidence_ids=retrieved_ids)
        finish = {"action": "finish", "sufficient": True, "missing_information": []}
        final_step = 3
    else:
        final_state = state_after_search
        finish = {"action": "finish", "sufficient": False,
                  "missing_information": ["No supporting local evidence was retrieved."]}
        final_step = 2
    samples.append(_sample(
        sample_id=f"{trajectory_id}-step-{final_step:03d}", original_id=original_id, group_id=group_id,
        trajectory_id=trajectory_id, state=final_state, target=finish,
        trajectory_class=trajectory_class, step=final_step, gold_source_ids=gold_ids,
    ))
    return samples


def build_trajectory(row: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Compatibility wrapper; index is ignored so row order cannot determine behavior."""
    del index
    return build_trajectory_samples(row)[0]


def build_no_tool_samples() -> list[dict[str, Any]]:
    prompts = ["Xin chào!", "Bạn làm được gì?", "Hãy giải thích cách sử dụng chatbot này.", "Cảm ơn bạn."]
    rows = []
    for index, question in enumerate(prompts, 1):
        group_id = f"no-tool-{index:03d}"
        state = _state(question=question, step=1, trajectory_class="no_tool", observations=[], evidence_ids=[])
        rows.append(_sample(
            sample_id=f"{group_id}-step-001", original_id=group_id, group_id=group_id,
            trajectory_id=group_id, state=state,
            target={"action": "finish", "sufficient": True, "missing_information": []},
            trajectory_class="no_tool", step=1, gold_source_ids=[], source_dataset="no_tool_seed",
        ))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create grounded, unrolled VN-history policy trajectories.")
    parser.add_argument("--input", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output", default="artifacts/training/research_agent/history_trajectories.jsonl")
    parser.add_argument("--include-no-tool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-source-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_rows = load_messages(args.input)
    if args.max_source_rows is not None:
        source_rows = source_rows[: max(args.max_source_rows, 0)]
    rows = [sample for row in source_rows for sample in build_trajectory_samples(row)]
    if args.include_no_tool:
        rows.extend(build_no_tool_samples())
    print(f"Wrote {write_jsonl(args.output, rows)} grounded state-action rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
