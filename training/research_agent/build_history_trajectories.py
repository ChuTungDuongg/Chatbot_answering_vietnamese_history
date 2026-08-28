from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
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

NO_TOOL_FAMILIES: dict[str, list[str]] = {
    "greeting": [
        "Xin chào!", "Chào bạn nhé.", "Hello bạn!", "Chào buổi sáng.",
        "Chào buổi chiều!", "Rất vui được gặp bạn.", "Có ai ở đây không?", "A lô, chào bạn.",
    ],
    "thanks": [
        "Cảm ơn bạn.", "Cảm ơn nhiều nhé!", "Mình hiểu rồi, cảm ơn.", "Thông tin hữu ích lắm.",
        "Tuyệt vời, cảm ơn bạn!", "Được rồi, xin cảm ơn.", "Cảm ơn vì đã hỗ trợ.", "Ổn rồi, cảm ơn nhé.",
    ],
    "farewell": [
        "Tạm biệt.", "Hẹn gặp lại!", "Chào nhé, mình đi đây.", "Kết thúc ở đây nhé.",
        "Bye bạn.", "Hẹn nói chuyện sau.", "Mình không hỏi thêm nữa.", "Chúc bạn một ngày tốt lành.",
    ],
    "capability": [
        "Bạn làm được gì?", "Chatbot này dùng để làm gì?", "Bạn hỗ trợ chủ đề nào?", "Khả năng của bạn là gì?",
        "Bạn có hỗ trợ câu hỏi lịch sử Việt Nam không?", "Tôi có thể nhờ bạn việc gì?", "Bạn là trợ lý kiểu gì?", "Phạm vi hỗ trợ của chatbot là gì?",
    ],
    "usage_help": [
        "Hãy giải thích cách sử dụng chatbot này.", "Tôi nên đặt câu hỏi như thế nào?", "Cho tôi hướng dẫn sử dụng.", "Tôi có thể hỏi loại câu nào?",
        "Hướng dẫn tôi cách đặt câu hỏi.", "Làm sao để hỏi cho rõ ý?", "Cho một mẫu câu hỏi phù hợp.", "Cách dùng tính năng tra cứu là gì?",
    ],
    "control": [
        "Dừng lại nhé.", "Bỏ qua câu trước.", "Đừng tra cứu thêm.", "Hãy bắt đầu lại cuộc trò chuyện.",
        "Chờ một chút.", "Tiếp tục khi tôi hỏi nhé.", "Không cần trả lời nữa.", "Xóa yêu cầu vừa rồi khỏi ngữ cảnh.",
    ],
    "acknowledgement": [
        "Ừ, mình rõ rồi.", "Đồng ý.", "Được.", "OK nhé.",
        "Mình đã hiểu.", "Rõ rồi.", "Chính xác.", "Tôi ghi nhận.",
    ],
    "repeat_reformat": [
        "Hãy lặp lại câu trả lời vừa rồi.", "Viết ngắn hơn nội dung vừa trả lời.", "Đổi câu trả lời trên thành gạch đầu dòng.", "Diễn đạt lại phần vừa nói cho dễ hiểu.",
        "Nhắc lại ý chính vừa nêu.", "Rút gọn câu trả lời trước.", "Định dạng lại phần trên.", "Viết lại câu vừa rồi theo cách trang trọng hơn.",
    ],
    "ui_help": [
        "Nút gửi nằm ở đâu?", "Tôi tải tài liệu lên bằng cách nào?", "Làm sao mở cuộc trò chuyện mới?", "Tôi xem lại lịch sử chat ở đâu?",
        "Làm sao đổi tên cuộc trò chuyện?", "Cách xóa một đoạn chat là gì?", "Tôi có thể đính kèm tệp không?", "Giao diện này có chế độ tối không?",
    ],
    "near_empty": [
        "...", "?", "À.", "Ừm.", "Này.", "Alo?", "Thế nhé.", "OK.",
    ],
}

BOUNDARY_FAMILIES: dict[str, list[str]] = {
    "history_with_greeting": [
        "Chào bạn, ai là Ngô Quyền?",
        "Xin chào, Hội nghị Genève năm 1954 bàn về vấn đề gì?",
        "Chào buổi sáng, chiến dịch Điện Biên Phủ kết thúc ngày nào?",
        "Hello, khởi nghĩa Hương Khê do ai lãnh đạo?",
    ],
    "history_with_thanks": [
        "Cảm ơn. Cho mình hỏi Điện Biên Phủ kết thúc ngày nào?",
        "Cảm ơn bạn, nhân tiện Hiệp định Genève được ký khi nào?",
        "Mình hiểu rồi, cảm ơn; ai chỉ huy trận Bạch Đằng năm 938?",
        "Thông tin hữu ích lắm. Cho hỏi nhà Trần chống quân nào?",
    ],
    "history_with_help_prefix": [
        "Bạn giúp mình với: Khởi nghĩa Hương Khê do ai lãnh đạo?",
        "Chatbot này cho tôi biết ai lãnh đạo Điện Biên Phủ?",
        "Cho tôi hỏi lịch sử: Bạch Đằng năm 938 chống quân nào?",
        "Bạn có thể cho tôi biết Ngô Quyền sinh năm nào?",
    ],
}


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


def stable_source_row_hash(row: dict[str, Any]) -> str:
    """Hash the complete source row deterministically for exact-duplicate detection."""
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_content_payload(row: dict[str, Any]) -> dict[str, Any]:
    user_text, assistant_text = first_user_assistant(row)
    return {
        "legacy_id": str(row.get("id") or ""),
        "type": str(row.get("type") or ""),
        "question": " ".join(extract_question_only(user_text).casefold().split()),
        "assistant": " ".join(assistant_text.split()),
        "reference_ids": extract_reference_ids(user_text),
    }


def canonical_content_fingerprint(row: dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_content_payload(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def trajectory_fingerprint(row: dict[str, Any]) -> str:
    """Create a deterministic per-source-row fingerprint without Python hash()."""
    payload = {
        **_canonical_content_payload(row),
        # Distinguish rows whose recorded reference text differs while the policy-visible
        # fields happen to match. Exact raw duplicates are removed before this is used.
        "source_row_sha256": stable_source_row_hash(row),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _id_component(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_-]+", "-", value.strip().casefold()).strip("-")
    return normalized or fallback


def source_dataset_statistics(rows: list[dict[str, Any]]) -> dict[str, int]:
    legacy_ids = Counter(str(row.get("id") or "") for row in rows)
    raw_fingerprints = Counter(stable_source_row_hash(row) for row in rows)
    content_fingerprints = Counter(canonical_content_fingerprint(row) for row in rows)
    return {
        "source_rows": len(rows),
        "legacy_unique_ids": len(legacy_ids),
        "legacy_duplicated_id_values": sum(count > 1 for count in legacy_ids.values()),
        "unique_content_rows": len(content_fingerprints),
        "exact_duplicate_rows": sum(count - 1 for count in raw_fingerprints.values()),
    }


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
            step: int, gold_source_ids: list[str], source_dataset: str = "vn_history_phase6",
            synthetic: bool = False, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
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
        "synthetic": synthetic,
        "metadata": metadata or {},
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

    legacy_id = str(row.get("id") or "").strip()
    original_id = legacy_id or f"question-{canonical_question_hash(question)}"
    group_source = str(row.get("original_sample_id") or original_id)
    group_id = f"history-{_id_component(group_source, fallback='unknown-group')}"
    sample_type = str(row.get("type") or "")
    type_component = _id_component(sample_type, fallback="unknown-type")
    trajectory_id = f"{group_id}-{type_component}-{trajectory_fingerprint(row)}"
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
    rows = []
    for category, prompts in NO_TOOL_FAMILIES.items():
        group_id = f"no-tool-{category.replace('_', '-')}"
        for index, question in enumerate(prompts, 1):
            trajectory_id = f"{group_id}-{index:03d}"
            state = _state(question=question, step=1, trajectory_class="no_tool", observations=[], evidence_ids=[])
            rows.append(_sample(
                sample_id=f"{trajectory_id}-step-001", original_id=trajectory_id, group_id=group_id,
                trajectory_id=trajectory_id, state=state,
                target={"action": "finish", "sufficient": True, "missing_information": []},
                trajectory_class="no_tool", step=1, gold_source_ids=[], source_dataset="no_tool_v23",
                synthetic=True,
                metadata={"no_tool_category": category, "semantic_group": group_id},
            ))
    return rows


def build_boundary_samples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, prompts in BOUNDARY_FAMILIES.items():
        group_id = f"boundary-{category.replace('_', '-')}"
        for index, question in enumerate(prompts, 1):
            trajectory_id = f"{group_id}-{index:03d}"
            state = _state(question=question, step=1, trajectory_class="local_only", observations=[], evidence_ids=[])
            rows.append(_sample(
                sample_id=f"{trajectory_id}-step-001", original_id=trajectory_id, group_id=group_id,
                trajectory_id=trajectory_id, state=state,
                target={"action": "tool", "tool_name": "search_history", "arguments": {"query": question, "top_k": 8}},
                trajectory_class="local_only", step=1, gold_source_ids=[], source_dataset="policy_boundary_v23",
                synthetic=True,
                metadata={"boundary_category": category, "semantic_group": group_id},
            ))
    return rows


def build_history_dataset(
    source_rows: list[dict[str, Any]], *, include_no_tool: bool = True, include_boundaries: bool | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build all rows while deduplicating only byte-for-byte-equivalent JSON objects."""
    stats = source_dataset_statistics(source_rows)
    if include_boundaries is None:
        include_boundaries = include_no_tool
    seen_source_hashes: set[str] = set()
    seen_trajectory_ids: set[str] = set()
    output: list[dict[str, Any]] = []
    malformed_source_rows = 0
    for row in source_rows:
        source_hash = stable_source_row_hash(row)
        if source_hash in seen_source_hashes:
            continue
        seen_source_hashes.add(source_hash)
        try:
            samples = build_trajectory_samples(row)
        except ValueError as exc:
            if "gold source IDs are absent" not in str(exc):
                raise
            malformed_source_rows += 1
            continue
        trajectory_id = str(samples[0]["trajectory_id"])
        if trajectory_id in seen_trajectory_ids:
            raise ValueError(f"deterministic trajectory ID collision: {trajectory_id}")
        seen_trajectory_ids.add(trajectory_id)
        output.extend(samples)
    if include_no_tool:
        no_tool_rows = build_no_tool_samples()
        no_tool_trajectory_ids = {str(row["trajectory_id"]) for row in no_tool_rows}
        overlap = seen_trajectory_ids & no_tool_trajectory_ids
        if overlap:
            raise ValueError(f"history/no-tool trajectory ID collision: {sorted(overlap)}")
        output.extend(no_tool_rows)
    if include_boundaries:
        boundary_rows = build_boundary_samples()
        boundary_ids = {str(row["trajectory_id"]) for row in boundary_rows}
        overlap = seen_trajectory_ids & boundary_ids
        if overlap:
            raise ValueError(f"history/boundary trajectory ID collision: {sorted(overlap)}")
        output.extend(boundary_rows)
    stats["no_tool_rows"] = sum(row["trajectory_class"] == "no_tool" for row in output)
    stats["boundary_rows"] = sum(row["source_dataset"] == "policy_boundary_v23" for row in output)
    stats["malformed_source_rows"] = malformed_source_rows
    return output, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create grounded, unrolled VN-history policy trajectories.")
    parser.add_argument("--input", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output", default="datasets/research_agent/history_trajectories.jsonl")
    parser.add_argument("--include-no-tool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-boundaries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-source-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_rows = load_messages(args.input)
    if args.max_source_rows is not None:
        source_rows = source_rows[: max(args.max_source_rows, 0)]
    rows, stats = build_history_dataset(
        source_rows,
        include_no_tool=args.include_no_tool,
        include_boundaries=args.include_boundaries,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print(f"Wrote {write_jsonl(args.output, rows)} grounded state-action rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
