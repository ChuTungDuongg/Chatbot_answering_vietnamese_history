from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..citations import format_evidence_citation
from ..schema import SEARCH_HISTORY_TOOL, make_trajectory, tool_call
from .common import AdapterError, get_messages, provenance, semantic_messages, trajectory_id


DATASET_ID = "minhxthanh/Vietnam-History-200K-Vi"
PREFERRED_PATTERNS = re.compile(
    r"nguyên nhân|ý nghĩa|tác động|hậu quả|bối cảnh|so sánh|tóm tắt|trình bày|vai trò|đóng góp",
    flags=re.IGNORECASE,
)
SHORTNESS_PATTERNS = re.compile(r"thật ngắn gọn|rất ngắn|trả lời súc tích|trình bày nhanh", flags=re.IGNORECASE)


def messages_without_assistant_analysis(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Drop source-declared assistant analysis before canonical role conversion.

    The channel is observable source metadata.  It must be handled here rather
    than guessed later from Vietnamese wording, because every canonical
    assistant message is a supervised target.
    """
    kept: list[dict[str, Any]] = []
    dropped = 0
    for message in get_messages(row):
        if not isinstance(message, dict):
            kept.append(message)
            continue
        role = str(message.get("role") or message.get("from") or "").casefold()
        channel = str(message.get("channel") or "").casefold()
        if role in {"assistant", "gpt"} and channel == "analysis":
            dropped += 1
            continue
        kept.append(message)
    return kept, dropped


def canonical_analysis_messages_remaining(row: dict[str, Any]) -> int:
    """Count observable Vietnam-History analysis leakage in canonical rows.

    Newly normalized rows retain the source-channel decision in provenance.
    Legacy canonical rows lost ``channel`` entirely, but their invalid shape is
    still observable as multiple supervised assistant text targets.
    """
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    explicit = sum(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and str(message.get("channel") or "").casefold() == "analysis"
        for message in messages
    )
    assistant_text_targets = sum(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and not message.get("tool_calls")
        and bool(str(message.get("content") or "").strip())
        for message in messages
    )
    return max(explicit, max(0, assistant_text_targets - 1))


def question_and_answer(messages: list[dict[str, Any]]) -> tuple[str, str]:
    question = next((str(message.get("content") or "").strip() for message in messages if message.get("role") == "user"), "")
    answer = next((str(message.get("content") or "").strip() for message in reversed(messages) if message.get("role") == "assistant"), "")
    if not question or not answer:
        raise AdapterError("Vietnam-History row requires non-empty user and assistant messages")
    return question, answer


def quality_filter(
    row: dict[str, Any],
    *,
    min_answer_words: int = 12,
    max_answer_words: int = 800,
    preferred_only: bool = False,
) -> str | None:
    source_messages, _ = messages_without_assistant_analysis(row)
    messages = semantic_messages(source_messages, include_reasoning=False)
    question, answer = question_and_answer(messages)
    words = len(answer.split())
    if words < min_answer_words:
        return f"answer too short: {words} words"
    if words > max_answer_words:
        return f"answer too long: {words} words"
    if SHORTNESS_PATTERNS.search(question) and not PREFERRED_PATTERNS.search(question):
        return "low-depth short-answer instruction"
    if preferred_only and not PREFERRED_PATTERNS.search(question):
        return "question is outside preferred analytical history task types"
    return None


def normalize_vietnam_history(
    row: dict[str, Any],
    *,
    index: int = 0,
    split: str = "train",
    mode: str = "style_only",
    retrieval_results: list[dict[str, Any]] | None = None,
    preferred_only: bool = False,
) -> dict[str, Any]:
    if mode not in {"style_only", "rag_grounded"}:
        raise AdapterError(f"unsupported Vietnam-History mode: {mode}")
    quality_reason = quality_filter(row, preferred_only=preferred_only)
    if quality_reason:
        raise AdapterError(quality_reason)
    source_messages, dropped_analysis = messages_without_assistant_analysis(row)
    original_messages = semantic_messages(source_messages, include_reasoning=False)
    question, answer = question_and_answer(original_messages)
    transformations = ["messages_to_canonical", f"mode:{mode}"]
    if dropped_analysis:
        transformations.append("dropped_assistant_analysis_channel")
    prov = provenance(
        dataset_id=DATASET_ID,
        split=split,
        row=row,
        index=index,
        license_name="MIT",
        transformations=transformations,
    )
    prov["dropped_assistant_analysis_messages"] = dropped_analysis
    if mode == "style_only":
        messages = original_messages
        tools: list[dict[str, Any]] = []
        task_type = "vietnamese_history_style"
    else:
        if not retrieval_results:
            raise AdapterError("rag_grounded mode requires non-empty real or precomputed retrieval_results")
        evidence_ids = [str(item.get("chunk_id") or "") for item in retrieval_results if item.get("chunk_id")]
        if not evidence_ids:
            raise AdapterError("rag_grounded retrieval results contain no chunk_id values")
        call_id = "call_search_history_0001"
        citations = " ".join(format_evidence_citation(evidence_id) for evidence_id in evidence_ids[:3])
        messages = [
            {"role": "system", "content": "Trả lời câu hỏi lịch sử Việt Nam từ bằng chứng công cụ và trích dẫn ID nguồn."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": None, "tool_calls": [tool_call(call_id, "search_history", {"query": question, "top_k": 8})]},
            {"role": "tool", "name": "search_history", "tool_call_id": call_id, "content": json.dumps(retrieval_results, ensure_ascii=False)},
            {"role": "assistant", "content": f"{answer}\n\nBằng chứng truy xuất: {citations}"},
        ]
        tools = [SEARCH_HISTORY_TOOL]
        task_type = "vietnamese_history_rag_grounded"
        prov.update({"grounded": True, "evidence_ids": evidence_ids, "source_group": evidence_ids[0]})
    return make_trajectory(
        trajectory_id=trajectory_id(f"{DATASET_ID}:{mode}", row, index),
        source_dataset="vietnam_history_200k",
        task_type=task_type,
        messages=messages,
        tools=tools,
        difficulty="medium" if PREFERRED_PATTERNS.search(question) else "easy",
        provenance=prov,
    )
