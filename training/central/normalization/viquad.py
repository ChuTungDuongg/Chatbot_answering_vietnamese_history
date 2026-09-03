from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from training.trajectory_dataset.schema import (
    CENTRAL_V2_SYSTEM_PROMPT,
    QWEN3_TOOL_TEMPLATE_CONTRACT,
    SEARCH_HISTORY_TOOL,
    make_trajectory,
    tool_call,
)
from training.trajectory_dataset.adapters.common import AdapterError, provenance, source_key, trajectory_id


UIT_VIQUAD2_DATASET_ID = "taidng/UIT-ViQuAD2.0"


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text).casefold())
    return "".join(character for character in value if unicodedata.category(character) != "Mn").replace("đ", "d")


_HISTORY_SIGNALS: dict[str, tuple[str, ...]] = {
    "dynasty_monarchy": ("trieu dai", "nha ly", "nha tran", "nha le", "nha nguyen", "vua ", "hoang de", "de vuong"),
    "war_battle": ("chien tranh", "tran danh", "chien dich", "quan xam luoc", "khang chien", "quan su"),
    "uprising_revolution": ("khoi nghia", "cach mang", "phong trao dau tranh", "noi day"),
    "treaty_colonial": ("hiep dinh", "hiep uoc", "thuoc dia", "phap thuoc", "thuc dan"),
    "historical_state": ("vuong quoc", "quoc gia co", "chinh quyen", "trieu dinh", "lich su viet nam"),
    "political_biography": ("la ai", "giu chuc", "lanh dao", "anh hung dan toc", "tuong linh", "chu tich nuoc"),
    "period_event": ("thoi ky", "giai doan lich su", "su kien lich su", "doc lap dan toc", "thong nhat dat nuoc"),
}


@dataclass(frozen=True)
class HistoryFilterResult:
    accepted: bool
    score: int
    reasons: tuple[str, ...]


def history_relevance(row: dict[str, Any], *, threshold: int = 4) -> HistoryFilterResult:
    if threshold < 1:
        raise ValueError("history filter threshold must be positive")
    title = _fold(str(row.get("title") or ""))
    question = _fold(str(row.get("question") or ""))
    context = _fold(str(row.get("context") or ""))
    score = 0
    reasons: list[str] = []
    strong_locations = 0
    for category, phrases in _HISTORY_SIGNALS.items():
        title_or_question = any(phrase in title or phrase in question for phrase in phrases)
        context_match = any(phrase in context for phrase in phrases)
        if title_or_question:
            score += 3
            strong_locations += 1
            reasons.append(f"title_or_question:{category}")
        elif context_match:
            score += 1
            reasons.append(f"context:{category}")
    # A year alone is intentionally never a positive signal.
    accepted = score >= threshold and (strong_locations > 0 or len(reasons) >= 2)
    return HistoryFilterResult(accepted=accepted, score=score, reasons=tuple(reasons))


def _answers(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        texts = value.get("text") or []
        starts = value.get("answer_start") or []
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(starts, int):
            starts = [starts]
        return [
            {"text": text, "answer_start": starts[index] if index < len(starts) else None}
            for index, text in enumerate(texts)
        ]
    return []


def _source_id(row: dict[str, Any], index: int) -> str:
    payload = json.dumps(source_key(row, index), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "viquad_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:18]


def answer_sentence(context: str, answer_text: str, answer_start: int | None) -> str:
    text = str(context)
    answer = str(answer_text).strip()
    if not answer:
        raise AdapterError("ViQuAD answer text is empty")
    start = int(answer_start) if isinstance(answer_start, int) else -1
    if start < 0 or text[start : start + len(answer)] != answer:
        start = text.find(answer)
    if start < 0:
        raise AdapterError("ViQuAD answer span cannot be located in context")
    left = max(text.rfind(mark, 0, start) for mark in (".", "!", "?", "\n")) + 1
    endpoints = [position for mark in (".", "!", "?", "\n") if (position := text.find(mark, start + len(answer))) >= 0]
    right = min(endpoints) + 1 if endpoints else len(text)
    sentence = " ".join(text[left:right].split()).strip()
    if answer not in sentence:
        raise AdapterError("ViQuAD extracted evidence sentence lost the answer span")
    if len(sentence) > 700:
        sentence = answer
    return sentence


def normalize_uit_viquad2(
    row: dict[str, Any],
    *,
    index: int,
    split: str = "train",
    history_threshold: int = 4,
    top_k: int = 6,
) -> dict[str, Any]:
    question = " ".join(str(row.get("question") or "").split())
    context = " ".join(str(row.get("context") or "").split())
    title = " ".join(str(row.get("title") or "").split())
    if not question or not context:
        raise AdapterError("ViQuAD row requires non-empty question and context")
    relevance = history_relevance(row, threshold=history_threshold)
    if not relevance.accepted:
        raise AdapterError(f"ViQuAD non-history row rejected (score={relevance.score})")
    bounded_top_k = min(max(int(top_k), 1), 10)
    source_id = _source_id(row, index)
    call_id = "call_search_history_0001"
    observation = {
        "observation_origin": "uit_viquad2_ground_truth_context",
        "results": [{
            "chunk_id": source_id,
            "title": title,
            "text": context,
            "source_kind": "viquad_context",
        }],
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CENTRAL_V2_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call(call_id, "search_history", {"query": question, "top_k": bounded_top_k})],
        },
        {
            "role": "tool",
            "name": "search_history",
            "tool_call_id": call_id,
            "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    impossible = bool(row.get("is_impossible", False))
    answer_rows = _answers(row.get("answers"))
    if impossible:
        final = "Bằng chứng được cung cấp không đủ để trả lời chắc chắn câu hỏi này."
        task_type = "history_qa_unanswerable"
        answer_text = None
    else:
        if not answer_rows:
            raise AdapterError("Answerable ViQuAD row has no answer spans")
        selected = answer_rows[0]
        answer_text = str(selected.get("text") or "").strip()
        sentence = answer_sentence(context, answer_text, selected.get("answer_start"))
        final = f"{sentence} [{source_id}]"
        task_type = "history_qa_answerable"
    messages.append({"role": "assistant", "content": final})

    source_provenance = provenance(
        dataset_id=UIT_VIQUAD2_DATASET_ID,
        split=split,
        row=row,
        index=index,
        license_name=None,
        transformations=[
            "conservative_history_filter", "ground_truth_context_as_marked_observation",
            "mandatory_search_history_first_turn", "deterministic_evidence_sentence",
            "qwen3_enable_thinking_false",
        ],
    )
    source_provenance.update({
        "grounded": True,
        "evidence_ids": [] if impossible else [source_id],
        "observation_origin": "uit_viquad2_ground_truth_context",
        "answerable": not impossible,
        "is_impossible": impossible,
        "answer_text": answer_text,
        "plausible_answers": copy.deepcopy(row.get("plausible_answers") or []),
        "history_filter_score": relevance.score,
        "history_filter_reasons": list(relevance.reasons),
        "source_group": f"viquad_title:{_fold(title)}",
        "chat_template_contract": copy.deepcopy(QWEN3_TOOL_TEMPLATE_CONTRACT),
    })
    return make_trajectory(
        trajectory_id=trajectory_id(UIT_VIQUAD2_DATASET_ID, row, index),
        source_dataset="uit_viquad2_grounded",
        task_type=task_type,
        messages=messages,
        tools=[copy.deepcopy(SEARCH_HISTORY_TOOL)],
        provenance=source_provenance,
        difficulty="medium" if impossible else "easy",
    )

