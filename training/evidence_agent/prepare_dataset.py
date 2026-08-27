from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from typing import Any

from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.schemas import EvidenceAgentRequest, EvidenceModelOutput, SelectedEvidence
from training.common.datasets import first_user_assistant, load_messages
from training.common.jsonl import write_jsonl
from training.history_answerer.evaluate import parse_source_ids


CONTEXT_RE = re.compile(r"(?ms)^\[([^\]]+)\]\s*([^\n]*)\n(.*?)(?=^\[[^\]]+\]|\Z)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
MULTIPART_RE = re.compile(r"\b(và|so sánh|khác.+ở điểm|những.+nào|các.+nào)\b", re.I)
GENERIC_SUMMARIES = {
    "Evidence đã được lọc từ context huấn luyện.",
    "Evidence không đủ.",
}


class InvalidSourceRowError(ValueError):
    """A legacy RAG row cannot be converted without inventing or remapping evidence."""
STOPWORDS = {
    "ai", "có", "cho", "của", "đã", "được", "gì", "khi", "là", "nào", "như", "ở", "theo",
    "thế", "trong", "từ", "và", "vào", "về", "với", "những", "các", "một", "sau", "năm",
}


def parse_question_and_evidence(user_text: str) -> tuple[str, list[dict[str, Any]]]:
    question_part, _, context_part = user_text.partition("Tài liệu tham khảo:")
    question = question_part.replace("Câu hỏi:", "", 1).strip()
    evidence = [
        {
            "evidence_id": match.group(1).strip(),
            "source_type": "local",
            "title": match.group(2).strip() or None,
            "url": None,
            "chunk_id": match.group(1).strip(),
            "text": match.group(3).strip(),
            "retrieval_score": None,
        }
        for match in CONTEXT_RE.finditer(context_part)
    ]
    return question, evidence


def extract_answer_text(assistant_text: str) -> str:
    _, marker, answer = assistant_text.partition("Trả lời:")
    value = answer if marker else assistant_text
    return " ".join(value.strip().split())


def stable_source_hash(row: dict[str, Any]) -> str:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in TOKEN_RE.finditer(text)
        if len(match.group(0)) > 1 and match.group(0).casefold() not in STOPWORDS
    }


def grounded_compression(question: str, answer: str, source_text: str, *, max_chars: int = 600) -> tuple[list[str], str]:
    """Select extractive, question-relevant sentences; every claim remains a source substring."""
    sentences = [" ".join(item.split()) for item in SENTENCE_RE.split(source_text) if len(" ".join(item.split())) >= 20]
    if not sentences:
        fallback = " ".join(source_text.split())[:max_chars].strip()
        return ([fallback] if fallback else []), fallback
    query_tokens = _tokens(question)
    answer_tokens = _tokens(answer)
    ranked: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_tokens = _tokens(sentence)
        score = 2.0 * len(sentence_tokens & query_tokens) + len(sentence_tokens & answer_tokens)
        score += 0.5 * len(re.findall(r"\b\d{3,4}\b", sentence))
        ranked.append((score, index, sentence))
    chosen = sorted(ranked, key=lambda item: (-item[0], len(item[2]), item[1]))[:3]
    chosen.sort(key=lambda item: item[1])
    claims: list[str] = []
    used = 0
    for _, _, sentence in chosen:
        remaining = max_chars - used - (1 if claims else 0)
        if remaining <= 0:
            break
        claim = sentence[:remaining].strip()
        if claim:
            claims.append(claim)
            used += len(claim)
    compressed = " ".join(claims).strip()
    return claims, compressed


def _id_component(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_-]+", "-", value.casefold()).strip("-")
    return normalized or fallback


def _selected_item(
    evidence_id: str,
    by_id: dict[str, dict[str, Any]],
    question: str,
    answer: str,
    *,
    max_chars: int,
) -> SelectedEvidence:
    source = by_id[evidence_id]
    claims, compressed = grounded_compression(question, answer, str(source["text"]), max_chars=max_chars)
    if not claims or not compressed:
        raise ValueError(f"selected evidence {evidence_id!r} produced empty compression")
    return SelectedEvidence(evidence_id=evidence_id, relevance=1.0, claims=claims, compressed_text=compressed)


def _base_record(row: dict[str, Any], *, compression_max_chars: int) -> dict[str, Any]:
    user_text, assistant_text = first_user_assistant(row)
    question, evidence = parse_question_and_evidence(user_text)
    if not question or not evidence:
        raise InvalidSourceRowError(f"source row {row.get('id')} has an empty question/evidence pool")
    selected_ids = list(dict.fromkeys(parse_source_ids(assistant_text)))
    by_id = {item["evidence_id"]: item for item in evidence}
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise InvalidSourceRowError(f"source row {row.get('id')} selects unknown IDs: {unknown}")
    answer = extract_answer_text(assistant_text)
    selected = [
        _selected_item(evidence_id, by_id, question, answer, max_chars=compression_max_chars)
        for evidence_id in selected_ids
    ]
    if selected:
        output = EvidenceModelOutput(
            status="sufficient",
            selected_evidence=selected,
            conflicts=[],
            missing_information=[],
            summary=(
                f"Evidence {', '.join(selected_ids)} cung cấp dữ kiện trực tiếp để trả lời câu hỏi: "
                f"{question}"
            ),
        )
    else:
        output = EvidenceModelOutput(
            status="insufficient",
            selected_evidence=[],
            conflicts=[],
            missing_information=["Evidence được cung cấp không nêu đủ thông tin để trả lời chắc chắn."],
            summary=f"Evidence được cung cấp chưa đủ để trả lời câu hỏi: {question}",
        )
    return {
        "source": row,
        "question": question,
        "evidence": evidence,
        "answer": answer,
        "output": output,
        "source_hash": stable_source_hash(row),
    }


def _mutate_fact(text: str) -> tuple[str, str, str] | None:
    match = re.search(r"(?<!\d)(\d{3,4})(?!\d)", text)
    if match is None:
        match = re.search(r"(?<!\d)(\d{1,2})(?!\d)", text)
    if match is None:
        return None
    old = match.group(1)
    value = int(old)
    new = str(value + 1 if value < 9999 else value - 1).zfill(len(old))
    return text[: match.start()] + new + text[match.end() :], old, new


def _apply_duplicate(record: dict[str, Any]) -> tuple[list[dict[str, Any]], EvidenceModelOutput, dict[str, Any]]:
    evidence = [dict(item) for item in record["evidence"]]
    selected = record["output"].selected_evidence[0]
    source = next(item for item in evidence if item["evidence_id"] == selected.evidence_id)
    exact_id = f"{selected.evidence_id}__dup_exact"
    overlap_id = f"{selected.evidence_id}__dup_overlap"
    paraphrase_id = f"{selected.evidence_id}__dup_paraphrase"
    cross_source_id = f"{selected.evidence_id}__dup_cross_source"
    evidence.extend([
        {
            **source,
            "evidence_id": exact_id,
            "chunk_id": exact_id,
            "source_type": "synthetic_duplicate_exact",
        },
        {
            **source,
            "evidence_id": overlap_id,
            "chunk_id": overlap_id,
            "source_type": "synthetic_duplicate_overlap",
            "text": selected.compressed_text,
        },
        {
            **source,
            "evidence_id": paraphrase_id,
            "chunk_id": paraphrase_id,
            "source_type": "synthetic_duplicate_paraphrase",
            "text": f"Theo evidence nguồn, {selected.compressed_text}",
        },
        {
            **source,
            "evidence_id": cross_source_id,
            "chunk_id": cross_source_id,
            "source_type": "synthetic_duplicate_cross_source",
            "text": selected.compressed_text,
            "url": None,
        },
    ])
    return evidence, record["output"], {
        "synthetic_duplicate": True,
        "duplicate_types": ["exact", "overlapping_chunk", "attribution_paraphrase", "cross_source_simulation"],
        "duplicate_of": selected.evidence_id,
    }


def _apply_conflict(record: dict[str, Any]) -> tuple[list[dict[str, Any]], EvidenceModelOutput, dict[str, Any]]:
    evidence = [dict(item) for item in record["evidence"]]
    original = record["output"].selected_evidence[0]
    mutation = _mutate_fact(original.compressed_text)
    if mutation is None:
        raise ValueError("conflict candidate has no controllable numeric factual slot")
    corrupted, old, new = mutation
    conflict_id = f"{original.evidence_id}__conflict"
    source = next(item for item in evidence if item["evidence_id"] == original.evidence_id)
    evidence.append({
        **source,
        "evidence_id": conflict_id,
        "chunk_id": conflict_id,
        "source_type": "synthetic_conflict",
        "text": corrupted,
    })
    corrupted_item = SelectedEvidence(
        evidence_id=conflict_id,
        relevance=1.0,
        claims=[corrupted],
        compressed_text=corrupted,
    )
    output = EvidenceModelOutput(
        status="conflicting",
        selected_evidence=[original, corrupted_item],
        conflicts=[
            f"{original.evidence_id} nêu giá trị {old}, trong khi {conflict_id} nêu giá trị {new} cho cùng fact."
        ],
        missing_information=["Cần nguồn đáng tin cậy để xác minh giá trị đang mâu thuẫn."],
        summary=(
            f"Evidence cho câu hỏi “{record['question']}” mâu thuẫn giữa "
            f"{original.evidence_id} và {conflict_id}; chưa thể kết luận chắc chắn."
        ),
    )
    return evidence, output, {
        "synthetic_conflict": True,
        "corrupted_evidence_id": conflict_id,
        "original_value": old,
        "corrupted_value": new,
    }


def _apply_partial(record: dict[str, Any]) -> tuple[list[dict[str, Any]], EvidenceModelOutput, dict[str, Any]]:
    original = record["output"].selected_evidence[0]
    partial_claim = original.claims[0]
    supporting_ids = {item.evidence_id for item in record["output"].selected_evidence}
    evidence = []
    for item in record["evidence"]:
        candidate = dict(item)
        if candidate["evidence_id"] == original.evidence_id:
            candidate["text"] = partial_claim
            candidate["source_type"] = "synthetic_partial"
        elif candidate["evidence_id"] in supporting_ids:
            # A multipart partial example keeps A but removes gold chunks that
            # would still reveal B and make the input sufficient.
            continue
        evidence.append(candidate)
    partial_item = SelectedEvidence(
        evidence_id=original.evidence_id,
        relevance=0.75,
        claims=[partial_claim],
        compressed_text=partial_claim,
    )
    output = EvidenceModelOutput(
        status="insufficient",
        selected_evidence=[partial_item],
        conflicts=[],
        missing_information=[f"Thiếu evidence cho các phần còn lại của câu hỏi: {record['question']}"],
        summary=f"Evidence {original.evidence_id} chỉ hỗ trợ một phần: {partial_claim}",
    )
    return evidence, output, {"synthetic_partial": True, "retained_evidence_id": original.evidence_id}


def _format_row(record: dict[str, Any], behavior: str, *, max_selected: int) -> dict[str, Any]:
    evidence = record["evidence"]
    output: EvidenceModelOutput = record["output"]
    metadata: dict[str, Any] = {"synthetic": False}
    if behavior == "duplicate":
        evidence, output, metadata = _apply_duplicate(record)
    elif behavior == "conflict":
        evidence, output, metadata = _apply_conflict(record)
    elif behavior == "partial":
        evidence, output, metadata = _apply_partial(record)
    request = EvidenceAgentRequest(question=record["question"], max_selected=max_selected, evidence=evidence)
    legacy_id = str(record["source"].get("id") or f"source-{record['source_hash'][:12]}")
    group_id = f"evidence-history-{_id_component(legacy_id, 'unknown-group')}"
    sample_id = f"{group_id}-{behavior}-{record['source_hash'][:16]}"
    output_payload = output.model_dump()
    request_payload = request.model_dump()
    return {
        "id": sample_id,
        "source_dataset": "vn_history_phase6",
        "original_sample_id": legacy_id,
        "group_id": group_id,
        "behavior": behavior,
        "question": request.question,
        "evidence": [item.model_dump() for item in request.evidence],
        "input": request_payload,
        "output": output_payload,
        "metadata": metadata,
        "messages": [
            {"role": "system", "content": EVIDENCE_AGENT_SYSTEM},
            {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(output_payload, ensure_ascii=False, sort_keys=True)},
        ],
    }


def build_dataset_v2(
    source_rows: list[dict[str, Any]],
    *,
    duplicate_ratio: float = 0.13,
    conflict_ratio: float = 0.13,
    partial_ratio: float = 0.12,
    compression_max_chars: int = 600,
    max_selected: int = 8,
) -> list[dict[str, Any]]:
    if any(value < 0 or value > 1 for value in (duplicate_ratio, conflict_ratio, partial_ratio)):
        raise ValueError("behavior ratios must be between 0 and 1")
    if duplicate_ratio + conflict_ratio + partial_ratio >= 0.8:
        raise ValueError("synthetic behavior ratios leave too little clean/insufficient supervision")
    unique_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in source_rows:
        fingerprint = stable_source_hash(row)
        if fingerprint not in seen_hashes:
            seen_hashes.add(fingerprint)
            unique_rows.append(row)
    # Legacy source rows occasionally cite chunks that are absent from their own
    # input context. Dropping those rows is safer than silently remapping IDs or
    # fabricating compression targets. The CLI reports the count explicitly.
    records: list[dict[str, Any]] = []
    for row in unique_rows:
        try:
            records.append(_base_record(row, compression_max_chars=compression_max_chars))
        except InvalidSourceRowError:
            continue
    sufficient_indices = [index for index, item in enumerate(records) if item["output"].status == "sufficient"]

    def ordered(indices: list[int], salt: str) -> list[int]:
        return sorted(indices, key=lambda index: hashlib.sha256(f"{salt}:{records[index]['source_hash']}".encode()).hexdigest())

    assigned: dict[int, str] = {}
    conflict_eligible = [
        index for index in sufficient_indices
        if _mutate_fact(records[index]["output"].selected_evidence[0].compressed_text) is not None
    ]
    conflict_target = round(len(records) * conflict_ratio)
    for index in ordered(conflict_eligible, "conflict")[:conflict_target]:
        assigned[index] = "conflict"

    remaining = [index for index in sufficient_indices if index not in assigned]
    duplicate_target = round(len(records) * duplicate_ratio)
    for index in ordered(remaining, "duplicate")[:duplicate_target]:
        assigned[index] = "duplicate"

    remaining = [index for index in sufficient_indices if index not in assigned]
    partial_eligible = [index for index in remaining if MULTIPART_RE.search(records[index]["question"])]
    partial_target = round(len(records) * partial_ratio)
    for index in ordered(partial_eligible, "partial")[:partial_target]:
        assigned[index] = "partial"

    output: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if index in assigned:
            behavior = assigned[index]
        elif record["output"].status == "insufficient":
            behavior = "insufficient"
        elif len(record["evidence"]) > len(record["output"].selected_evidence):
            behavior = "relevant_distractor"
        else:
            behavior = "clean_relevant"
        output.append(_format_row(record, behavior, max_selected=max_selected))
    ids = [row["id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("Evidence v2 builder produced duplicate row IDs")
    return output


def build_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compatibility helper returning a canonical clean v2 row for one source item."""
    record = _base_record(row, compression_max_chars=600)
    behavior = "insufficient" if record["output"].status == "insufficient" else (
        "relevant_distractor" if len(record["evidence"]) > len(record["output"].selected_evidence) else "clean_relevant"
    )
    return _format_row(record, behavior, max_selected=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Evidence Agent v2 data using the canonical runtime model-output schema.")
    parser.add_argument("--input", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output", default="datasets/evidence_agent/train_v2.jsonl")
    parser.add_argument("--duplicate-ratio", type=float, default=0.13)
    parser.add_argument("--conflict-ratio", type=float, default=0.13)
    parser.add_argument("--partial-ratio", type=float, default=0.12)
    parser.add_argument("--compression-max-chars", type=int, default=600)
    parser.add_argument("--max-selected", type=int, default=8)
    parser.add_argument("--max-source-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_rows = load_messages(args.input)
    if args.max_source_rows is not None:
        source_rows = source_rows[: max(args.max_source_rows, 0)]
    rows = build_dataset_v2(
        source_rows,
        duplicate_ratio=args.duplicate_ratio,
        conflict_ratio=args.conflict_ratio,
        partial_ratio=args.partial_ratio,
        compression_max_chars=args.compression_max_chars,
        max_selected=args.max_selected,
    )
    status = Counter(row["output"]["status"] for row in rows)
    behavior = Counter(row["behavior"] for row in rows)
    count = write_jsonl(args.output, rows)
    print(json.dumps({
        "source_rows": len(source_rows),
        "rows": count,
        "dropped_source_rows": len(source_rows) - count,
        "status": status,
        "behavior": behavior,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
