from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from typing import Any

from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.schemas import EvidenceAgentRequest, EvidenceModelOutput, SelectedEvidence
from training.common.datasets import first_user_assistant, load_messages
from training.common.jsonl import write_jsonl
from training.evidence_agent.coverage import (
    CoverageAssessment,
    assess_answer_coverage,
    evidence_relevance,
    specific_missing_information,
)
from training.evidence_agent.conflicts import (
    ConflictMutation,
    propose_irrelevant_disagreement,
    propose_question_relevant_conflict,
)
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
    return SelectedEvidence(
        evidence_id=evidence_id,
        relevance=evidence_relevance(question, str(source["text"])),
        claims=claims,
        compressed_text=compressed,
    )


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
    selected_source_texts = [str(by_id[item.evidence_id]["text"]) for item in selected]
    coverage = assess_answer_coverage(question, answer, selected_source_texts)
    return {
        "source": row,
        "question": question,
        "evidence": evidence,
        "answer": answer,
        "output": output,
        "source_hash": stable_source_hash(row),
        "base_coverage": coverage,
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


def _opaque_evidence_id(record: dict[str, Any], parent_id: str, purpose: str) -> str:
    digest = hashlib.sha256(
        f"{record['source_hash']}:{parent_id}:{purpose}".encode("utf-8")
    ).hexdigest()[:20]
    return f"ev_{digest}"


def _opaque_chunk_id(record: dict[str, Any], parent_id: str, purpose: str) -> str:
    digest = hashlib.sha256(
        f"chunk:{record['source_hash']}:{parent_id}:{purpose}".encode("utf-8")
    ).hexdigest()[:20]
    return f"chunk_{digest}"


def _date_paraphrase(text: str) -> str:
    word_date = re.search(
        r"\bngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{3,4})\b",
        text,
        re.I,
    )
    if word_date:
        replacement = f"ngày {word_date.group(1)}/{word_date.group(2)}/{word_date.group(3)}"
        return text[: word_date.start()] + replacement + text[word_date.end() :]
    slash_date = re.search(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{3,4})(?!\d)", text)
    if slash_date:
        replacement = (
            f"{slash_date.group(1)} tháng {slash_date.group(2)} năm {slash_date.group(3)}"
        )
        return text[: slash_date.start()] + replacement + text[slash_date.end() :]
    return f"Theo nguồn này, {text}"


def _apply_duplicate(record: dict[str, Any]) -> tuple[list[dict[str, Any]], EvidenceModelOutput, dict[str, Any]]:
    evidence = [dict(item) for item in record["evidence"]]
    selected = record["output"].selected_evidence[0]
    source = next(item for item in evidence if item["evidence_id"] == selected.evidence_id)
    exact_id = _opaque_evidence_id(record, selected.evidence_id, "duplicate-exact")
    overlap_id = _opaque_evidence_id(record, selected.evidence_id, "duplicate-overlap")
    paraphrase_id = _opaque_evidence_id(record, selected.evidence_id, "duplicate-paraphrase")
    cross_source_id = _opaque_evidence_id(record, selected.evidence_id, "duplicate-cross-source")
    evidence.extend([
        {
            **source,
            "evidence_id": exact_id,
            "chunk_id": _opaque_chunk_id(record, selected.evidence_id, "duplicate-exact"),
        },
        {
            **source,
            "evidence_id": overlap_id,
            "chunk_id": _opaque_chunk_id(record, selected.evidence_id, "duplicate-overlap"),
            "text": selected.compressed_text,
        },
        {
            **source,
            "evidence_id": paraphrase_id,
            "chunk_id": _opaque_chunk_id(record, selected.evidence_id, "duplicate-paraphrase"),
            "text": _date_paraphrase(selected.compressed_text),
        },
        {
            **source,
            "evidence_id": cross_source_id,
            "chunk_id": _opaque_chunk_id(record, selected.evidence_id, "duplicate-cross-source"),
            "text": selected.compressed_text,
            "url": None,
        },
    ])
    return evidence, record["output"], {
        "synthetic": True,
        "augmentation_type": "duplicate",
        "duplicate_types": ["exact", "overlapping_chunk", "attribution_paraphrase", "cross_source_simulation"],
        "parent_evidence_id": selected.evidence_id,
        "augmented_evidence_ids": [exact_id, overlap_id, paraphrase_id, cross_source_id],
    }


def _apply_conflict(
    record: dict[str, Any],
    variant: tuple[str, ConflictMutation],
) -> tuple[list[dict[str, Any]], EvidenceModelOutput, dict[str, Any]]:
    evidence = [dict(item) for item in record["evidence"]]
    parent_id, mutation = variant
    source = next(item for item in evidence if item["evidence_id"] == parent_id)
    conflict_id = _opaque_evidence_id(record, parent_id, f"conflict-{mutation.slot_key}")
    evidence.append({
        **source,
        "evidence_id": conflict_id,
        "chunk_id": _opaque_chunk_id(record, parent_id, f"conflict-{mutation.slot_key}"),
        "text": mutation.mutated_claim,
    })
    original_item = SelectedEvidence(
        evidence_id=parent_id,
        relevance=evidence_relevance(record["question"], mutation.original_claim),
        claims=[mutation.original_claim],
        compressed_text=mutation.original_claim,
    )
    corrupted_item = SelectedEvidence(
        evidence_id=conflict_id,
        relevance=evidence_relevance(record["question"], mutation.mutated_claim),
        claims=[mutation.mutated_claim],
        compressed_text=mutation.mutated_claim,
    )
    output = EvidenceModelOutput(
        status="conflicting",
        selected_evidence=[original_item, corrupted_item],
        conflicts=[
            f"{parent_id} nêu {mutation.slot_label} là {mutation.original_value}, trong khi "
            f"{conflict_id} nêu {mutation.slot_label} là {mutation.mutated_value}."
        ],
        missing_information=[f"Cần xác minh {mutation.slot_label} đang có hai giá trị không tương thích."],
        summary=(
            f"Evidence cho câu hỏi “{record['question']}” mâu thuẫn giữa "
            f"{parent_id} và {conflict_id}; chưa thể kết luận chắc chắn."
        ),
    )
    return evidence, output, {
        "synthetic": True,
        "augmentation_type": "conflict",
        "parent_evidence_id": parent_id,
        "augmented_evidence_ids": [conflict_id],
        "mutated_evidence_id": conflict_id,
        "mutated_slot": mutation.slot_key,
        "slot_label": mutation.slot_label,
        "conflict_type": mutation.conflict_type,
        "original_value": mutation.original_value,
        "mutated_value": mutation.mutated_value,
    }


def _apply_irrelevant_disagreement(
    record: dict[str, Any],
    variant: tuple[str, ConflictMutation],
) -> tuple[list[dict[str, Any]], EvidenceModelOutput, dict[str, Any]]:
    evidence = [dict(item) for item in record["evidence"]]
    parent_id, mutation = variant
    source = next(item for item in evidence if item["evidence_id"] == parent_id)
    augmented_id = _opaque_evidence_id(record, parent_id, "irrelevant-disagreement")
    evidence.append({
        **source,
        "evidence_id": augmented_id,
        "chunk_id": _opaque_chunk_id(record, parent_id, "irrelevant-disagreement"),
        "text": mutation.mutated_claim,
    })
    return evidence, record["output"], {
        "synthetic": True,
        "augmentation_type": "irrelevant_disagreement",
        "parent_evidence_id": parent_id,
        "augmented_evidence_ids": [augmented_id],
        "original_value": mutation.original_value,
        "mutated_value": mutation.mutated_value,
        "mutated_slot": "unrelated",
    }


def _partial_candidate(
    record: dict[str, Any],
    retained: list[SelectedEvidence],
    *,
    claim_override: tuple[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[SelectedEvidence], CoverageAssessment]:
    supporting_ids = {item.evidence_id for item in record["output"].selected_evidence}
    retained_by_id = {item.evidence_id: item for item in retained}
    evidence: list[dict[str, Any]] = []
    for item in record["evidence"]:
        candidate = dict(item)
        evidence_id = candidate["evidence_id"]
        if evidence_id not in supporting_ids:
            evidence.append(candidate)
            continue
        if evidence_id not in retained_by_id:
            continue
        if claim_override and evidence_id == claim_override[0]:
            candidate["text"] = claim_override[1]
        evidence.append(candidate)

    retained_ids = set(retained_by_id)
    retained_texts = [str(item["text"]) for item in evidence if item["evidence_id"] in retained_ids]
    retained_coverage = assess_answer_coverage(record["question"], record["answer"], retained_texts)
    # A nominal distractor may independently reveal the component being
    # ablated. Remove only those hidden-support chunks; otherwise the provided
    # pool would still be sufficient even though the gold selection is partial.
    filtered_evidence: list[dict[str, Any]] = []
    retained_components = set(retained_coverage.supported_keys)
    for item in evidence:
        if item["evidence_id"] in retained_ids:
            filtered_evidence.append(item)
            continue
        item_coverage = assess_answer_coverage(
            record["question"], record["answer"], [str(item["text"])]
        )
        if set(item_coverage.supported_keys) - retained_components:
            continue
        filtered_evidence.append(item)
    evidence = filtered_evidence

    assessment = assess_answer_coverage(
        record["question"], record["answer"], [str(item["text"]) for item in evidence]
    )
    return evidence, retained, assessment


def _propose_partial(
    record: dict[str, Any],
) -> tuple[tuple[list[dict[str, Any]], EvidenceModelOutput, dict[str, Any]] | None, dict[str, Any]]:
    base_coverage: CoverageAssessment = record["base_coverage"]
    audit_base = {
        "old_status": "insufficient",
        "old_behavior": "partial",
        "base_selected_coverage": base_coverage.as_dict(),
    }
    if not base_coverage.confident or len(base_coverage.requirements) < 2:
        original = record["output"].selected_evidence[0]
        claim = original.claims[0]
        _, _, retained_assessment = _partial_candidate(
            record,
            [
                SelectedEvidence(
                    evidence_id=original.evidence_id,
                    relevance=evidence_relevance(record["question"], claim),
                    claims=[claim],
                    compressed_text=claim,
                )
            ],
            claim_override=(original.evidence_id, claim),
        )
        if retained_assessment.full:
            outcome = "reclassified_to_sufficient"
            reason = "remaining evidence still covers complete gold answer"
        else:
            outcome = "dropped_invalid"
            reason = "no high-confidence multi-component ablation can prove partial coverage"
        return None, {
            **audit_base,
            "new_status": "sufficient",
            "audit_outcome": outcome,
            "reason": reason,
        }
    if not base_coverage.full:
        return None, {
            **audit_base,
            "new_status": "dropped",
            "audit_outcome": "dropped_invalid",
            "reason": "original selected evidence does not cover the complete gold answer",
        }

    original_selected = list(record["output"].selected_evidence)
    candidates: list[tuple[list[dict[str, Any]], list[SelectedEvidence], CoverageAssessment]] = []

    # Sentence-level ablation handles a single chunk containing separate answer
    # units (for example, date in one sentence and leader in another).
    for original in original_selected:
        for claim in original.claims:
            selected = SelectedEvidence(
                evidence_id=original.evidence_id,
                relevance=evidence_relevance(record["question"], claim),
                claims=[claim],
                compressed_text=claim,
            )
            candidates.append(
                _partial_candidate(record, [selected], claim_override=(original.evidence_id, claim))
            )

    # Chunk-level ablation handles A/B support distributed across chunks.
    for count in range(1, len(original_selected)):
        for subset in itertools.combinations(original_selected, count):
            candidates.append(_partial_candidate(record, list(subset)))

    def selected_coverage(
        item: tuple[list[dict[str, Any]], list[SelectedEvidence], CoverageAssessment]
    ) -> CoverageAssessment:
        evidence, selected, _ = item
        by_id = {candidate["evidence_id"]: candidate for candidate in evidence}
        return assess_answer_coverage(
            record["question"],
            record["answer"],
            [str(by_id[value.evidence_id]["text"]) for value in selected if value.evidence_id in by_id],
        )

    valid = [item for item in candidates if item[2].partial and selected_coverage(item).useful]
    if not valid:
        outcome = "reclassified_to_sufficient" if any(item[2].full for item in candidates) else "dropped_invalid"
        reason = (
            "remaining evidence still covers complete gold answer"
            if outcome == "reclassified_to_sufficient"
            else "proposed ablation retains no useful answer component"
        )
        return None, {
            **audit_base,
            "new_status": "sufficient",
            "audit_outcome": outcome,
            "reason": reason,
        }

    evidence, selected, assessment = sorted(
        valid,
        key=lambda item: (
            -len(item[2].supported_keys),
            tuple(
                index
                for index, requirement in enumerate(item[2].requirements)
                if requirement.key in item[2].supported_keys
            ),
            len(item[1]),
            tuple(selected.evidence_id for selected in item[1]),
        ),
    )[0]
    missing = specific_missing_information(assessment)
    output = EvidenceModelOutput(
        status="insufficient",
        selected_evidence=selected,
        conflicts=[],
        missing_information=missing,
        summary=(
            f"Evidence đã chọn hỗ trợ {len(assessment.supported_keys)}/{len(assessment.requirements)} "
            f"thành phần trả lời; còn thiếu: {'; '.join(missing)}"
        ),
    )
    audit = {
        **audit_base,
        "new_status": "insufficient",
        "audit_outcome": "remain_true_partial",
        "reason": "supporting-unit ablation causes verified answer-component loss",
        "remaining_coverage": assessment.as_dict(),
    }
    return (
        evidence,
        output,
        {
            "synthetic_partial": True,
            "retained_evidence_ids": [item.evidence_id for item in selected],
            "coverage_audit": audit,
        },
    ), audit


def _format_row(record: dict[str, Any], behavior: str, *, max_selected: int) -> dict[str, Any]:
    evidence = record["evidence"]
    output: EvidenceModelOutput = record["output"]
    metadata: dict[str, Any] = {"synthetic": False, "gold_answer": record["answer"]}
    if record.get("coverage_audit"):
        metadata["coverage_audit"] = record["coverage_audit"]
    if behavior == "duplicate":
        evidence, output, behavior_metadata = _apply_duplicate(record)
        metadata.update(behavior_metadata)
    elif behavior == "conflict":
        evidence, output, behavior_metadata = _apply_conflict(record, record["conflict_variant"])
        metadata.update(behavior_metadata)
    elif behavior == "irrelevant_disagreement":
        evidence, output, behavior_metadata = _apply_irrelevant_disagreement(
            record, record["irrelevant_disagreement_variant"]
        )
        metadata.update(behavior_metadata)
    elif behavior == "partial":
        evidence, output, behavior_metadata = record["partial_variant"]
        metadata.update(behavior_metadata)
    request = EvidenceAgentRequest(question=record["question"], max_selected=max_selected, evidence=evidence)
    candidates = {item.evidence_id: item for item in request.evidence}
    input_coverage = assess_answer_coverage(
        record["question"], record["answer"], [item.text for item in request.evidence]
    )
    selected_coverage = assess_answer_coverage(
        record["question"],
        record["answer"],
        [candidates[item.evidence_id].text for item in output.selected_evidence if item.evidence_id in candidates],
    )
    metadata["semantic_coverage"] = {
        "input": input_coverage.as_dict(),
        "selected": selected_coverage.as_dict(),
    }
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
    audit_report: dict[str, Any] | None = None,
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
    malformed_source_rows = 0
    for row in unique_rows:
        try:
            records.append(_base_record(row, compression_max_chars=compression_max_chars))
        except InvalidSourceRowError:
            malformed_source_rows += 1
            continue
    sufficient_indices = [index for index, item in enumerate(records) if item["output"].status == "sufficient"]
    coverage_invalid_indices = {
        index
        for index in sufficient_indices
        if records[index]["base_coverage"].confident and not records[index]["base_coverage"].full
    }
    suspicious_heuristic_indices = {
        index
        for index in sufficient_indices
        if not records[index]["base_coverage"].confident and not records[index]["base_coverage"].full
    }

    def ordered(indices: list[int], salt: str) -> list[int]:
        return sorted(indices, key=lambda index: hashlib.sha256(f"{salt}:{records[index]['source_hash']}".encode()).hexdigest())

    def diversified_conflict_order(indices: list[int]) -> list[int]:
        buckets: dict[str, list[int]] = {}
        for index in indices:
            conflict_type = records[index]["conflict_variant"][1].conflict_type
            buckets.setdefault(conflict_type, []).append(index)
        for conflict_type, values in buckets.items():
            buckets[conflict_type] = ordered(values, f"conflict-{conflict_type}")
        result: list[int] = []
        while any(buckets.values()):
            for conflict_type in sorted(buckets):
                if buckets[conflict_type]:
                    result.append(buckets[conflict_type].pop(0))
        return result

    # Reconstruct the v2 assignment order solely to audit all 119 rows that
    # were previously labelled partial. Final v2.1 assignment is computed
    # separately below so semantic partials can be prioritized.
    legacy_assigned: dict[int, str] = {}
    legacy_conflict_eligible = [
        index for index in sufficient_indices
        if _mutate_fact(records[index]["output"].selected_evidence[0].compressed_text) is not None
    ]
    conflict_target = round(len(records) * conflict_ratio)
    for index in ordered(legacy_conflict_eligible, "conflict")[:conflict_target]:
        legacy_assigned[index] = "conflict"

    remaining = [index for index in sufficient_indices if index not in legacy_assigned]
    duplicate_target = round(len(records) * duplicate_ratio)
    for index in ordered(remaining, "duplicate")[:duplicate_target]:
        legacy_assigned[index] = "duplicate"

    remaining = [index for index in sufficient_indices if index not in legacy_assigned]
    partial_eligible = [index for index in remaining if MULTIPART_RE.search(records[index]["question"])]
    partial_target = round(len(records) * partial_ratio)
    old_partial_indices = ordered(partial_eligible, "partial")[:partial_target]
    partial_outcomes: Counter[str] = Counter()
    for index in old_partial_indices:
        variant, audit = _propose_partial(records[index])
        partial_outcomes[audit["audit_outcome"]] += 1
        if variant is None:
            records[index]["coverage_audit"] = audit

    # V2.1 prioritizes every semantically verified partial candidate (up to the
    # configured cap), then backfills conflict/duplicate quotas from other
    # rows. Previously broken partial rows are protected from unrelated
    # synthetic relabelling so their corrected sufficient targets remain clear.
    assigned: dict[int, str] = {}
    safe_partial_variants: dict[int, tuple[list[dict[str, Any]], EvidenceModelOutput, dict[str, Any]]] = {}
    for index in sufficient_indices:
        if index in coverage_invalid_indices:
            continue
        variant, audit = _propose_partial(records[index])
        if variant is not None:
            safe_partial_variants[index] = variant
            if index not in old_partial_indices:
                variant[2]["coverage_audit"] = {
                    **audit,
                    "old_status": "sufficient",
                    "old_behavior": "source_sufficient",
                    "audit_outcome": "new_true_partial",
                }
    for index in ordered(list(safe_partial_variants), "semantic-partial")[:partial_target]:
        records[index]["partial_variant"] = safe_partial_variants[index]
        assigned[index] = "partial"

    conflict_eligible: list[int] = []
    for index in sufficient_indices:
        if index in assigned or index in coverage_invalid_indices:
            continue
        variant = propose_question_relevant_conflict(
            question=records[index]["question"],
            gold_answer=records[index]["answer"],
            selected=records[index]["output"].selected_evidence,
            evidence_texts=[str(item["text"]) for item in records[index]["evidence"]],
        )
        if variant is not None:
            records[index]["conflict_variant"] = variant
            conflict_eligible.append(index)
    for index in diversified_conflict_order(conflict_eligible)[:conflict_target]:
        assigned[index] = "conflict"

    irrelevant_disagreement_eligible: list[int] = []
    for index in sufficient_indices:
        if index in assigned or index in coverage_invalid_indices:
            continue
        variant = propose_irrelevant_disagreement(
            question=records[index]["question"],
            gold_answer=records[index]["answer"],
            selected=records[index]["output"].selected_evidence,
        )
        if variant is not None:
            records[index]["irrelevant_disagreement_variant"] = variant
            irrelevant_disagreement_eligible.append(index)
    hard_negative_target = min(max(12, conflict_target // 2), len(irrelevant_disagreement_eligible))
    for index in ordered(irrelevant_disagreement_eligible, "irrelevant-disagreement")[:hard_negative_target]:
        assigned[index] = "irrelevant_disagreement"

    duplicate_eligible = [
        index
        for index in sufficient_indices
        if index not in assigned and index not in coverage_invalid_indices
    ]
    for index in ordered(duplicate_eligible, "duplicate")[:duplicate_target]:
        assigned[index] = "duplicate"

    output: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if index in coverage_invalid_indices:
            continue
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
    if audit_report is not None:
        audit_report.clear()
        audit_report.update({
            "source_rows": len(source_rows),
            "unique_source_rows": len(unique_rows),
            "malformed_source_rows": malformed_source_rows,
            "coverage_invalid_source_rows": len(coverage_invalid_indices),
            "old_partial_rows": len(old_partial_indices),
            "reclassified_to_sufficient": partial_outcomes["reclassified_to_sufficient"],
            "remain_true_partial": partial_outcomes["remain_true_partial"],
            "dropped_invalid_partial_augmentation": partial_outcomes["dropped_invalid"],
            "new_verified_partial_rows": sum(value == "partial" for value in assigned.values()),
            "legacy_numeric_conflict_candidates": len(legacy_conflict_eligible),
            "question_relevant_conflict_candidates": len(conflict_eligible),
            "regenerated_conflict_rows": sum(value == "conflict" for value in assigned.values()),
            "irrelevant_disagreement_candidates": len(irrelevant_disagreement_eligible),
            "hard_negative_irrelevant_disagreement_rows": sum(
                value == "irrelevant_disagreement" for value in assigned.values()
            ),
            "conflict_type_distribution": dict(Counter(
                records[index]["conflict_variant"][1].conflict_type
                for index, behavior in assigned.items()
                if behavior == "conflict"
            )),
            "sufficient_audited": len(sufficient_indices),
            "suspicious_insufficient_coverage": len(coverage_invalid_indices) + len(suspicious_heuristic_indices),
            "high_confidence_insufficient_coverage": len(coverage_invalid_indices),
            "heuristic_coverage_warnings": len(suspicious_heuristic_indices),
        })
    return output


def build_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compatibility helper returning a canonical clean v2 row for one source item."""
    record = _base_record(row, compression_max_chars=600)
    behavior = "insufficient" if record["output"].status == "insufficient" else (
        "relevant_distractor" if len(record["evidence"]) > len(record["output"].selected_evidence) else "clean_relevant"
    )
    return _format_row(record, behavior, max_selected=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Evidence Agent v2.2 data using the canonical runtime model-output schema.")
    parser.add_argument("--input", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output", default="datasets/evidence_agent/train.jsonl")
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
    audit_report: dict[str, Any] = {}
    rows = build_dataset_v2(
        source_rows,
        duplicate_ratio=args.duplicate_ratio,
        conflict_ratio=args.conflict_ratio,
        partial_ratio=args.partial_ratio,
        compression_max_chars=args.compression_max_chars,
        max_selected=args.max_selected,
        audit_report=audit_report,
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
        "semantic_audit": audit_report,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
