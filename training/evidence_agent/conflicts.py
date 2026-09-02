from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from app.agents.evidence.schemas import SelectedEvidence
from app.agents.evidence.validation import (
    compressed_derived_from_own_claims,
    grounded_in_source,
    normalize_grounding,
    referenced_evidence_ids,
)
from training.evidence_agent.coverage import (
    AnswerRequirement,
    accentfold,
    extract_answer_requirements,
    requirement_supported,
)


WORD_DATE_RE = re.compile(
    r"\b(?:ngày\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{3,4})\b",
    re.I,
)
SLASH_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{3,4})(?!\d)")
YEAR_RE = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")
NUMBER_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
PROPER_TOKEN = r"[A-ZÀ-ỸĐ][0-9A-Za-zÀ-ỹĐđ'’\-]*"
PROPER_PHRASE = rf"{PROPER_TOKEN}(?:\s+{PROPER_TOKEN}){{0,4}}"
VIETNAMESE_SURNAME = (
    r"Nguyễn|Trần|Lê|Phan|Võ|Vũ|Hồ|Ngô|Đinh|Lý|Mai|Dương|Tôn|Chu|Hoàng|"
    r"Cao|Trương|Lương|Đặng|Bùi|Đỗ|Phạm|Quách|Triệu|Kiều"
)
PERSON_NAME_RE = re.compile(rf"\b(?:{VIETNAMESE_SURNAME})(?:\s+{PROPER_TOKEN}){{1,4}}")
PERSON_PATTERNS = (
    re.compile(rf"\b({PROPER_PHRASE})\s+(?:đã\s+)?(?:lãnh đạo|chỉ huy|chỉ đạo|dẫn đầu)\b"),
    re.compile(rf"(?i:\bdo)\s+({PROPER_PHRASE})\s+(?:lãnh đạo|chỉ huy|chỉ đạo)\b"),
)
PLACE_PATTERNS = (
    re.compile(rf"\b(?:tại|ở|đóng đô (?:tại|ở)|diễn ra (?:tại|ở))\s+({PROPER_PHRASE})"),
)
OPPONENT_PATTERNS = (
    re.compile(
        rf"(?i:\b(?:đánh bại|chống(?: lại)?|đối đầu với))\s+"
        rf"((?:(?:quân|thực dân|nhà)\s+)?{PROPER_PHRASE})"
    ),
)
STYLE_PATTERNS = {
    "style_name": re.compile(rf"(?i:\bhiệu)\s+(?:(?i:là)\s+)?({PROPER_PHRASE})"),
    "courtesy_name": re.compile(rf"(?i:\btự)\s+(?:(?i:là)\s+)?({PROPER_PHRASE})"),
}
MODEL_VISIBLE_RESERVED_MARKERS = (
    "__conflict",
    "__dup_",
    "synthetic_conflict",
    "synthetic_duplicate",
    "synthetic_partial",
    "augmentation_type",
    "parent_evidence_id",
    '"behavior"',
)


@dataclass(frozen=True)
class ConflictMutation:
    slot_key: str
    slot_label: str
    conflict_type: str
    original_value: str
    mutated_value: str
    original_claim: str
    mutated_claim: str

    def as_metadata(self) -> dict[str, str]:
        return asdict(self)


def _replace_match(text: str, match: re.Match[str], replacement: str, *, group: int = 0) -> str:
    start, end = match.span(group)
    return text[:start] + replacement + text[end:]


def _different_number(value: str, *, day: bool = False) -> str:
    number = int(value)
    if day:
        changed = number + 1 if number < 28 else number - 1
    else:
        changed = number + 1 if number < 9999 else number - 1
    return str(changed).zfill(len(value))


def _full_date_mutation(requirement: AnswerRequirement, claim: str, question: str) -> ConflictMutation | None:
    for pattern in (WORD_DATE_RE, SLASH_DATE_RE):
        match = pattern.search(claim)
        if not match:
            continue
        original = match.group(0)
        if not grounded_in_source(original, requirement.answer_fragment) or grounded_in_source(original, question):
            continue
        changed_day = _different_number(match.group(1), day=True)
        mutated = _replace_match(original, re.search(re.escape(match.group(1)), original), changed_day)
        return ConflictMutation(
            slot_key=requirement.key,
            slot_label=requirement.label,
            conflict_type="date",
            original_value=original,
            mutated_value=mutated,
            original_claim=claim,
            mutated_claim=_replace_match(claim, match, mutated),
        )
    return None


def _numeric_mutation(requirement: AnswerRequirement, claim: str, question: str) -> ConflictMutation | None:
    folded_question = accentfold(question)
    if requirement.key in {"time", "birth_time", "death_time"}:
        date = _full_date_mutation(requirement, claim, question)
        if date:
            return date
        for match in YEAR_RE.finditer(claim):
            old = match.group(1)
            if not grounded_in_source(old, requirement.answer_fragment) or re.search(rf"(?<!\d){re.escape(old)}(?!\d)", question):
                continue
            new = _different_number(old)
            return ConflictMutation(
                slot_key=requirement.key,
                slot_label=requirement.label,
                conflict_type="year",
                original_value=old,
                mutated_value=new,
                original_claim=claim,
                mutated_claim=_replace_match(claim, match, new, group=1),
            )
    if requirement.key == "count" or re.search(r"\b(?:bao nhieu|so luong|may)\b", folded_question):
        for match in NUMBER_RE.finditer(claim):
            old = match.group(1)
            if not grounded_in_source(old, requirement.answer_fragment) or re.search(rf"(?<!\d){re.escape(old)}(?!\d)", question):
                continue
            new = _different_number(old)
            return ConflictMutation(
                slot_key=requirement.key,
                slot_label=requirement.label,
                conflict_type="count",
                original_value=old,
                mutated_value=new,
                original_claim=claim,
                mutated_claim=_replace_match(claim, match, new, group=1),
            )
    return None


def _values(patterns: Iterable[re.Pattern[str]], texts: Iterable[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = match.group(1).strip(" ,.;:")
                words = value.split()
                folded = normalize_grounding(value)
                if not value or not any(word[:1].isupper() for word in words):
                    continue
                if folded in {"day", "do", "nay", "kia", "bao", "tong", "bo tong"}:
                    continue
                found.append(value)
    return list(dict.fromkeys(found))


def _person_names(text: str) -> list[str]:
    values: list[str] = []
    for match in PERSON_NAME_RE.finditer(text):
        previous = re.search(r"([0-9A-Za-zÀ-ỹĐđ'’\-]+)\s+$", text[: match.start()])
        if previous and previous.group(1)[:1].isupper():
            continue
        tokens = match.group(0).strip(" ,.;:").split()
        name_tokens = [tokens[0]]
        for token in tokens[1:]:
            if not token[:1].isupper():
                break
            name_tokens.append(token)
        if len(name_tokens) >= 2:
            value = " ".join(name_tokens)
            if normalize_grounding(name_tokens[-1]) in {"nha", "quan", "bo", "tong"}:
                continue
            values.append(value)
    return list(dict.fromkeys(values))


def _person_context_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for value in _person_names(text):
        if re.search(
            rf"(?i:\b(?:tại|ở|từ|đến|thuộc|hiệu|tự))\s+{re.escape(value)}\b",
            text,
        ):
            continue
        person_context = bool(
            re.search(
                rf"(?i:\b(?:ông|bà|tướng|đại tướng|thiếu tướng|vua|hoàng đế|"
                rf"chủ tịch|thủ lĩnh))\s+(?:\S+\s+){{0,3}}{re.escape(value)}\b",
                text,
            )
            or re.search(
                rf"\b{re.escape(value)}\b.{{0,35}}(?i:\b(?:lãnh đạo|chỉ huy|chỉ đạo|"
                rf"làm trưởng|được cử|được phong|thành lập|làm Tiết độ sứ))",
                text,
            )
        )
        if person_context:
            candidates.append(value)
    return candidates


def _person_question_requests_an_unknown_value(question: str) -> bool:
    folded = normalize_grounding(question)
    if "la ai" in folded:
        names = _person_names(question)
        if any(folded.startswith(normalize_grounding(name)) for name in names):
            return False
    return bool(
        re.search(
            r"\b(?:ai|do ai|nhung ai|gom.+ai|nhan vat nao|nguoi nao)\b",
            folded,
        )
    )


def _relation_mutation(
    requirement: AnswerRequirement,
    claim: str,
    question: str,
    evidence_texts: list[str],
) -> ConflictMutation | None:
    if requirement.key in {"person", "person_leader"}:
        if not _person_question_requests_an_unknown_value(question):
            return None
        location_values = {
            normalize_grounding(value) for value in _values(PLACE_PATTERNS, evidence_texts)
        }
        original_names = [
            value for value in _person_names(claim)
            if grounded_in_source(value, requirement.answer_fragment)
            and not grounded_in_source(value, question)
        ]
        alternative_names = [
            value
            for text in evidence_texts
            for value in _person_context_candidates(text)
        ]
        for original in original_names:
            choices = [
                value for value in alternative_names
                if normalize_grounding(value) != normalize_grounding(original)
                and not grounded_in_source(value, question)
                and not grounded_in_source(value, requirement.answer_fragment)
                and normalize_grounding(value) not in location_values
            ]
            if choices:
                replacement = sorted(
                    set(choices), key=lambda value: (len(value.split()), len(value), normalize_grounding(value))
                )[0]
                match = re.search(re.escape(original), claim)
                if match:
                    return ConflictMutation(
                        slot_key=requirement.key,
                        slot_label=requirement.label,
                        conflict_type="person",
                        original_value=original,
                        mutated_value=replacement,
                        original_claim=claim,
                        mutated_claim=_replace_match(claim, match, replacement),
                    )
        patterns = PERSON_PATTERNS
        conflict_type = "person"
    elif requirement.key == "place":
        patterns = PLACE_PATTERNS
        conflict_type = "location"
    elif requirement.key == "opponent":
        patterns = OPPONENT_PATTERNS
        conflict_type = "person"
    else:
        return None
    originals = _values(patterns, [claim])
    alternatives = _values(patterns, evidence_texts)
    for original in originals:
        if not grounded_in_source(original, requirement.answer_fragment):
            continue
        choices = [
            value for value in alternatives
            if normalize_grounding(value) != normalize_grounding(original)
            and normalize_grounding(value) not in normalize_grounding(original)
            and normalize_grounding(original) not in normalize_grounding(value)
            and not grounded_in_source(value, question)
        ]
        if not choices:
            continue
        replacement = sorted(choices, key=lambda value: (len(value), normalize_grounding(value)))[0]
        match = re.search(re.escape(original), claim)
        if not match:
            continue
        return ConflictMutation(
            slot_key=requirement.key,
            slot_label=requirement.label,
            conflict_type=conflict_type,
            original_value=original,
            mutated_value=replacement,
            original_claim=claim,
            mutated_claim=_replace_match(claim, match, replacement),
        )
    return None


def _title_mutation(requirement: AnswerRequirement, claim: str) -> ConflictMutation | None:
    pattern = STYLE_PATTERNS.get(requirement.key)
    other_pattern = STYLE_PATTERNS.get(
        "courtesy_name" if requirement.key == "style_name" else "style_name"
    )
    if pattern is None or other_pattern is None:
        return None
    original_match = pattern.search(claim)
    replacement_match = other_pattern.search(claim)
    if not original_match or not replacement_match:
        return None
    original = original_match.group(1).strip(" ,.;:")
    replacement = replacement_match.group(1).strip(" ,.;:")
    if not grounded_in_source(original, requirement.answer_fragment) or normalize_grounding(original) == normalize_grounding(replacement):
        return None
    return ConflictMutation(
        slot_key=requirement.key,
        slot_label=requirement.label,
        conflict_type="role/title",
        original_value=original,
        mutated_value=replacement,
        original_claim=claim,
        mutated_claim=_replace_match(claim, original_match, replacement, group=1),
    )


def propose_question_relevant_conflict(
    *,
    question: str,
    gold_answer: str,
    selected: Iterable[SelectedEvidence],
    evidence_texts: Iterable[str],
) -> tuple[str, ConflictMutation] | None:
    """Return one clean same-slot mutation, or None when correctness is uncertain."""
    requirements = [
        item for item in extract_answer_requirements(question, gold_answer)
        if item.confidence == "high" and item.key != "answer"
    ]
    pool = [str(text) for text in evidence_texts]
    for requirement in requirements:
        for item in selected:
            for claim in item.claims:
                if not requirement_supported(requirement, question=question, evidence_text=claim):
                    continue
                mutation = (
                    _numeric_mutation(requirement, claim, question)
                    or _title_mutation(requirement, claim)
                    or _relation_mutation(requirement, claim, question, pool)
                )
                if mutation and conflict_values_incompatible(
                    mutation.original_value, mutation.mutated_value, mutation.conflict_type
                ):
                    return item.evidence_id, mutation
    return None


def propose_irrelevant_disagreement(
    *,
    question: str,
    gold_answer: str,
    selected: Iterable[SelectedEvidence],
) -> tuple[str, ConflictMutation] | None:
    """Create a hard negative by changing a numeric fact outside every answer slot."""
    relevant_fragments = "\n".join(
        item.answer_fragment for item in extract_answer_requirements(question, gold_answer)
    )
    for item in selected:
        for claim in item.claims:
            for match in NUMBER_RE.finditer(claim):
                old = match.group(1)
                if re.search(rf"(?<!\d){re.escape(old)}(?!\d)", relevant_fragments):
                    continue
                new = _different_number(old)
                return item.evidence_id, ConflictMutation(
                    slot_key="unrelated",
                    slot_label="chi tiết không thuộc answer slot được hỏi",
                    conflict_type="irrelevant",
                    original_value=old,
                    mutated_value=new,
                    original_claim=claim,
                    mutated_claim=_replace_match(claim, match, new, group=1),
                )
    return None


def conflict_metadata_is_question_relevant(
    *,
    question: str,
    gold_answer: str,
    metadata: dict[str, Any],
) -> bool:
    slot_key = str(metadata.get("mutated_slot") or "")
    original_value = str(metadata.get("original_value") or "")
    requirements = extract_answer_requirements(question, gold_answer)
    return any(
        item.key == slot_key
        and item.confidence == "high"
        and grounded_in_source(original_value, item.answer_fragment)
        for item in requirements
    )


def canonical_conflict_value(value: str, conflict_type: str) -> str:
    word_date = WORD_DATE_RE.search(value)
    slash_date = SLASH_DATE_RE.search(value)
    if conflict_type == "date" and (word_date or slash_date):
        match = word_date or slash_date
        return f"{int(match.group(3)):04d}-{int(match.group(2)):02d}-{int(match.group(1)):02d}"
    if conflict_type in {"year", "count"}:
        number = NUMBER_RE.search(value)
        if number:
            return str(int(number.group(1)))
    return normalize_grounding(value)


def conflict_values_incompatible(original: str, mutated: str, conflict_type: str) -> bool:
    left = canonical_conflict_value(original, conflict_type)
    right = canonical_conflict_value(mutated, conflict_type)
    if not left or not right or left == right:
        return False
    if conflict_type in {"person", "location", "role/title"} and (left in right or right in left):
        return False
    return True
