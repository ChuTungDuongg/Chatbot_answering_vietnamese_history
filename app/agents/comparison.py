from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.rag.retrieval import extract_comparison_targets, match_norm, text_matches_target


ComparisonTarget = str

TARGET_A: ComparisonTarget = "target_a"
TARGET_B: ComparisonTarget = "target_b"
SHARED: ComparisonTarget = "shared"
UNKNOWN: ComparisonTarget = "unknown"

_LEADING_EVENT_PREFIXES = (
    "chien thang",
    "chien dich",
    "tran",
    "cach mang",
    "cuoc cach mang",
    "tong khoi nghia",
    "cuoc khoi nghia",
    "khoi nghia",
    "phong trao",
    "su kien",
)
_WEAK_TARGET_TERMS = {
    "chien",
    "thang",
    "chien thang",
    "chien dich",
    "tran",
    "cuoc",
    "cach",
    "mang",
    "khoi",
    "nghia",
    "phong",
    "trao",
    "su",
    "kien",
    "nam",
}
_NON_EVENT_SUBJECT_TITLE_CUES = {
    "tuong dai",
    "bao tang",
    "duong",
    "pho",
    "quan",
    "huyen",
    "thanh pho",
    "san bay",
    "truong",
    "ky niem",
    "ngay ky niem",
    "le hoi",
    "dia danh",
    "di tich",
}

COMPARISON_DIMENSION_PATTERNS: dict[str, tuple[str, ...]] = {
    "context": (
        "boi canh", "hoan canh", "truoc khi", "sau khi", "trong khi",
        "xam luoc", "chiem dong",
    ),
    "objective_nature": (
        "muc tieu", "nham ", "tinh chat", "nhiem vu", "muc dich",
    ),
    "participants_opponent": (
        "luc luong", "quan doi", "nhan dan", "viet minh", "thuc dan",
        "de quoc", "phat xit", "quan phap", "quan nhat", "doi phuong",
    ),
    "method": (
        "khoi nghia", "tong khoi nghia", "chien dich", "tien cong",
        "dau tranh", "vu trang", "quan su", "chinh tri", "ngoai giao",
    ),
    "result": (
        "ket qua", "gianh chinh quyen", "gianh doc lap", "chien thang",
        "thang loi", "that bai", "cham dut", "thanh lap", "dau hang",
        "ky ket", "giai phong",
    ),
    "consequence": (
        "hau qua", "he qua", "dan den", "buoc ", "lam pha san",
        "tao dieu kien", "tao tien de",
    ),
    "significance": (
        "y nghia", "danh dau", "mo ra", "khang dinh", "gop phan",
        "buoc ngoat", "vai tro", "tac dong",
    ),
    "time": ("nam ", "thang ", "ngay ", "thoi gian", "giai doan"),
}


@dataclass(frozen=True)
class TargetAttribution:
    label: ComparisonTarget
    scores: dict[ComparisonTarget, float]
    reasons: dict[ComparisonTarget, list[str]]


def comparison_target_names(question: str) -> dict[ComparisonTarget, str]:
    targets = extract_comparison_targets(question)
    if len(targets) < 2:
        return {}
    return {TARGET_A: targets[0], TARGET_B: targets[1]}


def comparison_group_skeleton(question: str) -> dict[str, Any]:
    targets = comparison_target_names(question)
    if not targets:
        return {}
    return {
        TARGET_A: {"name": targets[TARGET_A], "evidence": []},
        TARGET_B: {"name": targets[TARGET_B], "evidence": []},
        "shared_evidence": [],
        "unknown_evidence": [],
    }


def _words(value: str) -> list[str]:
    return [word for word in match_norm(value).split() if word]


def _target_aliases(target: str) -> list[str]:
    normalized = match_norm(target)
    aliases = [normalized] if normalized else []
    stripped = normalized
    changed = True
    while changed:
        changed = False
        for prefix in _LEADING_EVENT_PREFIXES:
            if stripped.startswith(prefix + " "):
                stripped = stripped[len(prefix) + 1 :].strip()
                changed = True
                if stripped:
                    aliases.append(stripped)
                break
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _distinctive_terms(target: str) -> list[str]:
    aliases = _target_aliases(target)
    aliases_by_specificity = sorted(aliases, key=lambda value: (len(_words(value)), len(value)))
    for alias in aliases_by_specificity:
        terms = [term for term in _words(alias) if term not in _WEAK_TARGET_TERMS and len(term) > 1]
        if len(terms) >= 2:
            return terms
    for alias in aliases_by_specificity:
        terms = [term for term in _words(alias) if len(term) > 1]
        if len(terms) >= 2:
            return terms
    return [term for term in _words(target) if len(term) > 1]


def _contains_phrase(haystack: str, phrase: str) -> bool:
    phrase = match_norm(phrase)
    haystack = match_norm(haystack)
    if not phrase or not haystack:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", haystack))


def _contains_terms(haystack: str, terms: list[str]) -> bool:
    haystack_norm = match_norm(haystack)
    if not haystack_norm or not terms:
        return False
    hits = sum(
        1
        for term in terms
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack_norm)
    )
    required = len(terms) if len(terms) <= 3 else max(3, int(len(terms) * 0.7))
    return hits >= required


def _target_is_event_like(target: str) -> bool:
    normalized = match_norm(target)
    return any(normalized.startswith(prefix + " ") or normalized == prefix for prefix in _LEADING_EVENT_PREFIXES)


def _non_event_subject_penalty(target: str, title: str) -> float:
    title_norm = match_norm(title)
    if not (_target_is_event_like(target) and title_norm):
        return 0.0
    return 2.15 if any(cue in title_norm for cue in _NON_EVENT_SUBJECT_TITLE_CUES) else 0.0


def _metadata_text(metadata: Any) -> str:
    if isinstance(metadata, dict):
        parts: list[str] = []
        for key, value in metadata.items():
            parts.append(str(key))
            parts.append(_metadata_text(value))
        return " ".join(parts)
    if isinstance(metadata, (list, tuple, set)):
        return " ".join(_metadata_text(value) for value in metadata)
    return str(metadata or "")


def _target_support(target: str, item: dict[str, Any]) -> tuple[float, list[str]]:
    title = str(item.get("title") or "")
    text = " ".join(
        [
            str(item.get("text") or item.get("compressed_text") or ""),
            " ".join(str(claim) for claim in item.get("claims", []) if str(claim).strip())
            if isinstance(item.get("claims"), list)
            else "",
        ]
    )
    metadata = _metadata_text(item.get("metadata"))
    aliases = _target_aliases(target)
    terms = _distinctive_terms(target)
    score = 0.0
    reasons: list[str] = []

    if any(_contains_phrase(title, alias) for alias in aliases):
        score += 4.0
        reasons.append("title_phrase")
    elif _contains_terms(title, terms):
        score += 3.0
        reasons.append("title_terms")

    if any(_contains_phrase(text, alias) for alias in aliases):
        score += 2.5
        reasons.append("text_phrase")
    elif _contains_terms(text, terms):
        score += 1.75
        reasons.append("text_terms")

    if metadata:
        if any(_contains_phrase(metadata, alias) for alias in aliases):
            score += 2.0
            reasons.append("metadata_phrase")
        elif _contains_terms(metadata, terms):
            score += 1.25
            reasons.append("metadata_terms")

    penalty = _non_event_subject_penalty(target, title)
    if penalty:
        score -= penalty
        reasons.append("non_event_subject_title_penalty")

    return score, reasons


def comparison_target_relevance(target: str, item: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic directness details, penalizing incidental body mentions."""
    score, reasons = _target_support(target, item)
    title_direct = any(reason.startswith("title_") for reason in reasons)
    metadata_direct = any(reason.startswith("metadata_") for reason in reasons)
    non_subject = "non_event_subject_title_penalty" in reasons
    body_only = bool(reasons) and not title_direct and not metadata_direct
    incidental_penalty = 1.25 if body_only else 0.0
    direct_subject_score = max(0.0, score - incidental_penalty)
    return {
        "score": direct_subject_score,
        "raw_score": score,
        "incidental_penalty": incidental_penalty,
        "direct": (title_direct or metadata_direct) and not non_subject,
        "direct_subject_score": direct_subject_score,
        "reasons": reasons,
    }


def classify_comparison_target(question: str, item: dict[str, Any]) -> TargetAttribution:
    targets = comparison_target_names(question)
    if not targets:
        return TargetAttribution(label=UNKNOWN, scores={}, reasons={})

    score_a, reasons_a = _target_support(targets[TARGET_A], item)
    score_b, reasons_b = _target_support(targets[TARGET_B], item)
    scores = {TARGET_A: score_a, TARGET_B: score_b}
    reasons = {TARGET_A: reasons_a, TARGET_B: reasons_b}
    threshold = 2.5

    if score_a >= threshold and score_b >= threshold:
        if abs(score_a - score_b) < 2.0:
            return TargetAttribution(label=SHARED, scores=scores, reasons=reasons)
        return TargetAttribution(label=TARGET_A if score_a > score_b else TARGET_B, scores=scores, reasons=reasons)
    if score_a >= threshold:
        return TargetAttribution(label=TARGET_A, scores=scores, reasons=reasons)
    if score_b >= threshold:
        return TargetAttribution(label=TARGET_B, scores=scores, reasons=reasons)
    return TargetAttribution(label=UNKNOWN, scores=scores, reasons=reasons)


def _claim_dimensions(claim: str) -> set[str]:
    normalized = match_norm(claim)
    if not normalized:
        return set()
    dimensions = {
        name
        for name, patterns in COMPARISON_DIMENSION_PATTERNS.items()
        if any(pattern in normalized for pattern in patterns)
    }
    if re.search(r"(?<!\d)\d{3,4}(?!\d)", normalized):
        dimensions.add("time")
    return dimensions


def _item_claims(item: dict[str, Any]) -> list[str]:
    claims = item.get("claims")
    if isinstance(claims, list) and any(str(claim).strip() for claim in claims):
        return [str(claim).strip() for claim in claims if str(claim).strip()]
    text = str(item.get("text") or item.get("compressed_text") or "").strip()
    return [
        span.strip()
        for span in re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        if span.strip()
    ]


def comparison_dimension_coverage(
    question: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact claim-grounded, two-sided comparison support map."""
    targets = comparison_target_names(question)
    if not targets:
        return {}

    supported: dict[str, dict[str, list[str]]] = {TARGET_A: {}, TARGET_B: {}}
    shared_dimensions: dict[str, list[str]] = {}
    for raw_item in evidence:
        item = dict(raw_item)
        label = str(item.get("comparison_target") or "").strip()
        if label not in {TARGET_A, TARGET_B, SHARED, UNKNOWN}:
            label = classify_comparison_target(question, item).label
        evidence_id = str(item.get("chunk_id") or item.get("evidence_id") or "").strip()
        if not evidence_id:
            continue
        for claim in _item_claims(item):
            dimensions = _claim_dimensions(claim)
            if label in {TARGET_A, TARGET_B}:
                for dimension in dimensions:
                    supported[label].setdefault(dimension, []).append(evidence_id)
            elif label == SHARED:
                for dimension in dimensions:
                    shared_dimensions.setdefault(dimension, []).append(evidence_id)
                    matches_a = text_matches_target(claim, targets[TARGET_A])
                    matches_b = text_matches_target(claim, targets[TARGET_B])
                    if matches_a and matches_b:
                        supported[TARGET_A].setdefault(dimension, []).append(evidence_id)
                        supported[TARGET_B].setdefault(dimension, []).append(evidence_id)

    for target_support in supported.values():
        for dimension, ids in list(target_support.items()):
            target_support[dimension] = list(dict.fromkeys(ids))
    for dimension, ids in list(shared_dimensions.items()):
        shared_dimensions[dimension] = list(dict.fromkeys(ids))

    dimensions_a = set(supported[TARGET_A])
    dimensions_b = set(supported[TARGET_B])
    two_sided = sorted(dimensions_a & dimensions_b)
    one_sided = {
        TARGET_A: sorted(dimensions_a - dimensions_b),
        TARGET_B: sorted(dimensions_b - dimensions_a),
    }
    return {
        TARGET_A: {
            "name": targets[TARGET_A],
            "supported_dimensions": supported[TARGET_A],
        },
        TARGET_B: {
            "name": targets[TARGET_B],
            "supported_dimensions": supported[TARGET_B],
        },
        "two_sided_dimensions": two_sided,
        "one_sided_dimensions": one_sided,
        "shared_dimensions": shared_dimensions,
        "limited_to_supported_dimensions": not bool(two_sided),
    }


def group_comparison_evidence(question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    groups = comparison_group_skeleton(question)
    if not groups:
        return {}
    for item in evidence:
        label = str(item.get("comparison_target") or "").strip()
        if label not in {TARGET_A, TARGET_B, SHARED, UNKNOWN}:
            label = classify_comparison_target(question, item).label
        if label == TARGET_A:
            groups[TARGET_A]["evidence"].append(item)
        elif label == TARGET_B:
            groups[TARGET_B]["evidence"].append(item)
        elif label == SHARED:
            groups["shared_evidence"].append(item)
        else:
            groups["unknown_evidence"].append(item)
    return groups
