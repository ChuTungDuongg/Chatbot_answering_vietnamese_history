from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.rag.retrieval import extract_comparison_targets, match_norm


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

    return score, reasons


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
