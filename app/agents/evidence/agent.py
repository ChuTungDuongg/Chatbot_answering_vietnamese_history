from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agents.common.model_runtime import RoleLLMBackend
from app.agents.common.model_registry import ROLE_MODELS
from app.agents.evidence.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.evidence.schemas import EvidenceAgentRequest, EvidenceCritique, EvidenceModelOutput, SelectedEvidence
from app.agents.common.comparison import (
    SHARED,
    TARGET_A,
    TARGET_B,
    UNKNOWN,
    classify_comparison_target,
    comparison_dimension_coverage,
    comparison_target_relevance,
)
from app.agents.evidence.validation import (
    compressed_derived_from_own_claims,
    grounded_in_source,
    normalize_grounding,
    referenced_evidence_ids,
)
from app.rag.retrieval import extract_comparison_targets, text_matches_target
from app.telemetry import current_request_telemetry, log_event


WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
YEAR_RE = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")
QUESTION_YEAR_RE = re.compile(r"\bnam\s+(\d{3,4})\b")
QUESTION_STOPWORDS = {
    "ai", "bao", "bi", "cai", "cho", "co", "cua", "da", "duoc", "gi",
    "khi", "la", "mot", "nam", "nao", "nhu", "nhung", "o", "ra", "sau",
    "tai", "the", "thi", "theo", "trong", "tu", "va", "ve", "voi",
}
RECOVERY_STOPWORDS = QUESTION_STOPWORDS | {
    "ay", "bang", "cau", "chi", "do", "nay", "nen", "thuoc",
}
EVIDENCE_TEXT_BUDGET = 7_200
MAX_EVIDENCE_ITEM_CHARS = 1_600
MIN_EVIDENCE_ITEM_CHARS = 650
MAX_RECOVERY_SPAN_CHARS = 700
SEMANTIC_RELEVANCE_MARGIN = 0.24
SEMANTIC_RELEVANCE_FLOOR = 0.34
CLAIM_RELEVANCE_FLOOR = 0.34
CAUSE_CUES = {
    "nguyen nhan", "vi sao", "tai sao", "dan den", "suy yeu",
}
SIGNIFICANCE_CUES = {
    "y nghia", "vai tro", "he qua", "tac dong",
}
ANALYTICAL_CUES = CAUSE_CUES | SIGNIFICANCE_CUES | {
    "so sanh", "phan tich", "danh gia",
}
ANALYTICAL_CLAIM_CUES = {
    "nguyen nhan", "vi ", "do ", "boi ", "dan den", "lam ", "khien",
    "that bai", "suy yeu", "phu thuoc", "chien luoc", "quan su",
    "chinh tri", "kinh te", "xa hoi",
}
SIGNIFICANCE_CLAIM_CUES = {
    "y nghia", "cham dut", "mo ra", "doc lap", "tu chu", "bac thuoc",
    "danh dau", "xung vuong", "chu quyen", "that bai", "ke hoach", "dai tiep",
}
COMPARISON_DETAIL_CUES = {
    "boi canh", "dien bien", "gianh", "chinh quyen", "thanh lap", "doc lap",
    "chien dich", "chien thang", "quan su", "ke hoach", "hiep dinh", "phap", "my",
}
FACTUAL_PREFIXES = {
    "ai",
    "khi nao",
    "o dau",
    "nhan vat nao",
    "vua nao",
    "tuong nao",
    "trieu dai nao",
    "su kien nao",
    "nam nao",
}
FACTUAL_CUES = {
    "duoc menh danh",
    "la ai",
    "ten gi",
    "ten la gi",
}
FACTOR_KEYWORDS = {
    "political": {"chinh tri", "quyen luc", "trieu dinh", "vua", "quan lai", "the che", "lanh dao", "ho quy ly"},
    "social": {"xa hoi", "nong dan", "khoi nghia", "bat binh", "bien dong", "dan chung", "noi day"},
    "military": {"chien tranh", "quan su", "ngoai xam", "cham", "chiem", "nguyen", "xung dot", "bien gioi"},
    "economic": {"kinh te", "thue", "ruong dat", "mat mua", "doi kem", "tai chinh", "san xuat"},
    "institutional": {"the che", "to chuc", "quan lieu", "suy thoai", "tham nhung", "cai cach", "ky cuong"},
    "religious": {"phat giao", "ton giao", "su tang", "chua", "tu vien"},
    "sovereignty": {"doc lap", "tu chu", "bac thuoc", "nam han", "chu quyen", "xung vuong"},
    "commemoration": {"mieu", "den", "le", "te", "thai lao", "co", "tuong niem"},
}
NAVIGATION_NOISE_CUES = {
    "xem them",
    "chu thich",
    "tham khao",
    "lien ket ngoai",
    "the loai",
    "chu de",
    "danh muc",
    "bai viet lien quan",
    "cong thong tin",
    "tro choi dien tu",
    "video game",
    "danh sach",
    "phim truyen hinh",
    "truyen thong dai chung",
    "thu muc",
    "isbn",
}
SUMMARY_CUES = {"tom tat", "tong quan", "khai quat", "so luoc"}
SUPERLATIVE_CUES = {
    "gioi nhat",
    "xuat sac nhat",
    "quan trong nhat",
    "tot nhat",
    "vi dai nhat",
    "duoc danh gia cao nhat",
}
BROAD_SUMMARY_FACETS = ["timeframe_context", "actors", "course", "result", "significance"]
EXPLANATORY_CLAIM_CUES = {
    " la ", " da ", " gianh ", " dien ra", " danh ", " khien ",
    " buoc ", " dan den", " thanh lap", " mo ra", " danh dau",
    " cham dut", " nham ", " tro thanh", " bat dau", " ket thuc",
    " to chuc", " lanh dao", " tan cong", " tien cong", " thang loi",
    " that bai", " gop phan", " ky ket",
}
logger = logging.getLogger(__name__)


def question_relevant_excerpt(text: str, question: str, *, max_chars: int) -> str:
    """Choose an extractive window that preserves question-relevant late passages."""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text

    terms = {
        normalize_grounding(match.group(0))
        for match in WORD_RE.finditer(str(question))
        if normalize_grounding(match.group(0)) not in QUESTION_STOPWORDS
    }
    tokens = [
        (normalize_grounding(match.group(0)), match.start(), match.end())
        for match in WORD_RE.finditer(text)
    ]
    matching = [token for token in tokens if token[0] in terms]
    if not matching:
        return text[:max_chars].rstrip()

    frequencies = Counter(token for token, _, _ in matching)
    starts = {0, max(0, len(text) - max_chars)}
    for _, start, _ in matching:
        starts.add(max(0, min(start - max_chars // 3, len(text) - max_chars)))
        starts.add(max(0, min(start - (2 * max_chars) // 3, len(text) - max_chars)))

    def window_score(start: int) -> tuple[float, int]:
        end = start + max_chars
        visible = [token for token, left, _ in matching if start <= left < end]
        unique = set(visible)
        coverage = sum(
            (2.0 if token.isdigit() else 1.0) + 1.0 / frequencies[token]
            for token in unique
        )
        density = len(visible) / max(len(matching), 1)
        return coverage + density, -start

    best_start = max(starts, key=window_score)
    best_end = min(len(text), best_start + max_chars)
    if best_start:
        next_space = text.find(" ", best_start)
        if 0 <= next_space < best_end:
            best_start = next_space + 1
    if best_end < len(text):
        previous_space = text.rfind(" ", best_start, best_end)
        if previous_space > best_start:
            best_end = previous_space
    return text[best_start:best_end].strip()


def _content_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for match in WORD_RE.finditer(str(value)):
        token = normalize_grounding(match.group(0))
        if len(token) > 1 and token not in RECOVERY_STOPWORDS:
            terms.add(token)
    return terms


def _source_sentence_spans(source_text: str) -> list[str]:
    spans = [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?\n。！？]+[.!?。！？]?", source_text)
        if match.group(0).strip()
    ]
    if not spans and source_text.strip():
        spans = [source_text.strip()]
    return spans


def _best_extractive_span(source_text: str, question: str, claim: str) -> str | None:
    claim_terms = _content_terms(claim)
    if not claim_terms:
        return None
    question_terms = _content_terms(question)
    candidates = [span for span in _source_sentence_spans(source_text) if len(span) <= MAX_RECOVERY_SPAN_CHARS]
    if not candidates:
        candidates = [question_relevant_excerpt(source_text, f"{question} {claim}", max_chars=MAX_RECOVERY_SPAN_CHARS)]

    scored: list[tuple[float, int, str]] = []
    for index, span in enumerate(candidates):
        span_terms = _content_terms(span)
        claim_hits = span_terms & claim_terms
        if not claim_hits:
            continue
        question_hits = span_terms & question_terms
        claim_ratio = len(claim_hits) / max(len(claim_terms), 1)
        question_ratio = len(question_hits) / max(len(question_terms), 1)
        has_enough_claim_overlap = len(claim_hits) >= 2 or claim_ratio >= 0.34
        has_question_support = bool(question_hits) or claim_ratio >= 0.34
        if not (has_enough_claim_overlap and has_question_support):
            continue
        score = (claim_ratio * 3.0) + question_ratio + (0.1 * len(claim_hits))
        scored.append((score, -index, span))

    if not scored:
        return None
    span = max(scored)[2].strip()
    return span if grounded_in_source(span, source_text) else None


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_grounding(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _normalized_text(value: str) -> str:
    return normalize_grounding(value)


def _question_is_analytical(question: str) -> bool:
    normalized = _normalized_text(question)
    return any(cue in normalized for cue in ANALYTICAL_CUES)


def _evidence_question_type(question: str) -> str:
    normalized = _normalized_text(question)
    if len(extract_comparison_targets(question)) >= 2:
        return "compare"
    if any(cue in normalized for cue in SUPERLATIVE_CUES):
        return "evaluative"
    if any(cue in normalized for cue in SUMMARY_CUES):
        return "broad_summary"
    if any(cue in normalized for cue in CAUSE_CUES):
        return "cause"
    if any(cue in normalized for cue in SIGNIFICANCE_CUES):
        return "significance"
    if any(cue in normalized for cue in ANALYTICAL_CUES - {"so sanh"}):
        return "analysis"
    if any(normalized == prefix or normalized.startswith(f"{prefix} ") for prefix in FACTUAL_PREFIXES):
        return "factual"
    if any(cue in normalized for cue in FACTUAL_CUES):
        return "factual"
    return "general"


def _question_years(question: str) -> set[str]:
    normalized = _normalized_text(question)
    years = set(QUESTION_YEAR_RE.findall(normalized))
    if years:
        return years
    all_years = set(YEAR_RE.findall(normalized))
    return all_years if len(all_years) == 1 else set()


def _title_year_conflicts_question(question: str, title: str | None) -> bool:
    question_years = _question_years(question)
    title_years = set(YEAR_RE.findall(_normalized_text(title or "")))
    return bool(question_years and title_years and question_years.isdisjoint(title_years))


def _text_factor_labels(value: str) -> set[str]:
    normalized = _normalized_text(value)
    return {
        label
        for label, keywords in FACTOR_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    }


def _claim_noise_reason(value: str) -> str | None:
    raw = " ".join(str(value).split()).strip()
    normalized = f" {_normalized_text(raw)[:700]} "
    if not raw:
        return "empty_claim"
    cue_hits = sum(1 for cue in NAVIGATION_NOISE_CUES if cue in normalized)
    if cue_hits >= 1:
        return "navigation_or_metadata"
    words = WORD_RE.findall(raw)
    if len(words) < 5:
        return "malformed_fragment"
    repeated_vietnam = len(re.findall(r"\bvi[eệ]t\s+nam\b", raw, flags=re.I))
    has_explanation = any(cue in normalized for cue in EXPLANATORY_CLAIM_CUES)
    capitalized = sum(1 for word in words if word[:1].isupper())
    without_terminal_punctuation = raw.rstrip(".!?。！？").strip()
    looks_like_entity_list = (
        len(words) >= 12
        and not re.search(r"[.!?。！？]", without_terminal_punctuation)
        and not has_explanation
        and (repeated_vietnam >= 2 or capitalized / max(len(words), 1) >= 0.34)
    )
    if looks_like_entity_list:
        return "entity_or_tag_enumeration"
    return None


def _navigation_noise_penalty(value: str) -> float:
    reason = _claim_noise_reason(value)
    if reason in {"navigation_or_metadata", "entity_or_tag_enumeration"}:
        return 0.85
    if reason:
        return 0.45
    return 0.0


def _retrieval_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(score, 1.0))


def _semantic_score(question: str, *, title: str | None, text: str, retrieval_score: Any = None) -> float:
    question_terms = _content_terms(question)
    haystack = f"{title or ''} {text}"
    text_terms = _content_terms(haystack)
    if not question_terms or not text_terms:
        return _retrieval_score(retrieval_score) * 0.1
    overlap = question_terms & text_terms
    overlap_score = len(overlap) / max(len(question_terms), 1)
    normalized_question = _normalized_text(question)
    normalized_haystack = _normalized_text(haystack)
    phrase_bonus = 0.0
    for phrase in ("nha tran", "suy yeu", "nguyen nhan", "dan den"):
        if phrase in normalized_question and phrase in normalized_haystack:
            phrase_bonus += 0.12
    factor_bonus = min(0.18, 0.06 * len(_text_factor_labels(haystack)))
    score = overlap_score + phrase_bonus + factor_bonus + _retrieval_score(retrieval_score) * 0.12
    score -= _navigation_noise_penalty(f"{title or ''} {text}")
    return max(0.0, min(1.0, score))


def _claim_relevance_score(
    question: str,
    claim: str,
    *,
    question_type: str | None = None,
) -> float:
    question_type = question_type or _evidence_question_type(question)
    question_terms = _content_terms(question)
    claim_terms = _content_terms(claim)
    if not question_terms or not claim_terms:
        return 0.0
    normalized_claim = _normalized_text(claim)
    overlap = len(question_terms & claim_terms) / max(len(question_terms), 1)
    score = overlap

    question_years = _question_years(question)
    claim_years = set(YEAR_RE.findall(normalized_claim))
    if question_years and question_type != "compare":
        if claim_years and question_years.isdisjoint(claim_years):
            score -= 0.55
        elif question_years & claim_years:
            score += 0.16

    if question_type == "significance":
        if any(cue in normalized_claim for cue in SIGNIFICANCE_CLAIM_CUES):
            score += 0.22
        elif overlap < 0.75:
            score -= 0.26
    elif question_type in {"analysis", "cause"}:
        if any(cue in normalized_claim for cue in ANALYTICAL_CLAIM_CUES):
            score += 0.16
        elif overlap < 0.65:
            score -= 0.12
    return max(0.0, min(1.0, score))


def _comparison_claim_relevance_score(
    question: str,
    target: str,
    claim: str,
    *,
    candidate_title: str | None = None,
) -> float:
    question_score = _claim_relevance_score(question, claim, question_type="compare")
    target_terms = _content_terms(target)
    claim_terms = _content_terms(f"{candidate_title or ''} {claim}")
    if not target_terms or not claim_terms:
        return question_score

    target_overlap = len(target_terms & claim_terms) / max(len(target_terms), 1)
    if not target_overlap and not text_matches_target(f"{candidate_title or ''} {claim}", target):
        return question_score

    normalized_claim = _normalized_text(claim)
    detail_bonus = 0.0
    if any(cue in normalized_claim for cue in COMPARISON_DETAIL_CUES):
        detail_bonus += 0.12
    if YEAR_RE.search(normalized_claim):
        detail_bonus += 0.06
    if text_matches_target(str(candidate_title or ""), target):
        detail_bonus += 0.08
    else:
        detail_bonus -= 0.18

    target_score = (target_overlap * 0.62) + detail_bonus
    target_score -= _navigation_noise_penalty(claim)
    return max(0.0, min(1.0, max(question_score, target_score)))


def _claim_novelty_score(existing_claims: list[str], claim: str) -> float:
    claim_terms = _content_terms(claim)
    if not claim_terms:
        return 0.0
    if not existing_claims:
        return 1.0
    max_overlap = 0.0
    for existing in existing_claims:
        existing_terms = _content_terms(existing)
        if not existing_terms:
            continue
        max_overlap = max(max_overlap, len(claim_terms & existing_terms) / max(len(claim_terms), 1))
    return max(0.0, 1.0 - max_overlap)


def _selected_useful_claims(
    question: str,
    item: SelectedEvidence,
    *,
    question_type: str,
    compare_target: str | None = None,
    candidate_title: str | None = None,
) -> list[str]:
    claims = [
        claim
        for claim in item.claims
        if not _claim_noise_reason(claim) and (
            _comparison_claim_relevance_score(
                question,
                compare_target,
                claim,
                candidate_title=candidate_title,
            )
            if question_type == "compare" and compare_target
            else _claim_relevance_score(question, claim, question_type=question_type)
        ) >= CLAIM_RELEVANCE_FLOOR
    ]
    return _dedupe_preserve_order(claims)


def _best_candidate_claims(
    question: str,
    candidate: Any,
    *,
    max_claims: int = 2,
    compare_target: str | None = None,
) -> list[str]:
    source_text = str(getattr(candidate, "text", "") or "")
    scored: list[tuple[float, int, str]] = []
    question_type = _evidence_question_type(question)
    candidate_title = str(getattr(candidate, "title", "") or "")
    for index, span in enumerate(_source_sentence_spans(source_text)):
        span = span.strip()
        if not span:
            continue
        if _claim_noise_reason(span):
            continue
        if len(span) > MAX_RECOVERY_SPAN_CHARS:
            span = question_relevant_excerpt(span, question, max_chars=MAX_RECOVERY_SPAN_CHARS)
        if not grounded_in_source(span, source_text):
            continue
        if question_type == "compare" and compare_target:
            score = _comparison_claim_relevance_score(
                question,
                compare_target,
                span,
                candidate_title=candidate_title,
            )
        else:
            score = _claim_relevance_score(question, span, question_type=question_type)
        if score < CLAIM_RELEVANCE_FLOOR:
            continue
        scored.append((score, -index, span))
    scored.sort(reverse=True)
    claims = _dedupe_preserve_order([span for _, _, span in scored[:max_claims]])
    if claims:
        return claims
    excerpt = question_relevant_excerpt(source_text, question, max_chars=MAX_RECOVERY_SPAN_CHARS)
    if (
        excerpt
        and grounded_in_source(excerpt, source_text)
        and not _claim_noise_reason(excerpt)
        and (
            _comparison_claim_relevance_score(
                question,
                compare_target,
                excerpt,
                candidate_title=candidate_title,
            )
            if question_type == "compare" and compare_target
            else _claim_relevance_score(question, excerpt, question_type=question_type)
        ) >= CLAIM_RELEVANCE_FLOOR
    ):
        return [excerpt]
    return []


class EvidenceModelContractError(ValueError):
    """The Evidence adapter returned output that cannot satisfy the production contract."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "evidence",
        code: str = "grounding_contract_failed",
        evidence_ids: list[str] | None = None,
        repair_attempted: bool = False,
        validation_errors: list[dict[str, Any]] | None = None,
        user_message: str = "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.",
    ):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.evidence_ids = evidence_ids or []
        self.repair_attempted = repair_attempted
        self.validation_errors = validation_errors or []
        self.user_message = user_message


@dataclass(frozen=True)
class EvidenceValidationIssue:
    code: str
    message: str
    evidence_id: str | None = None
    recoverable: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        if self.evidence_id is not None:
            payload["evidence_id"] = self.evidence_id
        return payload


@dataclass(frozen=True)
class EvidenceRebucketResult:
    output: EvidenceModelOutput
    moved_claim_count: int
    destination_evidence_ids: list[str]


def _source_kind(item: dict[str, Any]) -> str:
    value = str(item.get("source_kind") or item.get("source_type") or "history").strip().lower()
    if value in {"local", "history"}:
        return "history"
    if value in {"wikipedia", "web", "attachment"}:
        return value
    return "web" if value.startswith("http") else value


def _candidate_quality(question: str, item: dict[str, Any]) -> float:
    score = item.get("final_retrieval_score")
    if score is None:
        score = item.get("reranker_score")
    if score is None:
        score = item.get("score")
    quality = _semantic_score(
        question,
        title=str(item.get("title") or ""),
        text=str(item.get("text") or ""),
        retrieval_score=score,
    )
    if _title_year_conflicts_question(question, str(item.get("title") or "")):
        quality -= 0.6
    comparison_targets = extract_comparison_targets(question)
    if len(comparison_targets) >= 2:
        relevance = [comparison_target_relevance(target, item) for target in comparison_targets[:2]]
        best = max(relevance, key=lambda detail: detail["score"])
        quality += min(0.18, 0.06 * float(best["score"]))
        quality -= min(0.24, 0.12 * float(best["incidental_penalty"]))
        if not bool(best.get("direct")) and any(bool(detail.get("direct")) for detail in relevance):
            quality -= 0.18
    if not _candidate_affiliation_constraint_pass(question, item):
        quality -= 0.75
    return max(0.0, quality)


def _candidate_matches_target(item: Any, target: str) -> bool:
    if isinstance(item, dict):
        title = item.get("title", "")
        text = item.get("text", "")
    else:
        title = getattr(item, "title", "")
        text = getattr(item, "text", "")
    return text_matches_target(f"{title or ''} {text or ''}", target)


def _candidate_title_matches_target(item: Any, target: str) -> bool:
    title = item.get("title", "") if isinstance(item, dict) else getattr(item, "title", "")
    return text_matches_target(str(title or ""), target)


def _comparison_target_from_existing_role(candidate: Any, targets: list[str]) -> str | None:
    if len(targets) < 2:
        return None
    raw = candidate if isinstance(candidate, dict) else {
        "comparison_target": getattr(candidate, "comparison_target", None),
        "retrieval_query_roles": getattr(candidate, "retrieval_query_roles", []),
    }
    label = str(raw.get("comparison_target") or "").strip()
    if label == TARGET_A:
        return targets[0]
    if label == TARGET_B:
        return targets[1]
    roles = set(raw.get("retrieval_query_roles") or [])
    if TARGET_B in roles and TARGET_A not in roles:
        return targets[1]
    if TARGET_A in roles and TARGET_B not in roles:
        return targets[0]
    return None


def _selected_matches_target(item: SelectedEvidence, candidate: Any, target: str) -> bool:
    return text_matches_target(
        f"{getattr(candidate, 'title', '') or ''} {item.compressed_text} {' '.join(item.claims)}",
        target,
    )


def _candidate_best_comparison_target(candidate: Any, targets: list[str]) -> str | None:
    existing = _comparison_target_from_existing_role(candidate, targets)
    if existing:
        return existing
    raw = candidate if isinstance(candidate, dict) else {
        "title": getattr(candidate, "title", ""),
        "text": getattr(candidate, "text", ""),
        "metadata": getattr(candidate, "metadata", {}),
    }
    ranked = sorted(
        ((comparison_target_relevance(target, raw)["score"], target) for target in targets),
        reverse=True,
    )
    if ranked and ranked[0][0] >= 2.5:
        return ranked[0][1]
    for target in targets:
        if _candidate_matches_target(candidate, target):
            return target
    return None


def _extract_affiliation_constraint(question: str) -> dict[str, str] | None:
    normalized = _normalized_text(question)
    patterns = (
        r"\b(?:tuong|nhan vat|nguoi|lanh dao|si quan)\s+(?:phe|thuoc|cua)\s+([a-z0-9 ]{2,40})",
        r"\bphe\s+([a-z0-9][a-z0-9 ]{1,38})",
        r"\b(?:vua|quan|tuong)\s+nha\s+([a-z0-9 ]{2,40})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        group = match.group(1).strip(" ?.,;:")
        group = re.split(r"\b(?:nao|gioi|xuat|quan trong|tot|vi dai|co|la)\b", group, maxsplit=1)[0].strip()
        if group:
            return {"group": group}
    return None


def _candidate_affiliation_constraint_pass(question: str, item: dict[str, Any]) -> bool:
    constraint = _extract_affiliation_constraint(question)
    if not constraint:
        return True
    group = constraint["group"]
    title = _normalized_text(str(item.get("title") or ""))
    metadata = _normalized_text(str(item.get("metadata") or ""))
    text = _normalized_text(str(item.get("text") or ""))
    subject = f"{title} {metadata}".strip()
    negative_window = rf"\b(?:chong|danh bai|thang|tien cong|doi dau|giao tranh voi)\s+{re.escape(group)}\b"
    if re.search(negative_window, text):
        return False
    if re.search(rf"\b{re.escape(group)}\b", subject):
        return True
    positive_patterns = (
        rf"\b(?:thuoc|cua|phe|phia|trong|phuc vu|chi huy|tuong|si quan|quan luc|quan doi)\s+{re.escape(group)}\b",
        rf"\b{re.escape(group)}\s+(?:cong hoa|luc luong|quan luc|quan doi|tuong|si quan|chi huy)\b",
    )
    if any(re.search(pattern, text) for pattern in positive_patterns):
        return not bool(re.search(negative_window, text))
    return False


def _broad_summary_facets_for_text(value: str) -> set[str]:
    normalized = _normalized_text(value)
    facets: set[str] = set()
    if YEAR_RE.search(normalized) or any(cue in normalized for cue in ("giai doan", "bat dau", "ket thuc", "boi canh")):
        facets.add("timeframe_context")
    if any(cue in normalized for cue in ("luc luong", "chinh quyen", "quan doi", "tuong ", "chi huy", "my", "phap", "vnch", "viet nam dan chu cong hoa", "viet minh")):
        facets.add("actors")
    if any(cue in normalized for cue in ("dien bien", "chien dich", "tan cong", "tong tien cong", "giai doan", "leo thang")):
        facets.add("course")
    if any(cue in normalized for cue in ("ket qua", "ket thuc", "chien thang", "that bai", "thang loi", "hiep dinh", "sup do", "giai phong")):
        facets.add("result")
    if any(cue in normalized for cue in ("y nghia", "he qua", "hau qua", "tac dong", "dan den", "thong nhat", "chia cat", "buoc ngoat", "mo ra")):
        facets.add("significance")
    return facets


class EvidenceCriticAgent:
    def __init__(
        self,
        *,
        max_contexts: int = 8,
        model_runtime: RoleLLMBackend | None = None,
        allow_model_fallback: bool = False,
    ):
        self.max_contexts = max_contexts
        self.model_runtime = model_runtime
        self.allow_model_fallback = allow_model_fallback

    def compress(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        *,
        final_k: int,
        request_id: str | None = None,
    ) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        if self.model_runtime is not None:
            try:
                return self._model_compress(
                    question,
                    evidence,
                    final_k=final_k,
                    request_id=request_id,
                )
            except (ValueError, ValidationError, KeyError, TypeError) as exc:
                if not self.allow_model_fallback:
                    if isinstance(exc, EvidenceModelContractError):
                        raise
                    raise EvidenceModelContractError(
                        f"Evidence model output failed canonical schema validation: {exc}",
                        code="invalid_evidence_schema" if isinstance(exc, ValidationError) else "grounding_contract_failed",
                        validation_errors=[{
                            "code": type(exc).__name__,
                            "message": str(exc),
                        }],
                    ) from exc
                critique, contexts = self._deterministic_compress(
                    question,
                    evidence,
                    final_k=final_k,
                )
                critique.warnings.append(f"model_output_invalid_debug_fallback_used:{type(exc).__name__}")
                return critique, contexts
        return self._deterministic_compress(question, evidence, final_k=final_k)

    def _deterministic_compress(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        *,
        final_k: int,
    ) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        rejected_ids: list[str] = []
        for chunk in evidence:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            text = str(chunk.get("text", "")).strip()
            if not text:
                rejected_ids.append(chunk_id)
                continue
            selected.append(chunk)
            if len(selected) >= min(max(final_k, 1), self.max_contexts):
                break
        selected_ids = [str(chunk.get("chunk_id")) for chunk in selected]
        compressed_context = "\n\n".join(
            f"[{chunk.get('chunk_id')}] {chunk.get('title') or ''}\n{str(chunk.get('text', ''))[:900]}"
            for chunk in selected
        )
        critique = EvidenceCritique(
            status="sufficient" if selected else "insufficient",
            selected_evidence=[
                SelectedEvidence(
                    evidence_id=str(chunk.get("chunk_id")),
                    relevance=max(
                        0.0,
                        min(1.0, float(chunk.get("score") or chunk.get("reranker_score") or 0.0)),
                    ),
                    compressed_text=str(chunk.get("text", ""))[:900],
                )
                for chunk in selected
            ],
            selected_ids=selected_ids,
            rejected_ids=rejected_ids,
            compressed_context=compressed_context,
            sufficient=bool(selected),
            warnings=[] if selected else ["no_supported_evidence"],
            model_input_evidence=[
                {
                    "evidence_id": str(chunk.get("chunk_id") or ""),
                    "title": chunk.get("title"),
                    "text_preview": question_relevant_excerpt(
                        str(chunk.get("text") or ""),
                        question,
                        max_chars=220,
                    ),
                }
                for chunk in evidence
                if chunk.get("chunk_id")
            ],
        )
        return critique, selected

    def _build_evidence_request(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        *,
        final_k: int,
    ) -> tuple[EvidenceAgentRequest, dict[str, dict[str, Any]], dict[str, Any]]:
        # Canonical Evidence SFT contains at most seven candidates. Keep the
        # production pool close to that distribution while preserving source diversity.
        raw_available: dict[str, dict[str, Any]] = {}
        for item in evidence:
            chunk_id = str(item.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in raw_available:
                continue
            raw_available[chunk_id] = item

        comparison_targets = extract_comparison_targets(question)
        question_type = _evidence_question_type(question)
        if question_type == "factual":
            visible_limit = min(max(3, final_k), self.max_contexts, 5)
        elif question_type in {"compare", "cause", "analysis", "significance", "broad_summary", "evaluative"}:
            visible_limit = min(max(final_k, 6), self.max_contexts)
        else:
            visible_limit = min(max(final_k, 1), self.max_contexts, 6)
        ranked = sorted(
            raw_available.items(),
            key=lambda pair: (_candidate_quality(question, pair[1]), -list(raw_available).index(pair[0])),
            reverse=True,
        )
        selected_ids: list[str] = []
        target_candidate_counts: list[int] = []
        target_reserved_ids: dict[str, list[str]] = {TARGET_A: [], TARGET_B: []}
        if len(comparison_targets) >= 2 and visible_limit >= 2:
            reserve_per_target = min(2, max(1, visible_limit // 3))
            for target_index, target in enumerate(comparison_targets[:2]):
                target_label = TARGET_A if target_index == 0 else TARGET_B
                matches = [
                    (chunk_id, item)
                    for chunk_id, item in ranked
                    if _candidate_matches_target(item, target)
                ]
                matches.sort(
                    key=lambda pair: (
                        comparison_target_relevance(target, pair[1])["direct"],
                        comparison_target_relevance(target, pair[1])["score"],
                        _candidate_quality(question, pair[1]),
                    ),
                    reverse=True,
                )
                target_candidate_counts.append(len(matches))
                for chunk_id, item in matches:
                    if len(selected_ids) >= visible_limit or len(target_reserved_ids[target_label]) >= reserve_per_target:
                        break
                    if chunk_id in selected_ids:
                        continue
                    directness = comparison_target_relevance(target, item)
                    if directness["score"] < 1.0 or not _candidate_affiliation_constraint_pass(question, item):
                        continue
                    selected_ids.append(chunk_id)
                    target_reserved_ids[target_label].append(chunk_id)
        else:
            target_candidate_counts = [0, 0]
        while len(target_candidate_counts) < 2:
            target_candidate_counts.append(0)
        for preferred_kind in ("attachment", "wikipedia", "web", "history"):
            best = next(
                (
                    chunk_id
                    for chunk_id, item in ranked
                    if _source_kind(item) == preferred_kind
                    and chunk_id not in selected_ids
                    and _candidate_affiliation_constraint_pass(question, item)
                ),
                None,
            )
            if best is not None:
                selected_ids.append(best)
            if len(selected_ids) >= visible_limit:
                break
        for chunk_id, _ in ranked:
            if len(selected_ids) >= visible_limit:
                break
            if chunk_id not in selected_ids and _candidate_affiliation_constraint_pass(question, raw_available[chunk_id]):
                selected_ids.append(chunk_id)
        available = {chunk_id: raw_available[chunk_id] for chunk_id in selected_ids}
        dropped_ids = [chunk_id for chunk_id in raw_available if chunk_id not in available]
        raw_kind_counts = Counter(_source_kind(item) for item in raw_available.values())
        visible_kind_counts = Counter(_source_kind(item) for item in available.values())
        target_visible_counts = [
            sum(1 for item in available.values() if _candidate_matches_target(item, target))
            for target in comparison_targets[:2]
        ]
        while len(target_visible_counts) < 2:
            target_visible_counts.append(0)
        dropped_reasons = {chunk_id: "budget_not_model_visible" for chunk_id in dropped_ids}
        per_item_limit = min(
            MAX_EVIDENCE_ITEM_CHARS,
            max(MIN_EVIDENCE_ITEM_CHARS, EVIDENCE_TEXT_BUDGET // max(len(available), 1)),
        )
        evidence_payload = [
            {
                "evidence_id": chunk_id,
                "source_type": _source_kind(item),
                "title": item.get("title"),
                "url": item.get("url"),
                "chunk_id": chunk_id,
                "text": question_relevant_excerpt(
                    str(item.get("text", "")),
                    question,
                    max_chars=per_item_limit,
                ),
                "retrieval_score": item.get("final_retrieval_score") or item.get("score") or item.get("reranker_score"),
            }
            for chunk_id, item in available.items()
        ]
        request = EvidenceAgentRequest.model_validate({
            "question": question,
            "max_selected": min(max(final_k, 1), self.max_contexts),
            "evidence": evidence_payload,
        })
        request_dump = request.model_dump()
        model_input_chars = len(json.dumps(request_dump, ensure_ascii=False, sort_keys=True))
        candidate_roles: dict[str, str] = {}
        direct_subject_scores: dict[str, float] = {}
        affiliation_pass: dict[str, bool] = {}
        for chunk_id, item in raw_available.items():
            role = str(item.get("comparison_target") or "")
            if not role and len(comparison_targets) >= 2:
                target = _candidate_best_comparison_target(item, comparison_targets[:2])
                role = TARGET_A if target == comparison_targets[0] else TARGET_B if target == comparison_targets[1] else UNKNOWN
            candidate_roles[chunk_id] = role or UNKNOWN
            if len(comparison_targets) >= 2:
                direct_subject_scores[chunk_id] = max(
                    float(comparison_target_relevance(target, item).get("direct_subject_score", 0.0))
                    for target in comparison_targets[:2]
                )
            affiliation_pass[chunk_id] = _candidate_affiliation_constraint_pass(question, item)
        broad_facets_by_candidate = {
            chunk_id: sorted(_broad_summary_facets_for_text(f"{item.get('title') or ''} {item.get('text') or ''}"))
            for chunk_id, item in raw_available.items()
        }
        broad_facets_covered = sorted({
            facet
            for chunk_id in selected_ids
            for facet in broad_facets_by_candidate.get(chunk_id, [])
        })
        budget_report = {
            "raw_candidate_count": len(raw_available),
            "model_visible_candidate_count": len(available),
            "dropped_for_budget_count": len(dropped_ids),
            "dropped_ids": dropped_ids,
            "dropped_reasons": dropped_reasons,
            "dropped_source_kinds": {chunk_id: _source_kind(raw_available[chunk_id]) for chunk_id in dropped_ids},
            "source_kind_counts_raw": dict(raw_kind_counts),
            "source_kind_counts_visible": dict(visible_kind_counts),
            "question_type": question_type,
            "model_input_chars": model_input_chars,
            "model_input_tokens_estimate": max(1, model_input_chars // 4) if model_input_chars else 0,
            "candidate_roles": candidate_roles,
            "direct_subject_scores": direct_subject_scores,
            "affiliation_constraint_pass": affiliation_pass,
            "broad_summary_facets_requested": BROAD_SUMMARY_FACETS if question_type == "broad_summary" else [],
            "broad_summary_facets_covered": broad_facets_covered if question_type == "broad_summary" else [],
            "broad_summary_facets_by_candidate": broad_facets_by_candidate if question_type == "broad_summary" else {},
            "comparison_targets": comparison_targets[:2],
            "target_a_candidate_count": target_candidate_counts[0],
            "target_b_candidate_count": target_candidate_counts[1],
            "target_a_model_visible_count": target_visible_counts[0],
            "target_b_model_visible_count": target_visible_counts[1],
            "target_reserved_ids": target_reserved_ids,
            "incidental_target_penalty_ids": [
                chunk_id
                for chunk_id, item in raw_available.items()
                if comparison_targets
                and max(
                    comparison_target_relevance(target, item)["incidental_penalty"]
                    for target in comparison_targets[:2]
                ) > 0
            ],
        }
        return request, available, budget_report

    @staticmethod
    def _evidence_messages(request: EvidenceAgentRequest) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": EVIDENCE_AGENT_SYSTEM},
            {"role": "user", "content": json.dumps(request.model_dump(), ensure_ascii=False, sort_keys=True)},
        ]

    def _parse_model_output(
        self,
        output: Any,
        *,
        repair_attempted: bool = False,
    ) -> EvidenceModelOutput:
        raw_selected = output.get("selected_evidence", []) if isinstance(output, dict) else []
        if any(isinstance(item, str) for item in raw_selected):
            raise EvidenceModelContractError(
                "Evidence model returned legacy selected_evidence format list[str]. Retrain or migrate the Evidence Agent.",
                code="invalid_evidence_schema",
                repair_attempted=repair_attempted,
                validation_errors=[{
                    "code": "legacy_selected_evidence",
                    "message": "selected_evidence must contain objects, not evidence_id strings.",
                }],
            )
        try:
            return EvidenceModelOutput.model_validate(output)
        except ValidationError as exc:
            raise EvidenceModelContractError(
                f"Evidence model returned invalid canonical output: {exc}",
                code="invalid_evidence_schema",
                repair_attempted=repair_attempted,
                validation_errors=[
                    {
                        "code": "schema_validation",
                        "loc": list(error.get("loc", ())),
                        "message": str(error.get("msg", "")),
                    }
                    for error in exc.errors()
                ],
            ) from exc

    def _contract_issues(
        self,
        model_output: EvidenceModelOutput,
        visible_sources: dict[str, str],
    ) -> list[EvidenceValidationIssue]:
        issues: list[EvidenceValidationIssue] = []
        selected_ids = [item.evidence_id for item in model_output.selected_evidence]
        for evidence_id in selected_ids:
            if evidence_id not in visible_sources:
                issues.append(EvidenceValidationIssue(
                    code="invented_evidence_id",
                    message=f"Evidence model invented ID: {evidence_id}",
                    evidence_id=evidence_id,
                    recoverable=False,
                ))
        if issues:
            return issues

        for item in model_output.selected_evidence:
            source_text = visible_sources[item.evidence_id]
            for claim in item.claims:
                if grounded_in_source(claim, source_text):
                    continue
                other_sources = [
                    evidence_id
                    for evidence_id, other_text in visible_sources.items()
                    if evidence_id != item.evidence_id and grounded_in_source(claim, other_text)
                ]
                if other_sources:
                    issues.append(EvidenceValidationIssue(
                        code="cross_id_claim",
                        message=f"claim under {item.evidence_id!r} is not grounded in that same evidence source; it appears under another evidence ID",
                        evidence_id=item.evidence_id,
                        recoverable=False,
                    ))
                else:
                    issues.append(EvidenceValidationIssue(
                        code="claim_not_extractive",
                        message=f"claim under {item.evidence_id!r} is not grounded in that same evidence source",
                        evidence_id=item.evidence_id,
                        recoverable=True,
                    ))

            if not compressed_derived_from_own_claims(item, source_text):
                if grounded_in_source(item.compressed_text, source_text):
                    code = "compressed_not_claim_derived"
                    recoverable = True
                else:
                    other_sources = [
                        evidence_id
                        for evidence_id, other_text in visible_sources.items()
                        if evidence_id != item.evidence_id and grounded_in_source(item.compressed_text, other_text)
                    ]
                    if not other_sources:
                        compressed_spans = [
                            span
                            for span in _source_sentence_spans(item.compressed_text)
                            if span and not grounded_in_source(span, source_text)
                        ]
                        other_sources = [
                            evidence_id
                            for evidence_id, other_text in visible_sources.items()
                            if evidence_id != item.evidence_id
                            and any(grounded_in_source(span, other_text) for span in compressed_spans)
                        ]
                    code = "cross_id_compressed_text" if other_sources else "compressed_not_extractive"
                    recoverable = not other_sources
                issues.append(EvidenceValidationIssue(
                    code=code,
                    message=f"compressed_text under {item.evidence_id!r} is not derivable from its own grounded claims",
                    evidence_id=item.evidence_id,
                    recoverable=recoverable,
                ))

        if model_output.status == "conflicting":
            for conflict in model_output.conflicts:
                mentioned = referenced_evidence_ids(conflict, visible_sources)
                if len(mentioned) < 2:
                    issues.append(EvidenceValidationIssue(
                        code="conflict_requires_two_ids",
                        message="each conflict must reference at least two supplied evidence IDs",
                        recoverable=False,
                    ))
        return issues

    def _recover_extractive_output(
        self,
        question: str,
        model_output: EvidenceModelOutput,
        visible_sources: dict[str, str],
    ) -> EvidenceModelOutput | None:
        recovered_items: list[SelectedEvidence] = []
        for item in model_output.selected_evidence:
            source_text = visible_sources[item.evidence_id]
            claims: list[str] = []
            for claim in item.claims:
                if grounded_in_source(claim, source_text):
                    claims.append(claim)
                    continue
                span = _best_extractive_span(source_text, question, claim)
                if span is None:
                    return None
                claims.append(span)
            claims = _dedupe_preserve_order(claims)
            if not claims:
                return None
            recovered_items.append(SelectedEvidence(
                evidence_id=item.evidence_id,
                relevance=item.relevance,
                claims=claims,
                compressed_text=" ".join(claims),
            ))

        try:
            return EvidenceModelOutput(
                status=model_output.status,
                selected_evidence=recovered_items,
                conflicts=model_output.conflicts,
                missing_information=model_output.missing_information,
                summary=model_output.summary,
            )
        except ValidationError:
            return None

    def _rebucket_cross_id_claims(
        self,
        model_output: EvidenceModelOutput,
        visible_sources: dict[str, str],
    ) -> EvidenceRebucketResult | None:
        grouped: dict[str, SelectedEvidence] = {}
        moved_claim_count = 0
        destination_ids: list[str] = []

        for item in model_output.selected_evidence:
            if item.evidence_id not in visible_sources:
                return None
            source_text = visible_sources[item.evidence_id]
            for claim in item.claims:
                target_id = item.evidence_id
                if not grounded_in_source(claim, source_text):
                    matches = [
                        evidence_id
                        for evidence_id, other_text in visible_sources.items()
                        if grounded_in_source(claim, other_text)
                    ]
                    if len(matches) != 1 or matches[0] == item.evidence_id:
                        return None
                    target_id = matches[0]
                    moved_claim_count += 1
                    destination_ids.append(target_id)

                if target_id not in grouped:
                    grouped[target_id] = SelectedEvidence(
                        evidence_id=target_id,
                        relevance=item.relevance,
                        claims=[],
                        compressed_text="",
                    )
                grouped_item = grouped[target_id]
                grouped_item.relevance = max(grouped_item.relevance, item.relevance)
                grouped_item.claims = _dedupe_preserve_order([*grouped_item.claims, claim])

        if moved_claim_count == 0:
            return None

        selected: list[SelectedEvidence] = []
        for item in grouped.values():
            claims = _dedupe_preserve_order(item.claims)
            if not claims:
                continue
            selected.append(SelectedEvidence(
                evidence_id=item.evidence_id,
                relevance=item.relevance,
                claims=claims,
                compressed_text=" ".join(claims),
            ))
        if not selected:
            return None

        try:
            output = EvidenceModelOutput(
                status=model_output.status,
                selected_evidence=selected,
                conflicts=model_output.conflicts,
                missing_information=model_output.missing_information,
                summary=model_output.summary,
            )
        except ValidationError:
            return None
        return EvidenceRebucketResult(
            output=output,
            moved_claim_count=moved_claim_count,
            destination_evidence_ids=_dedupe_preserve_order(destination_ids),
        )

    def _repair_messages(
        self,
        *,
        request: EvidenceAgentRequest,
        invalid_output: Any,
        validation_errors: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        repair_payload = {
            "instructions": [
                "Return canonical JSON only.",
                "Preserve status unless the supplied evidence requires correction.",
                "Use only supplied evidence IDs.",
                "Every claim MUST be copied verbatim/extractively from its own evidence text.",
                "compressed_text must be exact source text or a concatenation of exact claims.",
                "Do not paraphrase.",
            ],
            "question": request.question,
            "evidence": [item.model_dump() for item in request.evidence],
            "invalid_output": invalid_output,
            "validator_errors": validation_errors,
        }
        return [
            {"role": "system", "content": EVIDENCE_AGENT_SYSTEM},
            {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False, sort_keys=True)},
        ]

    def _semantic_guard_findings(
        self,
        model_output: EvidenceModelOutput,
        request: EvidenceAgentRequest,
    ) -> dict[str, Any]:
        question_type = _evidence_question_type(request.question)
        base_findings: dict[str, Any] = {
            "question_type": question_type,
            "guard_policy": f"{question_type}_evidence_policy",
            "selected_ids": [item.evidence_id for item in model_output.selected_evidence],
        }
        if model_output.status != "sufficient" or not model_output.selected_evidence or not request.evidence:
            return {
                **base_findings,
                "triggered": False,
                "relevance_triggered": False,
                "coverage_triggered": False,
                "best_candidate_ids": [],
                "candidate_factors": [],
                "selected_factors": [],
                "year_conflict_ids": [],
                "comparison_targets": [],
                "missing_comparison_targets": [],
                "comparison_target_coverage": {},
                "retained_selected_ids": [],
                "evidence_pruned_claim_count": 0,
                "evidence_supplemented_count": 0,
                "evidence_supplemented_ids": [],
            }
        if question_type == "general":
            return {
                **base_findings,
                "triggered": False,
                "relevance_triggered": False,
                "coverage_triggered": False,
                "best_candidate_ids": [],
                "candidate_factors": [],
                "selected_factors": [],
                "year_conflict_ids": [],
                "comparison_targets": [],
                "missing_comparison_targets": [],
                "comparison_target_coverage": {},
                "retained_selected_ids": [item.evidence_id for item in model_output.selected_evidence],
                "evidence_pruned_claim_count": 0,
                "evidence_supplemented_count": 0,
                "evidence_supplemented_ids": [],
                "guard_policy": "accept_valid_general_first_pass",
            }

        candidates_by_id = {item.evidence_id: item for item in request.evidence}
        selected_ids = {item.evidence_id for item in model_output.selected_evidence}
        comparison_targets = extract_comparison_targets(request.question)

        candidate_claims: dict[str, list[str]] = {
            item.evidence_id: _best_candidate_claims(
                request.question,
                item,
                max_claims=2 if question_type == "factual" else 3,
                compare_target=(
                    _candidate_best_comparison_target(item, comparison_targets[:2])
                    if question_type == "compare"
                    else None
                ),
            )
            for item in request.evidence
        }
        candidate_claim_scores: dict[str, float] = {
            evidence_id: max(
                (
                    (
                        _comparison_claim_relevance_score(
                            request.question,
                            compare_target,
                            claim,
                            candidate_title=candidates_by_id[evidence_id].title,
                        )
                        if question_type == "compare" and (compare_target := _candidate_best_comparison_target(candidates_by_id[evidence_id], comparison_targets[:2]))
                        else _claim_relevance_score(
                            request.question,
                            claim,
                            question_type=question_type,
                        )
                    )
                    for claim in claims
                ),
                default=0.0,
            )
            for evidence_id, claims in candidate_claims.items()
        }
        ranked = sorted(
            (
                (
                    max(
                        candidate_claim_scores.get(item.evidence_id, 0.0),
                        0.35
                        * _semantic_score(
                            request.question,
                            title=item.title,
                            text=item.text,
                            retrieval_score=item.retrieval_score,
                        ),
                    ),
                    item.evidence_id,
                    item,
                )
                for item in request.evidence
            ),
            reverse=True,
        )

        selected_useful_claims_by_id = {
            item.evidence_id: _selected_useful_claims(
                request.question,
                item,
                question_type=question_type,
                compare_target=(
                    _candidate_best_comparison_target(candidates_by_id[item.evidence_id], comparison_targets[:2])
                    if question_type == "compare"
                    else None
                ),
                candidate_title=candidates_by_id[item.evidence_id].title,
            )
            for item in model_output.selected_evidence
            if item.evidence_id in candidates_by_id
        }
        selected_scores = [
            (
                evidence_id,
                max(
                    (
                        (
                            _comparison_claim_relevance_score(
                                request.question,
                                compare_target,
                                claim,
                                candidate_title=candidates_by_id[evidence_id].title,
                            )
                            if question_type == "compare" and (compare_target := _candidate_best_comparison_target(candidates_by_id[evidence_id], comparison_targets[:2]))
                            else _claim_relevance_score(
                                request.question,
                                claim,
                                question_type=question_type,
                            )
                        )
                        for claim in claims
                    ),
                    default=0.0,
                ),
            )
            for evidence_id, claims in selected_useful_claims_by_id.items()
        ]
        best_selected = max((score for _, score in selected_scores), default=0.0)
        best_direct_selected = best_selected
        best_unselected_score, best_unselected_id = next(
            (
                (score, evidence_id)
                for score, evidence_id, _ in ranked
                if evidence_id not in selected_ids and candidate_claims.get(evidence_id)
            ),
            (0.0, None),
        )
        relevance_triggered = (
            best_unselected_id is not None
            and best_unselected_score >= SEMANTIC_RELEVANCE_FLOOR
            and best_unselected_score > best_selected + SEMANTIC_RELEVANCE_MARGIN
        )
        weak_selected_ids = [
            item.evidence_id
            for item in model_output.selected_evidence
            if (
                item.evidence_id in candidates_by_id
                and (
                    not selected_useful_claims_by_id.get(item.evidence_id)
                    or (
                        _navigation_noise_penalty(item.compressed_text)
                        and max(best_unselected_score, best_selected) >= SEMANTIC_RELEVANCE_FLOOR
                    )
                )
            )
        ]
        if weak_selected_ids:
            relevance_triggered = True
        year_conflict_ids = [
            item.evidence_id
            for item in model_output.selected_evidence
            if item.evidence_id in candidates_by_id
            and _title_year_conflicts_question(request.question, candidates_by_id[item.evidence_id].title)
        ]
        if year_conflict_ids:
            relevance_triggered = True
        affiliation_failed_ids = [
            item.evidence_id
            for item in model_output.selected_evidence
            if item.evidence_id in candidates_by_id
            and not _candidate_affiliation_constraint_pass(
                request.question,
                candidates_by_id[item.evidence_id].model_dump(),
            )
        ]
        if affiliation_failed_ids:
            relevance_triggered = True

        retained_selected_ids = [
            item.evidence_id
            for item in model_output.selected_evidence
            if item.evidence_id in candidates_by_id
            and item.evidence_id not in weak_selected_ids
            and item.evidence_id not in year_conflict_ids
            and item.evidence_id not in affiliation_failed_ids
            and selected_useful_claims_by_id.get(item.evidence_id)
        ]
        selected_useful_claims = [
            claim
            for evidence_id in retained_selected_ids
            for claim in selected_useful_claims_by_id.get(evidence_id, [])
        ]
        selected_claim_total = sum(len(item.claims) for item in model_output.selected_evidence)
        selected_claim_retained = sum(
            len(selected_useful_claims_by_id.get(item.evidence_id, []))
            for item in model_output.selected_evidence
            if item.evidence_id in retained_selected_ids
        )
        evidence_pruned_claim_count = max(0, selected_claim_total - selected_claim_retained)

        if (
            question_type == "factual"
            and not year_conflict_ids
            and not weak_selected_ids
            and best_direct_selected >= SEMANTIC_RELEVANCE_FLOOR
        ):
            return {
                **base_findings,
                "triggered": False,
                "relevance_triggered": False,
                "coverage_triggered": False,
                "best_candidate_ids": [],
                "best_unselected_id": best_unselected_id,
                "best_unselected_score": best_unselected_score,
                "best_selected_score": best_selected,
                "best_direct_selected_score": best_direct_selected,
                "weak_selected_ids": [],
                "year_conflict_ids": [],
                "candidate_factors": [],
                "selected_factors": [],
                "comparison_targets": [],
                "missing_comparison_targets": [],
                "comparison_target_coverage": {},
                "retained_selected_ids": list(retained_selected_ids),
                "evidence_pruned_claim_count": 0,
                "evidence_supplemented_count": 0,
                "evidence_supplemented_ids": [],
                "guard_policy": "accept_valid_factual_first_pass",
            }

        analytical = question_type in {"analysis", "cause", "significance"} or _question_is_analytical(request.question)
        strong_candidates: list[tuple[float, str, set[str], list[str]]] = []
        novel_candidate_ids: list[str] = []
        for score, evidence_id, candidate in ranked:
            if _title_year_conflicts_question(request.question, candidate.title):
                continue
            if not _candidate_affiliation_constraint_pass(request.question, candidate.model_dump()):
                continue
            if score < SEMANTIC_RELEVANCE_FLOOR:
                continue
            factors = _text_factor_labels(f"{candidate.title or ''} {candidate.text}")
            claims = candidate_claims.get(evidence_id) or []
            if not claims:
                continue
            if evidence_id in retained_selected_ids:
                continue
            best_novelty = max((_claim_novelty_score(selected_useful_claims, claim) for claim in claims), default=0.0)
            if best_novelty < 0.28:
                continue
            strong_candidates.append((score, evidence_id, factors, claims))
            if evidence_id not in selected_ids:
                novel_candidate_ids.append(evidence_id)
        candidate_factors = set().union(*(factors for _, _, factors, _ in strong_candidates)) if strong_candidates else set()
        selected_factors = set().union(*(
            _text_factor_labels(f"{candidates_by_id[evidence_id].title or ''} {' '.join(claims)}")
            for evidence_id, claims in selected_useful_claims_by_id.items()
            if evidence_id in candidates_by_id
        )) if selected_ids else set()
        comparison_target_coverage: dict[str, bool] = {}
        missing_comparison_targets: list[str] = []
        comparison_candidate_ids: list[str] = []
        broad_summary_facets_requested: list[str] = []
        broad_summary_facets_covered: list[str] = []
        if len(comparison_targets) >= 2:
            for target in comparison_targets[:2]:
                target_candidates = []
                for score, evidence_id, candidate in ranked:
                    if score < SEMANTIC_RELEVANCE_FLOOR:
                        continue
                    if _title_year_conflicts_question(request.question, candidate.title):
                        continue
                    if not _candidate_affiliation_constraint_pass(request.question, candidate.model_dump()):
                        continue
                    if not _candidate_matches_target(candidate, target):
                        continue
                    if not _best_candidate_claims(
                        request.question,
                        candidate,
                        max_claims=3,
                        compare_target=target,
                    ):
                        continue
                    target_candidates.append((score, evidence_id, candidate))
                target_candidates.sort(
                    key=lambda value: (
                        _candidate_title_matches_target(value[2], target),
                        _navigation_noise_penalty(value[2].text) == 0.0,
                        value[0],
                    ),
                    reverse=True,
                )
                target_has_title_candidate = any(
                    _candidate_title_matches_target(candidate, target)
                    for _, _, candidate in target_candidates
                )
                target_selected = any(
                    item.evidence_id in candidates_by_id
                    and _selected_matches_target(item, candidates_by_id[item.evidence_id], target)
                    and (
                        not target_has_title_candidate
                        or _candidate_title_matches_target(candidates_by_id[item.evidence_id], target)
                    )
                    for item in model_output.selected_evidence
                )
                target_has_candidate = bool(target_candidates)
                comparison_target_coverage[target] = target_selected
                if target_has_candidate and not target_selected:
                    missing_comparison_targets.append(target)
                if target_candidates:
                    best_target_id = target_candidates[0][1]
                    if best_target_id not in comparison_candidate_ids:
                        comparison_candidate_ids.append(best_target_id)
            if (
                not missing_comparison_targets
                and relevance_triggered
                and best_unselected_id is not None
                and not weak_selected_ids
                and not year_conflict_ids
            ):
                best_unselected_candidate = candidates_by_id.get(best_unselected_id)
                if best_unselected_candidate is not None and not any(
                    _candidate_title_matches_target(best_unselected_candidate, target)
                    for target in comparison_targets[:2]
                ):
                    relevance_triggered = False
            if not missing_comparison_targets:
                relevance_triggered = bool(year_conflict_ids or weak_selected_ids)
        if len(comparison_targets) >= 2:
            coverage_triggered = bool(missing_comparison_targets)
        elif question_type == "broad_summary":
            broad_summary_facets_requested = BROAD_SUMMARY_FACETS
            selected_facet_set = {
                facet
                for evidence_id in retained_selected_ids
                if evidence_id in candidates_by_id
                for facet in _broad_summary_facets_for_text(
                    f"{candidates_by_id[evidence_id].title or ''} {candidates_by_id[evidence_id].text}"
                )
            }
            candidate_facet_ids: list[str] = []
            seen_facets = set(selected_facet_set)
            for score, evidence_id, candidate in ranked:
                if score < SEMANTIC_RELEVANCE_FLOOR * 0.7:
                    continue
                if evidence_id in retained_selected_ids:
                    continue
                candidate_facets = _broad_summary_facets_for_text(f"{candidate.title or ''} {candidate.text}")
                if not candidate_facets - seen_facets:
                    continue
                candidate_facet_ids.append(evidence_id)
                seen_facets.update(candidate_facets)
                if len(seen_facets) >= 3:
                    break
            broad_summary_facets_covered = sorted(selected_facet_set)
            if len(selected_facet_set) <= 1 and candidate_facet_ids:
                coverage_triggered = True
                novel_candidate_ids = _dedupe_preserve_order([*candidate_facet_ids, *novel_candidate_ids])
            else:
                coverage_triggered = False
        else:
            coverage_triggered = (
                analytical
                and bool(novel_candidate_ids)
                and (
                    len(retained_selected_ids) <= 1
                    or len(selected_useful_claims) <= 2
                    or evidence_pruned_claim_count > 0
                )
            )
        evidence_supplemented_ids = _dedupe_preserve_order(
            comparison_candidate_ids if len(comparison_targets) >= 2 else novel_candidate_ids[:2]
        )

        return {
            **base_findings,
            "triggered": relevance_triggered or coverage_triggered,
            "relevance_triggered": relevance_triggered,
            "coverage_triggered": coverage_triggered,
            "best_candidate_ids": (
                _dedupe_preserve_order(comparison_candidate_ids)
                if len(comparison_targets) >= 2
                else _dedupe_preserve_order([
                    *retained_selected_ids,
                    *evidence_supplemented_ids,
                    *[evidence_id for _, evidence_id, _, _ in strong_candidates[: self.max_contexts]],
                ])
            ),
            "best_unselected_id": best_unselected_id,
            "best_unselected_score": best_unselected_score,
            "best_selected_score": best_selected,
            "best_direct_selected_score": best_direct_selected,
            "weak_selected_ids": weak_selected_ids,
            "year_conflict_ids": year_conflict_ids,
            "affiliation_failed_ids": affiliation_failed_ids,
            "candidate_factors": sorted(candidate_factors),
            "selected_factors": sorted(selected_factors),
            "comparison_targets": comparison_targets[:2],
            "missing_comparison_targets": missing_comparison_targets,
            "comparison_target_coverage": comparison_target_coverage,
            "broad_summary_facets_requested": broad_summary_facets_requested,
            "broad_summary_facets_covered": broad_summary_facets_covered,
            "retained_selected_ids": retained_selected_ids,
            "evidence_pruned_claim_count": evidence_pruned_claim_count,
            "evidence_supplemented_count": len([
                evidence_id for evidence_id in evidence_supplemented_ids if evidence_id not in selected_ids
            ]),
            "evidence_supplemented_ids": [
                evidence_id for evidence_id in evidence_supplemented_ids if evidence_id not in selected_ids
            ],
        }

    def _semantic_reconsideration_messages(
        self,
        *,
        request: EvidenceAgentRequest,
        invalid_output: EvidenceModelOutput,
        findings: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = {
            "instructions": [
                "Return canonical JSON only.",
                "Reconsider the evidence selection for semantic relevance to the exact question.",
                "Do not select an evidence title whose explicit year conflicts with the explicit year in the question.",
                "For analytical questions, keep directly relevant grounded explanatory claims that add non-duplicate information; do not rely on broad factor labels alone.",
                "Use only supplied evidence IDs.",
                "Every claim must be copied verbatim/extractively from its own evidence text.",
                "The summary may mention only selected_evidence IDs unless an ID is explicitly marked as rejected candidate evidence.",
            ],
            "question": request.question,
            "evidence": [item.model_dump() for item in request.evidence],
            "previous_output": invalid_output.model_dump(),
            "guard_findings": findings,
        }
        return [
            {"role": "system", "content": EVIDENCE_AGENT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]

    def _semantic_guard_output(
        self,
        *,
        request: EvidenceAgentRequest,
        findings: dict[str, Any],
    ) -> EvidenceModelOutput | None:
        selected: list[SelectedEvidence] = []
        comparison_targets = extract_comparison_targets(request.question)
        if len(comparison_targets) >= 2 and findings.get("coverage_triggered"):
            candidate_ids = list(findings.get("best_candidate_ids") or [])
        elif len(comparison_targets) >= 2 and findings.get("relevance_triggered"):
            candidate_ids = [
                *list(findings.get("best_candidate_ids") or []),
                *list(findings.get("retained_selected_ids") or []),
            ]
        else:
            candidate_ids = [
                *list(findings.get("retained_selected_ids") or []),
                *list(findings.get("best_candidate_ids") or []),
            ]
        if (
            len(comparison_targets) < 2
            and findings.get("best_unselected_id")
            and findings["best_unselected_id"] not in candidate_ids
        ):
            candidate_ids = [findings["best_unselected_id"], *candidate_ids]
        candidate_ids = _dedupe_preserve_order([str(evidence_id) for evidence_id in candidate_ids])
        weak_selected_ids = set(findings.get("weak_selected_ids") or [])
        year_conflict_ids = set(findings.get("year_conflict_ids") or [])
        affiliation_failed_ids = set(findings.get("affiliation_failed_ids") or [])
        candidates_by_id = {item.evidence_id: item for item in request.evidence}
        for evidence_id in candidate_ids:
            candidate = candidates_by_id.get(str(evidence_id))
            if candidate is None:
                continue
            if (
                candidate.evidence_id in weak_selected_ids
                or candidate.evidence_id in year_conflict_ids
                or candidate.evidence_id in affiliation_failed_ids
            ):
                continue
            if _title_year_conflicts_question(request.question, candidate.title):
                continue
            if not _candidate_affiliation_constraint_pass(request.question, candidate.model_dump()):
                continue
            compare_target = (
                _candidate_best_comparison_target(candidate, comparison_targets[:2])
                if len(comparison_targets) >= 2
                else None
            )
            claims = _best_candidate_claims(
                request.question,
                candidate,
                compare_target=compare_target,
            )
            if not claims:
                continue
            question_type = str(findings.get("question_type") or _evidence_question_type(request.question))
            claim_score = max(
                (
                    (
                        _comparison_claim_relevance_score(
                            request.question,
                            compare_target,
                            claim,
                            candidate_title=candidate.title,
                        )
                        if question_type == "compare" and compare_target
                        else _claim_relevance_score(
                            request.question,
                            claim,
                            question_type=question_type,
                        )
                    )
                    for claim in claims
                ),
                default=0.0,
            )
            if claim_score < CLAIM_RELEVANCE_FLOOR:
                continue
            selected.append(SelectedEvidence(
                evidence_id=candidate.evidence_id,
                relevance=claim_score,
                claims=claims,
                compressed_text=" ".join(claims),
            ))
            if len(selected) >= request.max_selected:
                break
        if not selected:
            return None
        try:
            return EvidenceModelOutput(
                status="sufficient",
                selected_evidence=selected,
                conflicts=[],
                missing_information=[],
                summary="Các bằng chứng đã giữ lại trả lời trực tiếp câu hỏi: "
                + ", ".join(item.evidence_id for item in selected)
                + ".",
            )
        except ValidationError:
            return None

    def _source_local_parse_failure_output(
        self,
        request: EvidenceAgentRequest,
        *,
        question_type: str,
    ) -> EvidenceModelOutput | None:
        if question_type == "compare":
            return self._comparison_parse_failure_output(request)

        ranked = sorted(
            request.evidence,
            key=lambda item: _semantic_score(
                request.question,
                title=item.title,
                text=item.text,
                retrieval_score=item.retrieval_score,
            ),
            reverse=True,
        )
        max_selected = 1 if question_type == "factual" else min(request.max_selected, 4)
        max_claims = 2 if question_type == "factual" else 3
        selected: list[SelectedEvidence] = []
        selected_factors: set[str] = set()
        for candidate in ranked:
            if _title_year_conflicts_question(request.question, candidate.title):
                continue
            claims = _best_candidate_claims(request.question, candidate, max_claims=max_claims)
            if not claims:
                continue
            claim_text = " ".join(claims)
            claim_score = _semantic_score(
                request.question,
                title=None,
                text=claim_text,
                retrieval_score=None,
            )
            if claim_score < SEMANTIC_RELEVANCE_FLOOR:
                continue
            if question_type == "analysis":
                candidate_factors = _text_factor_labels(f"{candidate.title or ''} {claim_text}")
                if candidate_factors and candidate_factors <= selected_factors and len(selected) >= 2:
                    continue
                selected_factors.update(candidate_factors)
            selected.append(SelectedEvidence(
                evidence_id=candidate.evidence_id,
                relevance=claim_score,
                claims=claims,
                compressed_text=claim_text,
            ))
            if len(selected) >= max_selected:
                break

        if not selected:
            try:
                return EvidenceModelOutput(
                    status="insufficient",
                    selected_evidence=[],
                    conflicts=[],
                    missing_information=[
                        "Không tìm thấy bằng chứng nguồn-cục bộ đủ trực tiếp sau lỗi JSON của Evidence."
                    ],
                    summary="Evidence model emitted invalid JSON; deterministic source-local recovery found no direct grounded claim.",
                )
            except ValidationError:
                return None
        try:
            return EvidenceModelOutput(
                status="sufficient",
                selected_evidence=selected,
                conflicts=[],
                missing_information=[],
                summary="Evidence model emitted invalid JSON; selected source-local grounded evidence: "
                + ", ".join(item.evidence_id for item in selected)
                + ".",
            )
        except ValidationError:
            return None

    def _parse_failure_output(
        self,
        request: EvidenceAgentRequest,
    ) -> EvidenceModelOutput | None:
        question_type = _evidence_question_type(request.question)
        if question_type == "compare":
            return self._comparison_parse_failure_output(request)
        return self._source_local_parse_failure_output(request, question_type=question_type)

    def _comparison_parse_failure_output(
        self,
        request: EvidenceAgentRequest,
    ) -> EvidenceModelOutput | None:
        comparison_targets = extract_comparison_targets(request.question)
        if len(comparison_targets) < 2:
            return None

        ranked = sorted(
            request.evidence,
            key=lambda item: _semantic_score(
                request.question,
                title=item.title,
                text=item.text,
                retrieval_score=item.retrieval_score,
            ),
            reverse=True,
        )
        selected: list[SelectedEvidence] = []
        selected_ids: set[str] = set()
        for target in comparison_targets[:2]:
            target_ranked = [
                item
                for item in ranked
                if item.evidence_id not in selected_ids
                and not _title_year_conflicts_question(request.question, item.title)
                and _candidate_matches_target(item, target)
            ]
            target_ranked.sort(
                key=lambda item: (
                    _candidate_title_matches_target(item, target),
                    _semantic_score(
                        request.question,
                        title=item.title,
                        text=item.text,
                        retrieval_score=item.retrieval_score,
                    ),
                ),
                reverse=True,
            )
            candidate = target_ranked[0] if target_ranked else None
            if candidate is None:
                continue
            claims = _best_candidate_claims(
                request.question,
                candidate,
                compare_target=target,
            )
            if not claims:
                continue
            selected.append(SelectedEvidence(
                evidence_id=candidate.evidence_id,
                relevance=max(
                    (
                        _comparison_claim_relevance_score(
                            request.question,
                            target,
                            claim,
                            candidate_title=candidate.title,
                        )
                        for claim in claims
                    ),
                    default=0.0,
                ),
                claims=claims,
                compressed_text=" ".join(claims),
            ))
            selected_ids.add(candidate.evidence_id)
            if len(selected) >= request.max_selected:
                break

        if not selected:
            return None
        try:
            return EvidenceModelOutput(
                status="sufficient",
                selected_evidence=selected,
                conflicts=[],
                missing_information=[],
                summary="Evidence model emitted invalid JSON; selected source-local comparison evidence: "
                + ", ".join(item.evidence_id for item in selected)
                + ".",
            )
        except ValidationError:
            return None

    @staticmethod
    def _summary_consistent_with_selected(
        model_output: EvidenceModelOutput,
        visible_sources: dict[str, str],
    ) -> EvidenceModelOutput:
        selected_ids = {item.evidence_id for item in model_output.selected_evidence}
        referenced_ids = set(referenced_evidence_ids(model_output.summary, visible_sources))
        extra_ids = referenced_ids - selected_ids
        if not extra_ids:
            return model_output
        status_text = {
            "sufficient": "đủ",
            "insufficient": "chưa đủ",
            "conflicting": "mâu thuẫn",
        }[model_output.status]
        selected_text = ", ".join(item.evidence_id for item in model_output.selected_evidence) or "không có"
        return EvidenceModelOutput(
            status=model_output.status,
            selected_evidence=model_output.selected_evidence,
            conflicts=model_output.conflicts,
            missing_information=model_output.missing_information,
            summary=f"Evidence {status_text}; các bằng chứng được giữ lại: {selected_text}.",
        )

    def _raise_contract_error(
        self,
        issues: list[EvidenceValidationIssue],
        *,
        repair_attempted: bool,
    ) -> None:
        primary = issues[0]
        evidence_ids = [
            issue.evidence_id
            for issue in issues
            if issue.evidence_id is not None
        ]
        raise EvidenceModelContractError(
            primary.message,
            code=primary.code if primary.code.startswith(("invented", "invalid", "cross", "conflict")) else "grounding_contract_failed",
            evidence_ids=_dedupe_preserve_order(evidence_ids),
            repair_attempted=repair_attempted,
            validation_errors=[issue.as_dict() for issue in issues],
        )

    @staticmethod
    def _log_validation_failed(
        request_id: str,
        issues: list[EvidenceValidationIssue],
        repair_path: str,
    ) -> None:
        primary = issues[0]
        logger.warning(
            "evidence_validation_failed",
            extra={
                "request_id": request_id,
                "reason": primary.code,
                "evidence_id": primary.evidence_id,
                "repair_path": repair_path,
            },
        )

    def _critique_from_output(
        self,
        model_output: EvidenceModelOutput,
        *,
        available: dict[str, dict[str, Any]],
        request: EvidenceAgentRequest,
        budget_report: dict[str, Any],
        question: str,
        generation_calls: int,
        repair_used: bool,
        repair_path: str | None,
        first_model_output: dict[str, Any] | None,
        first_validation_issues: list[dict[str, Any]],
        final_validation_issues: list[dict[str, Any]],
        semantic_guard_findings: dict[str, Any],
    ) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        selected = model_output.selected_evidence[: self.max_contexts]
        contexts: list[dict[str, Any]] = []
        request_by_id = {item.evidence_id: item for item in request.evidence}
        for item in selected:
            context = dict(available[item.evidence_id])
            # Citation validation must use the original stored source, while
            # ``context["text"]`` remains the compact model-visible excerpt.
            context["validated_source_text"] = str(
                context.get("text") or request_by_id[item.evidence_id].text
            )
            context["text"] = item.compressed_text or str(context.get("text", ""))
            context["claims"] = list(item.claims)
            if len(budget_report.get("comparison_targets", [])) >= 2:
                attribution = classify_comparison_target(question, context)
                context["comparison_target"] = attribution.label
                context["comparison_target_scores"] = attribution.scores
                context["comparison_target_reasons"] = attribution.reasons
            contexts.append(context)
        selected_ids = [item.evidence_id for item in selected]
        comparison_target_map = {
            str(context["chunk_id"]): str(context.get("comparison_target") or UNKNOWN)
            for context in contexts
            if context.get("comparison_target")
        }
        target_a_selected_evidence = [
            str(context["chunk_id"])
            for context in contexts
            if context.get("comparison_target") == TARGET_A
        ]
        target_b_selected_evidence = [
            str(context["chunk_id"])
            for context in contexts
            if context.get("comparison_target") == TARGET_B
        ]
        shared_selected_evidence = [
            str(context["chunk_id"])
            for context in contexts
            if context.get("comparison_target") == SHARED
        ]
        unknown_selected_evidence = [
            str(context["chunk_id"])
            for context in contexts
            if context.get("comparison_target") == UNKNOWN
        ]
        comparison_coverage = {}
        for target in budget_report.get("comparison_targets", []):
            target_has_title_candidate = any(
                _candidate_title_matches_target(request_item, str(target))
                for request_item in request.evidence
                if _candidate_matches_target(request_item, str(target))
            )
            comparison_coverage[str(target)] = any(
                _selected_matches_target(item, request_item, str(target))
                and (
                    not target_has_title_candidate
                    or _candidate_title_matches_target(request_item, str(target))
                )
                for item in selected
                for request_item in request.evidence
                if request_item.evidence_id == item.evidence_id
            )
        dimension_coverage = (
            comparison_dimension_coverage(question, contexts)
            if len(budget_report.get("comparison_targets", [])) >= 2
            else {}
        )
        comparison_targets_covered = bool(comparison_coverage) and all(comparison_coverage.values())
        comparison_limited = bool(
            comparison_targets_covered
            and dimension_coverage
            and not dimension_coverage.get("two_sided_dimensions")
        )
        critique = EvidenceCritique(
            status=model_output.status,
            selected_evidence=selected,
            selected_ids=selected_ids,
            rejected_ids=[chunk_id for chunk_id in available if chunk_id not in selected_ids],
            compressed_context="\n\n".join(
                f"[{item.evidence_id}] {item.compressed_text}" for item in selected
            ),
            conflicts=model_output.conflicts,
            sufficient=model_output.status == "sufficient" and bool(contexts),
            warnings=[],
            missing_information=model_output.missing_information,
            summary=model_output.summary,
            model_input_evidence=[
                {
                    "evidence_id": item.evidence_id,
                    "title": item.title,
                    "text_preview": question_relevant_excerpt(
                        item.text,
                        question,
                        max_chars=220,
                    ),
                }
                for item in request.evidence
            ],
            raw_candidate_count=int(budget_report["raw_candidate_count"]),
            model_visible_candidate_count=int(budget_report["model_visible_candidate_count"]),
            evidence_model_input_chars=int(budget_report.get("model_input_chars", 0)),
            evidence_model_input_tokens=int(budget_report.get("model_input_tokens_estimate", 0)),
            dropped_for_budget_count=int(budget_report["dropped_for_budget_count"]),
            dropped_ids=list(budget_report["dropped_ids"]),
            dropped_reasons=dict(budget_report["dropped_reasons"]),
            source_kind_counts_raw=dict(budget_report["source_kind_counts_raw"]),
            source_kind_counts_visible=dict(budget_report["source_kind_counts_visible"]),
            question_type=str(budget_report.get("question_type") or _evidence_question_type(question)),
            first_model_output=first_model_output,
            first_validation_issues=first_validation_issues,
            final_validation_issues=final_validation_issues,
            semantic_guard_findings=semantic_guard_findings,
            comparison_targets=list(budget_report.get("comparison_targets", [])),
            target_a_candidate_count=int(budget_report.get("target_a_candidate_count", 0)),
            target_b_candidate_count=int(budget_report.get("target_b_candidate_count", 0)),
            target_a_model_visible_count=int(budget_report.get("target_a_model_visible_count", 0)),
            target_b_model_visible_count=int(budget_report.get("target_b_model_visible_count", 0)),
            comparison_target_coverage=comparison_coverage,
            comparison_dimension_coverage=dimension_coverage,
            comparison_evidence_sufficient=comparison_targets_covered,
            comparison_evidence_limited=comparison_limited,
            comparison_target_map=comparison_target_map,
            candidate_roles=dict(budget_report.get("candidate_roles") or {}),
            direct_subject_scores=dict(budget_report.get("direct_subject_scores") or {}),
            affiliation_constraint_pass=dict(budget_report.get("affiliation_constraint_pass") or {}),
            broad_summary_facets_requested=list(budget_report.get("broad_summary_facets_requested") or []),
            broad_summary_facets_covered=list(budget_report.get("broad_summary_facets_covered") or []),
            target_reserved_ids=dict(budget_report.get("target_reserved_ids") or {}),
            incidental_target_penalty_ids=list(budget_report.get("incidental_target_penalty_ids") or []),
            target_a_selected_evidence=target_a_selected_evidence,
            target_b_selected_evidence=target_b_selected_evidence,
            shared_selected_evidence=shared_selected_evidence,
            unknown_selected_evidence=unknown_selected_evidence,
            evidence_pruned_claim_count=int(semantic_guard_findings.get("evidence_pruned_claim_count", 0)),
            evidence_supplemented_count=int(semantic_guard_findings.get("evidence_supplemented_count", 0)),
            evidence_supplemented_ids=list(semantic_guard_findings.get("evidence_supplemented_ids") or []),
            generation_calls=generation_calls,
            repair_used=repair_used,
            repair_path=repair_path,
        )
        return critique, contexts

    def _model_compress(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        *,
        final_k: int,
        request_id: str | None = None,
    ) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        started = time.perf_counter()
        request_id = request_id or "anonymous"
        telemetry = current_request_telemetry()
        calls_before = telemetry.total_llm_calls if telemetry is not None else 0
        if telemetry is not None:
            telemetry.evidence_attempts += 1
        request, available, budget_report = self._build_evidence_request(question, evidence, final_k=final_k)
        # Keep the compact excerpts sent to the model separate from the full
        # source text used for citation/grounding validation.  A shortened
        # excerpt is an input-size optimization, not a replacement for the
        # authoritative text stored in ``available``.
        visible_sources = {
            item.evidence_id: str(available.get(item.evidence_id, {}).get("text") or item.text)
            for item in request.evidence
        }
        selected_candidate_count = min(max(final_k, 1), self.max_contexts)
        generation_calls = 1
        repair_used = False
        repair_path: str | None = None
        first_model_output: dict[str, Any] | None = None
        first_validation_issues: list[dict[str, Any]] = []
        final_validation_issues: list[dict[str, Any]] = []
        semantic_guard_findings: dict[str, Any] = {
            "question_type": str(budget_report.get("question_type") or _evidence_question_type(question)),
            "guard_policy": f"{budget_report.get('question_type') or _evidence_question_type(question)}_evidence_policy",
            "triggered": False,
            "relevance_triggered": False,
            "coverage_triggered": False,
        }
        log_event(
            "EVIDENCE_START",
            request_id=request_id,
            attempt=telemetry.evidence_attempts if telemetry is not None else None,
            candidate_count=len(evidence),
            model_visible_count=len(request.evidence),
            final_k=final_k,
            per_item_limit=len(request.evidence[0].text) if request.evidence else 0,
            text_budget=EVIDENCE_TEXT_BUDGET,
            question_type=budget_report.get("question_type"),
        )
        if telemetry is not None:
            telemetry.evidence_candidate_count = len(request.evidence)
            telemetry.evidence_candidate_count_raw = int(budget_report["raw_candidate_count"])
            telemetry.evidence_candidate_count_model_visible = int(budget_report["model_visible_candidate_count"])
            telemetry.evidence_dropped_for_budget_count = int(budget_report["dropped_for_budget_count"])
            telemetry.evidence_dropped_ids = list(budget_report["dropped_ids"])
            telemetry.evidence_source_kind_counts_raw = dict(budget_report["source_kind_counts_raw"])
            telemetry.evidence_source_kind_counts_visible = dict(budget_report["source_kind_counts_visible"])
            telemetry.comparison_targets = list(budget_report["comparison_targets"])
            telemetry.target_a_candidate_count = int(budget_report["target_a_candidate_count"])
            telemetry.target_b_candidate_count = int(budget_report["target_b_candidate_count"])
            telemetry.target_a_model_visible_count = int(budget_report["target_a_model_visible_count"])
            telemetry.target_b_model_visible_count = int(budget_report["target_b_model_visible_count"])
            telemetry.evidence_model_input_chars = int(budget_report.get("model_input_chars", 0))
            telemetry.evidence_model_input_tokens = int(budget_report.get("model_input_tokens_estimate", 0))
            telemetry.candidate_roles = dict(budget_report.get("candidate_roles") or {})
            telemetry.direct_subject_scores = dict(budget_report.get("direct_subject_scores") or {})
            telemetry.affiliation_constraint_pass = dict(budget_report.get("affiliation_constraint_pass") or {})
            telemetry.broad_summary_facets_requested = list(
                budget_report.get("broad_summary_facets_requested") or []
            )
            telemetry.broad_summary_facets_covered = list(
                budget_report.get("broad_summary_facets_covered") or []
            )
            telemetry.external_evidence_collected_count = sum(
                count for kind, count in telemetry.evidence_source_kind_counts_raw.items()
                if kind in {"wikipedia", "web"}
            )
            telemetry.external_evidence_model_visible_count = sum(
                count for kind, count in telemetry.evidence_source_kind_counts_visible.items()
                if kind in {"wikipedia", "web"}
            )
            telemetry.external_evidence_rejection_reasons = {
                chunk_id: reason
                for chunk_id, reason in dict(budget_report["dropped_reasons"]).items()
                if dict(budget_report["dropped_source_kinds"]).get(chunk_id) in {"wikipedia", "web"}
            }

        logger.info(
            "evidence_generation_start",
            extra={
                "request_id": request_id,
                "generation_call": generation_calls,
                "input_evidence_count": len(request.evidence),
                "selected_candidate_count": selected_candidate_count,
            },
        )
        log_event("EVIDENCE_GENERATION_START", request_id=request_id, generation_number=1)
        generation_started = time.perf_counter()
        try:
            output = self.model_runtime.generate_json(
                adapter="evidence",
                messages=self._evidence_messages(request),
                max_new_tokens=int(ROLE_MODELS["evidence"].generation["max_new_tokens"]),
                repair=False,
            )
            model_output = self._parse_model_output(output)
            first_model_output = model_output.model_dump()
        except (ValueError, ValidationError, EvidenceModelContractError, KeyError, TypeError) as exc:
            if isinstance(exc, EvidenceModelContractError):
                raise
            parse_failure_question_type = _evidence_question_type(question)
            deterministic_output = self._parse_failure_output(request)
            deterministic_issues = (
                self._contract_issues(deterministic_output, visible_sources)
                if deterministic_output is not None
                else [EvidenceValidationIssue(
                    code="invalid_evidence_schema",
                    message=(
                        "Evidence model emitted invalid JSON and deterministic source-local "
                        f"{parse_failure_question_type} recovery was unavailable."
                    ),
                    recoverable=False,
                )]
            )
            log_event(
                "EVIDENCE_PARSE_FAILURE",
                request_id=request_id,
                error_type=type(exc).__name__,
                question_type=parse_failure_question_type,
                deterministic_recovery_attempted=True,
                deterministic_recovery_succeeded=bool(deterministic_output and not deterministic_issues),
                remaining_issue_codes=[issue.code for issue in deterministic_issues],
            )
            if deterministic_output is None or deterministic_issues:
                raise EvidenceModelContractError(
                    f"Evidence model output failed canonical schema validation: {exc}",
                    code=(
                        exc.code
                        if isinstance(exc, EvidenceModelContractError)
                        else "invalid_evidence_schema" if isinstance(exc, ValidationError) else "grounding_contract_failed"
                    ),
                    evidence_ids=exc.evidence_ids if isinstance(exc, EvidenceModelContractError) else [],
                    repair_attempted=exc.repair_attempted if isinstance(exc, EvidenceModelContractError) else False,
                    validation_errors=[
                        *(
                            exc.validation_errors
                            if isinstance(exc, EvidenceModelContractError) and exc.validation_errors
                            else [{"code": type(exc).__name__, "message": str(exc)}]
                        ),
                        *[issue.as_dict() for issue in deterministic_issues],
                    ],
                ) from exc
            model_output = deterministic_output
            first_model_output = None
            repair_used = True
            repair_path = "deterministic_parse_failure"
            if telemetry is not None:
                telemetry.evidence_recovery_used = True
        first_pass_latency_ms = (time.perf_counter() - generation_started) * 1000
        if telemetry is not None:
            telemetry.evidence_first_pass_latency_ms += first_pass_latency_ms
        log_event(
            "EVIDENCE_GENERATION_END",
            request_id=request_id,
            generation_number=1,
            elapsed_ms=first_pass_latency_ms,
        )
        log_event(
            "EVIDENCE_PARSE_RESULT",
            request_id=request_id,
            status=model_output.status,
            selected_count=len(model_output.selected_evidence),
            selected_ids=[item.evidence_id for item in model_output.selected_evidence],
            claim_counts_by_evidence={
                item.evidence_id: len(item.claims) for item in model_output.selected_evidence
            },
        )
        issues = self._contract_issues(model_output, visible_sources)
        first_validation_issues = [issue.as_dict() for issue in issues]
        final_validation_issues = list(first_validation_issues)

        if issues:
            repair_path = "deterministic" if all(issue.recoverable for issue in issues) else "hard_failure"
            for issue in issues:
                log_event(
                    "EVIDENCE_VALIDATION_ISSUE",
                    request_id=request_id,
                    validation_pass=False,
                    code=issue.code,
                    evidence_id=issue.evidence_id,
                    recoverable=issue.recoverable,
                )
            self._log_validation_failed(request_id, issues, repair_path)
            if any(issue.code == "cross_id_claim" for issue in issues):
                telemetry = current_request_telemetry()
                if telemetry is not None:
                    telemetry.evidence_rebucket_attempted = True
                log_event(
                    "EVIDENCE_REBUCKET_START",
                    request_id=request_id,
                    issue_codes=[issue.code for issue in issues],
                )
                rebucketed = self._rebucket_cross_id_claims(model_output, visible_sources)
                rebucket_issues = self._contract_issues(rebucketed.output, visible_sources) if rebucketed else issues
                rebucket_success = bool(rebucketed and not rebucket_issues)
                if telemetry is not None:
                    telemetry.evidence_rebucket_succeeded = telemetry.evidence_rebucket_succeeded or rebucket_success
                    telemetry.evidence_rebucket_moved_claim_count += (
                        rebucketed.moved_claim_count if rebucketed else 0
                    )
                    if rebucketed:
                        telemetry.evidence_rebucket_destination_ids = _dedupe_preserve_order([
                            *telemetry.evidence_rebucket_destination_ids,
                            *rebucketed.destination_evidence_ids,
                        ])
                    telemetry.evidence_final_validation_result = "pass" if rebucket_success else "fail"
                log_event(
                    "EVIDENCE_REBUCKET_END",
                    request_id=request_id,
                    attempted=True,
                    success=rebucket_success,
                    moved_claim_count=rebucketed.moved_claim_count if rebucketed else 0,
                    destination_evidence_ids=rebucketed.destination_evidence_ids if rebucketed else [],
                    final_validation_result="pass" if rebucket_success else "fail",
                    remaining_issue_codes=[issue.code for issue in rebucket_issues],
                )
                if rebucket_success and rebucketed is not None:
                    model_output = rebucketed.output
                    repair_used = True
                    repair_path = "deterministic_rebucket"
                    if telemetry is not None:
                        telemetry.evidence_recovery_used = True
                    issues = []
                    final_validation_issues = []
                elif rebucket_issues:
                    self._log_validation_failed(request_id, rebucket_issues, "failed_after_rebucket")
                    self._raise_contract_error(rebucket_issues, repair_attempted=False)

            if not all(issue.recoverable for issue in issues):
                self._raise_contract_error(issues, repair_attempted=False)

            if any(issue.code == "claim_not_extractive" for issue in issues) and not extract_comparison_targets(question):
                self._raise_contract_error(issues, repair_attempted=False)

            if issues:
                recovery_started = time.perf_counter()
                log_event("EVIDENCE_RECOVERY_START", request_id=request_id, issue_count=len(issues))
                recovered = self._recover_extractive_output(question, model_output, visible_sources)
                recovered_issues = self._contract_issues(recovered, visible_sources) if recovered else issues
                if recovered:
                    original_by_id = {item.evidence_id: item for item in model_output.selected_evidence}
                    for item in recovered.selected_evidence:
                        before = len(original_by_id.get(item.evidence_id, item).claims)
                        log_event(
                            "EVIDENCE_RECOVERY_ITEM",
                            request_id=request_id,
                            evidence_id=item.evidence_id,
                            claims_before=before,
                            claims_recovered=len(item.claims),
                            success=bool(item.claims),
                        )
                log_event(
                    "EVIDENCE_RECOVERY_END",
                    request_id=request_id,
                    elapsed_ms=(time.perf_counter() - recovery_started) * 1000,
                    success=bool(recovered and not recovered_issues),
                    remaining_issue_codes=[issue.code for issue in recovered_issues],
                )
                logger.info(
                    "evidence_extractive_recovery",
                    extra={
                        "request_id": request_id,
                        "success": bool(recovered and not recovered_issues),
                        "recovered_claim_count": sum(len(item.claims) for item in recovered.selected_evidence) if recovered else 0,
                    },
                )

                if recovered and not recovered_issues:
                    model_output = recovered
                    repair_used = True
                    repair_path = "deterministic"
                    if telemetry is not None:
                        telemetry.evidence_recovery_used = True
                    final_validation_issues = []
                else:
                    self._raise_contract_error(recovered_issues, repair_attempted=False)

        guard_started = time.perf_counter()
        try:
            guard_findings = self._semantic_guard_findings(model_output, request)
            semantic_guard_findings = dict(guard_findings)
            if guard_findings["triggered"]:
                telemetry = current_request_telemetry()
                if telemetry is not None:
                    telemetry.evidence_relevance_guard_triggered = (
                        telemetry.evidence_relevance_guard_triggered
                        or bool(guard_findings["relevance_triggered"])
                    )
                    telemetry.evidence_coverage_guard_triggered = (
                        telemetry.evidence_coverage_guard_triggered
                        or bool(guard_findings["coverage_triggered"])
                    )
                    log_event(
                        "EVIDENCE_SEMANTIC_GUARD_TRIGGERED",
                        request_id=request_id,
                        relevance_triggered=guard_findings["relevance_triggered"],
                        coverage_triggered=guard_findings["coverage_triggered"],
                        best_candidate_ids=guard_findings["best_candidate_ids"],
                        selected_ids=[item.evidence_id for item in model_output.selected_evidence],
                        candidate_factors=guard_findings["candidate_factors"],
                        selected_factors=guard_findings["selected_factors"],
                        comparison_targets=guard_findings.get("comparison_targets", []),
                        comparison_target_coverage=guard_findings.get("comparison_target_coverage", {}),
                        question_type=guard_findings.get("question_type"),
                        guard_policy=guard_findings.get("guard_policy"),
                        evidence_pruned_claim_count=guard_findings.get("evidence_pruned_claim_count", 0),
                        evidence_supplemented_count=guard_findings.get("evidence_supplemented_count", 0),
                        evidence_supplemented_ids=guard_findings.get("evidence_supplemented_ids", []),
                    )
                deterministic_output = self._semantic_guard_output(
                    request=request,
                    findings=guard_findings,
                )
                deterministic_issues = (
                    self._contract_issues(deterministic_output, visible_sources)
                    if deterministic_output is not None
                    else [EvidenceValidationIssue(
                        code="semantic_guard_failed",
                        message="Evidence semantic guard could not build grounded replacement evidence.",
                        recoverable=False,
                    )]
                )
                deterministic_guard = (
                    self._semantic_guard_findings(deterministic_output, request)
                    if deterministic_output is not None and not deterministic_issues
                    else {"triggered": True}
                )
                if deterministic_output is not None and not deterministic_issues and not deterministic_guard["triggered"]:
                    model_output = deterministic_output
                    repair_used = True
                    repair_path = "deterministic_semantic_guard"
                    guard_findings = {
                        **deterministic_guard,
                        "evidence_pruned_claim_count": guard_findings.get("evidence_pruned_claim_count", 0),
                        "evidence_supplemented_count": guard_findings.get("evidence_supplemented_count", 0),
                        "evidence_supplemented_ids": guard_findings.get("evidence_supplemented_ids", []),
                        "guard_correction_triggered": True,
                    }
                    semantic_guard_findings = dict(guard_findings)
                    final_validation_issues = []
                    if telemetry is not None:
                        telemetry.evidence_recovery_used = True
                        telemetry.evidence_pruned_claim_count += int(
                            guard_findings.get("evidence_pruned_claim_count", 0)
                        )
                        telemetry.evidence_supplemented_count += int(
                            guard_findings.get("evidence_supplemented_count", 0)
                        )
                        telemetry.evidence_supplemented_ids = _dedupe_preserve_order([
                            *telemetry.evidence_supplemented_ids,
                            *list(guard_findings.get("evidence_supplemented_ids") or []),
                        ])
                        telemetry.comparison_target_coverage = dict(
                            deterministic_guard.get("comparison_target_coverage") or {}
                        )
                else:
                    final_validation_issues = [issue.as_dict() for issue in deterministic_issues]
                    log_event(
                        "EVIDENCE_DETERMINISTIC_GUARD_FALLBACK",
                        request_id=request_id,
                        success=False,
                        remaining_issue_codes=[issue.code for issue in deterministic_issues],
                    )
                reconsidered_output: EvidenceModelOutput | None = None
                if guard_findings["triggered"]:
                    reconsideration_started = time.perf_counter()
                    try:
                        generation_calls += 1
                        if telemetry is not None:
                            telemetry.evidence_reconsideration_used = True
                        log_event("EVIDENCE_RECONSIDERATION_START", request_id=request_id)
                        reconsidered_raw = self.model_runtime.generate_json(
                            adapter="evidence",
                            messages=self._semantic_reconsideration_messages(
                                request=request,
                                invalid_output=model_output,
                                findings=guard_findings,
                            ),
                            max_new_tokens=int(ROLE_MODELS["evidence"].generation["max_new_tokens"]),
                            repair=False,
                        )
                        reconsidered_output = self._parse_model_output(
                            reconsidered_raw,
                            repair_attempted=True,
                        )
                        reconsidered_issues = self._contract_issues(reconsidered_output, visible_sources)
                        if reconsidered_issues:
                            self._raise_contract_error(reconsidered_issues, repair_attempted=True)
                        reconsidered_guard = self._semantic_guard_findings(reconsidered_output, request)
                        if not reconsidered_guard["triggered"]:
                            model_output = reconsidered_output
                            repair_used = True
                            repair_path = "semantic_reconsideration"
                            guard_findings = reconsidered_guard
                            semantic_guard_findings = dict(reconsidered_guard)
                            final_validation_issues = []
                        else:
                            guard_findings = reconsidered_guard
                            semantic_guard_findings = dict(reconsidered_guard)
                    finally:
                        elapsed_reconsideration_ms = (time.perf_counter() - reconsideration_started) * 1000
                        if telemetry is not None:
                            telemetry.evidence_reconsideration_latency_ms += elapsed_reconsideration_ms
                        log_event(
                            "EVIDENCE_RECONSIDERATION_END",
                            request_id=request_id,
                            elapsed_ms=elapsed_reconsideration_ms,
                        )

                    if guard_findings["triggered"]:
                        deterministic_output = self._semantic_guard_output(
                            request=request,
                            findings=guard_findings,
                        )
                        deterministic_issues = (
                            self._contract_issues(deterministic_output, visible_sources)
                            if deterministic_output is not None
                            else [EvidenceValidationIssue(
                                code="semantic_guard_failed",
                                message="Evidence semantic guard could not build grounded replacement evidence.",
                                recoverable=False,
                            )]
                        )
                        if deterministic_output is None or deterministic_issues:
                            final_validation_issues = [issue.as_dict() for issue in deterministic_issues]
                            self._raise_contract_error(deterministic_issues, repair_attempted=True)
                        deterministic_guard = self._semantic_guard_findings(deterministic_output, request)
                        if deterministic_guard["triggered"]:
                            deterministic_issues = [EvidenceValidationIssue(
                                code="semantic_guard_failed",
                                message="Evidence semantic guard replacement still failed semantic coverage.",
                                recoverable=False,
                            )]
                            final_validation_issues = [issue.as_dict() for issue in deterministic_issues]
                            self._raise_contract_error(deterministic_issues, repair_attempted=True)
                        model_output = deterministic_output
                        repair_used = True
                        repair_path = "deterministic_semantic_guard"
                        correction_findings = guard_findings
                        guard_findings = {
                            **deterministic_guard,
                            "evidence_pruned_claim_count": correction_findings.get("evidence_pruned_claim_count", 0),
                            "evidence_supplemented_count": correction_findings.get("evidence_supplemented_count", 0),
                            "evidence_supplemented_ids": correction_findings.get("evidence_supplemented_ids", []),
                            "guard_correction_triggered": True,
                        }
                        semantic_guard_findings = dict(guard_findings)
                        final_validation_issues = []
                        if telemetry is not None:
                            telemetry.evidence_recovery_used = True
                            telemetry.evidence_pruned_claim_count += int(
                                guard_findings.get("evidence_pruned_claim_count", 0)
                            )
                            telemetry.evidence_supplemented_count += int(
                                guard_findings.get("evidence_supplemented_count", 0)
                            )
                            telemetry.evidence_supplemented_ids = _dedupe_preserve_order([
                                *telemetry.evidence_supplemented_ids,
                                *list(guard_findings.get("evidence_supplemented_ids") or []),
                            ])
        finally:
            elapsed_guard_ms = (time.perf_counter() - guard_started) * 1000
            telemetry = current_request_telemetry()
            if telemetry is not None:
                telemetry.evidence_guard_latency_ms += elapsed_guard_ms
            log_event(
                "EVIDENCE_GUARD_END",
                request_id=request_id,
                elapsed_ms=elapsed_guard_ms,
                triggered=bool(semantic_guard_findings.get("triggered")),
                question_type=semantic_guard_findings.get("question_type"),
                guard_policy=semantic_guard_findings.get("guard_policy"),
            )

        model_output = self._summary_consistent_with_selected(model_output, visible_sources)
        final_contract_issues = self._contract_issues(model_output, visible_sources)
        final_validation_issues = [issue.as_dict() for issue in final_contract_issues]
        if final_contract_issues:
            self._raise_contract_error(final_contract_issues, repair_attempted=repair_used)
        if telemetry is not None:
            telemetry.evidence_final_validation_result = "pass"
        critique, contexts = self._critique_from_output(
            model_output,
            available=available,
            request=request,
            budget_report=budget_report,
            question=question,
            generation_calls=generation_calls,
            repair_used=repair_used,
            repair_path=repair_path,
            first_model_output=first_model_output,
            first_validation_issues=first_validation_issues,
            final_validation_issues=final_validation_issues,
            semantic_guard_findings=semantic_guard_findings,
        )
        logger.info(
            "evidence_complete",
            extra={
                "request_id": request_id,
                "status": critique.status,
                "selected_count": len(contexts),
                "generation_calls": generation_calls,
                "repair_used": repair_used,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        telemetry = current_request_telemetry()
        actual_calls = (telemetry.total_llm_calls - calls_before) if telemetry is not None else generation_calls
        if telemetry is not None:
            telemetry.evidence_ms += elapsed_ms
            telemetry.evidence_selected_count = len(contexts)
            telemetry.evidence_pruned_claim_count = max(
                telemetry.evidence_pruned_claim_count,
                critique.evidence_pruned_claim_count,
            )
            telemetry.evidence_supplemented_count = max(
                telemetry.evidence_supplemented_count,
                critique.evidence_supplemented_count,
            )
            telemetry.evidence_supplemented_ids = _dedupe_preserve_order([
                *telemetry.evidence_supplemented_ids,
                *critique.evidence_supplemented_ids,
            ])
            telemetry.comparison_target_coverage = dict(critique.comparison_target_coverage)
            telemetry.external_evidence_selected_count = sum(
                1 for context in contexts if _source_kind(context) in {"wikipedia", "web"}
            )
            telemetry.external_evidence_rejected_count = max(
                0,
                telemetry.external_evidence_collected_count - telemetry.external_evidence_selected_count,
            )
            selected_ids = set(critique.selected_ids)
            for chunk_id, item in available.items():
                if _source_kind(item) in {"wikipedia", "web"} and chunk_id not in selected_ids:
                    telemetry.external_evidence_rejection_reasons.setdefault(
                        chunk_id,
                        "model_visible_not_selected",
                    )
            telemetry.evidence_repair_used = telemetry.evidence_repair_used or repair_path == "model"
            telemetry.evidence_recovery_used = telemetry.evidence_recovery_used or repair_path in {
                "deterministic",
                "deterministic_rebucket",
                "model",
                "semantic_reconsideration",
                "deterministic_semantic_guard",
            }
        log_event(
            "EVIDENCE_COMPLETE",
            request_id=request_id,
            attempt=telemetry.evidence_attempts if telemetry is not None else None,
            status=critique.status,
            actual_llm_calls=actual_calls,
            recovery_used=repair_path in {"deterministic", "deterministic_rebucket", "model"},
            repair_used=repair_path == "model",
            selected_count=len(contexts),
            elapsed_ms=elapsed_ms,
        )
        return critique, contexts
