from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.agents.history_answerer.contract import (
    HistoryAnswerContractError,
    ParsedHistoryAnswer,
    SAFE_INSUFFICIENT_ANSWER,
    SAFE_OOD_ANSWER,
    build_history_answerer_messages,
    parse_history_answer_output,
)
from app.agents.common.model_runtime import RoleLLMBackend
from app.agents.common.comparison import (
    SHARED,
    TARGET_A,
    TARGET_B,
    UNKNOWN,
    classify_comparison_target,
    comparison_dimension_coverage,
    group_comparison_evidence,
)
from app.rag.retrieval import extract_comparison_targets, text_matches_target
from app.telemetry import current_request_telemetry, log_event

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
YEAR_RE = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
GENERIC_PROVENANCE_PREFIX_RE = re.compile(
    r"(?P<boundary>\A|(?<=[.!?。！？])\s+|\n+)\s*(?:"
    r"theo\s+(?:các\s+)?(?:tài liệu|nguồn)(?:\s+(?:được cung cấp|tham khảo|đã kiểm chứng))?"
    r"|dựa\s+trên\s+(?:các\s+)?(?:tài liệu|nguồn)(?:\s+(?:được cung cấp|tham khảo|đã kiểm chứng))?"
    r"|từ\s+(?:các\s+)?(?:tài liệu|nguồn)(?:\s+(?:được cung cấp|tham khảo|đã kiểm chứng))?"
    r"|tài\s+liệu\s+(?:nêu(?:\s+rằng)?|cho\s+thấy(?:\s+rằng)?)"
    r"|các\s+nguồn\s+(?:nêu(?:\s+rằng)?|cho\s+thấy(?:\s+rằng)?|cung\s+cấp\s+dữ\s+kiện(?:\s+rằng)?)"
    r"|các\s+bằng\s+chứng(?:\s+đã\s+(?:chọn|kiểm\s+chứng))?\s+(?:nêu(?:\s+rằng)?|cho\s+thấy(?:\s+rằng)?)"
    r"|nhóm\s+bằng\s+chứng\s+(?:nêu(?:\s+rằng)?|cho\s+thấy(?:\s+rằng)?)"
    r"|các\s+dữ\s+kiện\s+được\s+cung\s+cấp\s+(?:nêu(?:\s+rằng)?|cho\s+thấy(?:\s+rằng)?)"
    r"|từ\s+các\s+dữ\s+kiện\s+này"
    r"|câu\s+trả\s+lời\s+nên\s+được\s+hiểu(?:\s+rằng)?"
    r"|các\s+khía\s+cạnh\s+được\s+tài\s+liệu\s+hỗ\s+trợ(?:\s+(?:gồm|là))?"
    r")\s*[,;:]?\s*",
    re.I | re.M,
)
HISTORY_STOPWORDS = {
    "ai", "bao", "bi", "cai", "cho", "co", "cua", "da", "duoc", "gi",
    "khi", "la", "mot", "nam", "nao", "nhu", "nhung", "o", "ra", "sau",
    "tai", "the", "thi", "theo", "trong", "tu", "va", "ve", "voi",
}
DEEP_CAUSE_CUES = {
    "nguyen nhan", "vi sao", "tai sao", "dan den", "suy yeu",
}
DEEP_SIGNIFICANCE_CUES = {
    "y nghia", "vai tro", "he qua", "tac dong",
}
DEEP_ANALYTICAL_CUES = DEEP_CAUSE_CUES | DEEP_SIGNIFICANCE_CUES | {
    "so sanh", "phan tich", "danh gia",
}
DEEP_FACTUAL_PREFIXES = {
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
DEEP_FACTUAL_CUES = {
    "duoc menh danh",
    "la ai",
    "ten gi",
    "ten la gi",
}
INTERNAL_META_PHRASES = {
    "ket luan truc tiep",
    "cac khia canh duoc tai lieu ho tro",
    "tong hop",
    "tu cac du kien nay",
    "cau tra loi nen duoc hieu",
    "can duoc dat canh nhau theo dung nhom bang chung",
    "cac nguon cung cap du kien",
    "cac nguon cho phep so sanh",
    "cac bang chung da chon",
    "cac nguon cung cap du kien rieng",
    "theo dung nhom bang chung da kiem chung",
    "cac bang chung da kiem chung",
    "cac nguon da kiem chung",
    "nhom bang chung",
    "retrieval",
    "evidence",
    "cac du kien duoc cung cap",
}
RETRYABLE_DEEP_ISSUES = {
    "deep_answer_collapse",
    "shallow_comparison",
    "internal_meta_language",
    "comparison_target_leakage",
}


def _normalize_text(value: str) -> str:
    value = str(value).lower()
    replacements = {
        "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a", "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a", "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
        "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e", "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
        "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
        "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o", "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o", "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
        "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u", "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
        "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y", "đ": "d",
    }
    return "".join(replacements.get(char, char) for char in value)


def _content_terms(value: str) -> set[str]:
    return {
        token
        for token in WORD_RE.findall(_normalize_text(value))
        if len(token) > 1 and token not in HISTORY_STOPWORDS
    }


def _is_deep_analytical_question(question: str) -> bool:
    normalized = _normalize_text(question)
    return any(cue in normalized for cue in DEEP_ANALYTICAL_CUES)


def _history_question_type(question: str) -> str:
    normalized = _normalize_text(question)
    if len(extract_comparison_targets(question)) >= 2:
        return "compare"
    if any(cue in normalized for cue in DEEP_CAUSE_CUES):
        return "cause"
    if any(cue in normalized for cue in DEEP_SIGNIFICANCE_CUES):
        return "significance"
    if any(cue in normalized for cue in DEEP_ANALYTICAL_CUES - {"so sanh"}):
        return "analysis"
    if any(normalized == prefix or normalized.startswith(f"{prefix} ") for prefix in DEEP_FACTUAL_PREFIXES):
        return "factual"
    if any(cue in normalized for cue in DEEP_FACTUAL_CUES):
        return "factual"
    return "general"


def _candidate_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(str(text)):
        sentence = sentence.strip(" \t\r\n-•")
        if sentence:
            sentences.append(sentence)
    return sentences


def _remove_generic_source_prefix(answer: str) -> str:
    return GENERIC_PROVENANCE_PREFIX_RE.sub(lambda match: match.group("boundary"), str(answer)).strip()


def _context_source_kind(context: dict[str, Any]) -> str:
    value = str(context.get("source_kind") or context.get("source_type") or "history").strip().lower()
    if value in {"local", "history"}:
        return "history"
    if value in {"attachment", "wikipedia", "web"}:
        return value
    return "web" if value.startswith("http") else value


def _context_claims(context: dict[str, Any]) -> list[str]:
    claims = context.get("claims")
    if not isinstance(claims, list):
        claims = context.get("evidence_claims")
    return [
        str(claim).strip()
        for claim in claims
        if str(claim).strip()
    ] if isinstance(claims, list) else []


def _history_input_claim_count(contexts: list[dict[str, Any]]) -> int:
    return sum(len(_context_claims(context)) for context in contexts)


def _supported_years(contexts: list[dict[str, Any]], cited_ids: list[str]) -> set[str]:
    cited = set(cited_ids)
    text = " ".join(
        str(context.get("validated_source_text") or context.get("text") or "")
        for context in contexts
        if str(context.get("chunk_id") or "") in cited
    )
    return set(YEAR_RE.findall(_normalize_text(text)))


def _unsupported_years(answer: str, contexts: list[dict[str, Any]], cited_ids: list[str]) -> list[str]:
    answer_years = set(YEAR_RE.findall(_normalize_text(answer)))
    return sorted(answer_years - _supported_years(contexts, cited_ids))


def _question_years(question: str) -> set[str]:
    normalized = _normalize_text(question)
    years = set(re.findall(r"\bnam\s+(\d{3,4})\b", normalized))
    if years:
        return years
    all_years = set(YEAR_RE.findall(normalized))
    return all_years if len(all_years) == 1 else set()


def _sentence_count(answer: str) -> int:
    return len(_candidate_sentences(answer))


def _word_count(answer: str) -> int:
    return len(WORD_RE.findall(str(answer)))


@dataclass(frozen=True)
class _HistoryGeneration:
    parsed: ParsedHistoryAnswer
    messages: list[dict[str, str]]
    answer_text: str
    quality_issues: list[str]
    unsupported_years: list[str]
    latency_ms: float

    @property
    def answer_chars(self) -> int:
        return len(self.answer_text)

    @property
    def answer_words(self) -> int:
        return _word_count(self.answer_text)


def _important_claim_score(question: str, claim: str, *, question_type: str) -> float:
    question_terms = _content_terms(question)
    claim_terms = _content_terms(claim)
    if not question_terms or not claim_terms:
        return 0.0
    overlap = len(question_terms & claim_terms) / max(len(question_terms), 1)
    normalized_claim = _normalize_text(claim)
    score = overlap

    q_years = _question_years(question)
    claim_years = set(YEAR_RE.findall(normalized_claim))
    if q_years and question_type != "compare":
        if claim_years and q_years.isdisjoint(claim_years):
            score -= 0.5
        elif q_years & claim_years:
            score += 0.16

    if question_type == "significance":
        significance_cues = (
            "y nghia", "cham dut", "mo ra", "doc lap", "tu chu", "bac thuoc",
            "danh dau", "xung vuong", "chu quyen", "that bai", "ke hoach", "dai tiep",
        )
        if any(cue in normalized_claim for cue in significance_cues):
            score += 0.2
        elif overlap < 0.75:
            score -= 0.22
    elif question_type in {"analysis", "cause"}:
        analytical_cues = (
            "nguyen nhan", "vi ", "do ", "boi ", "dan den", "lam ", "khien",
            "that bai", "suy yeu", "phu thuoc", "chien luoc", "quan su",
            "chinh tri", "kinh te", "xa hoi",
        )
        if any(cue in normalized_claim for cue in analytical_cues):
            score += 0.14
        elif overlap < 0.65:
            score -= 0.12
    return max(0.0, min(1.0, score))


def _important_context_claims(
    question: str,
    contexts: list[dict[str, Any]],
    *,
    question_type: str,
) -> list[tuple[str, str]]:
    scored: list[tuple[float, int, str, str]] = []
    for context_index, context in enumerate(contexts):
        evidence_id = str(context.get("chunk_id") or "").strip()
        if not evidence_id:
            continue
        claims = _context_claims(context) or _candidate_sentences(str(context.get("text") or ""))
        for claim_index, claim in enumerate(claims):
            score = _important_claim_score(question, claim, question_type=question_type)
            if score >= 0.34:
                scored.append((score, -context_index * 100 - claim_index, evidence_id, claim))
    scored.sort(reverse=True)
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, _, evidence_id, claim in scored:
        key = _normalize_text(claim)
        if key in seen:
            continue
        seen.add(key)
        result.append((evidence_id, claim))
    return result


def _covered_important_claim_count(
    question: str,
    answer: str,
    important_claims: list[tuple[str, str]],
) -> int:
    question_terms = _content_terms(question)
    answer_terms = _content_terms(answer)
    covered = 0
    for _, claim in important_claims:
        claim_terms = _content_terms(claim)
        distinctive_terms = claim_terms - question_terms
        if not distinctive_terms:
            distinctive_terms = claim_terms
        if not distinctive_terms:
            continue
        hit_count = len(distinctive_terms & answer_terms)
        hit_ratio = hit_count / max(len(distinctive_terms), 1)
        if hit_ratio >= 0.34 or hit_count >= 3:
            covered += 1
    return covered


def _internal_meta_language_issues(answer: str) -> list[str]:
    normalized_answer = _normalize_text(answer)
    for phrase in INTERNAL_META_PHRASES:
        if phrase in normalized_answer:
            return ["internal_meta_language"]
    return []


def _answer_mentions_supported_compare_targets(
    question: str,
    contexts: list[dict[str, Any]],
    answer: str,
) -> list[str]:
    targets = extract_comparison_targets(question)
    if len(targets) < 2:
        return []
    missing: list[str] = []
    evidence_text = " ".join(str(context.get("title") or "") + " " + str(context.get("text") or "") for context in contexts)
    for target in targets[:2]:
        if text_matches_target(evidence_text, target) and not text_matches_target(answer, target):
            missing.append(target)
    return missing


def _comparison_target_leakage_issues(
    question: str,
    contexts: list[dict[str, Any]],
    answer: str,
) -> list[str]:
    groups = _comparison_evidence_groups(question, contexts)
    if not groups:
        return []

    target_claims = {
        TARGET_A: [claim for _, claim in _claim_lines(groups[TARGET_A]["evidence"], max_claims_per_evidence=4)],
        TARGET_B: [claim for _, claim in _claim_lines(groups[TARGET_B]["evidence"], max_claims_per_evidence=4)],
    }
    target_names = {
        TARGET_A: _normalize_text(groups[TARGET_A]["name"]),
        TARGET_B: _normalize_text(groups[TARGET_B]["name"]),
    }
    current_section: str | None = None
    for line in str(answer).splitlines():
        normalized_line = _normalize_text(line)
        mentions_a = target_names[TARGET_A] and target_names[TARGET_A] in normalized_line
        mentions_b = target_names[TARGET_B] and target_names[TARGET_B] in normalized_line
        if mentions_a and not mentions_b:
            current_section = TARGET_A
        elif mentions_b and not mentions_a:
            current_section = TARGET_B
        if current_section == TARGET_A:
            wrong_claims = target_claims[TARGET_B]
        elif current_section == TARGET_B:
            wrong_claims = target_claims[TARGET_A]
        else:
            continue
        if any(claim and claim in line for claim in wrong_claims):
            return ["comparison_target_leakage"]
    return []


def _comparison_structure_issues(
    question: str,
    contexts: list[dict[str, Any]],
    answer: str,
) -> list[str]:
    targets = extract_comparison_targets(question)
    if len(targets) < 2:
        return []
    normalized_answer = _normalize_text(answer)
    explicit_section_marker = any(
        marker in normalized_answer
        for marker in ("diem giong", "diem khac", "giong nhau", "khac nhau", "so voi")
    )
    comparative_clause = False
    for sentence in _candidate_sentences(answer):
        normalized_sentence = _normalize_text(sentence)
        if "ca hai" in normalized_sentence and any(
            cue in normalized_sentence for cue in ("deu", "nhung", "trong khi", "khac", "tuong dong")
        ):
            comparative_clause = True
            break
        mentions_both = all(text_matches_target(sentence, target) for target in targets[:2])
        if mentions_both and any(
            cue in normalized_sentence
            for cue in ("nhung", "trong khi", "con ", "khac", "giong", "tuong dong", "trai lai")
        ):
            comparative_clause = True
            break

    if not (explicit_section_marker or comparative_clause):
        return ["shallow_comparison"]

    dimensions = comparison_dimension_coverage(question, contexts)
    two_sided_dimensions = set(dimensions.get("two_sided_dimensions") or [])
    if not two_sided_dimensions:
        return []
    dimension_cues = {
        "context": ("boi canh", "hoan canh"),
        "objective_nature": ("muc tieu", "tinh chat", "nhiem vu"),
        "participants_opponent": ("luc luong", "doi phuong", "thuc dan", "de quoc", "nhan dan"),
        "method": ("khoi nghia", "chien dich", "dau tranh", "quan su", "chinh tri"),
        "result": ("ket qua", "gianh", "chien thang", "thang loi", "that bai", "cham dut"),
        "consequence": ("hau qua", "he qua", "dan den", "buoc ", "tao tien de"),
        "significance": ("y nghia", "danh dau", "mo ra", "gop phan", "buoc ngoat", "vai tro"),
        "time": ("nam ", "thoi gian", "giai doan"),
    }
    discusses_supported_dimension = any(
        any(cue in f" {normalized_answer} " for cue in dimension_cues.get(dimension, ()))
        for dimension in two_sided_dimensions
    )
    return [] if discusses_supported_dimension else ["shallow_comparison"]


def _deep_answer_quality_issues(
    question: str,
    contexts: list[dict[str, Any]],
    answer: str,
    *,
    answer_depth: str,
) -> list[str]:
    if not answer.strip():
        return ["empty_answer"]

    issues: list[str] = []
    issues.extend(_internal_meta_language_issues(answer))
    if answer_depth != "deep":
        return list(dict.fromkeys(issues))

    question_type = _history_question_type(question)
    validated_claim_count = _history_input_claim_count(contexts)
    breadth = max(validated_claim_count, len(contexts))
    sentence_count = _sentence_count(answer)
    answer_is_short = len(answer) < 520
    if question_type == "compare":
        missing_targets = _answer_mentions_supported_compare_targets(question, contexts, answer)
        issues.extend(f"missing_compare_target:{target}" for target in missing_targets)
        issues.extend(_comparison_target_leakage_issues(question, contexts, answer))
        issues.extend(_comparison_structure_issues(question, contexts, answer))
    elif question_type in {"analysis", "cause", "significance"}:
        important_claims = _important_context_claims(
            question,
            contexts,
            question_type=question_type,
        )
        covered_claim_count = _covered_important_claim_count(question, answer, important_claims)
        if (
            (breadth >= 2 and sentence_count <= 1 and len(answer) < 320)
            or (breadth >= 4 and answer_is_short and sentence_count <= 2)
            or (
                len(important_claims) >= 3
                and covered_claim_count <= 1
                and (sentence_count <= 2 or len(answer) < 420)
            )
        ):
            issues.append("deep_answer_collapse")
    return list(dict.fromkeys(issues))


def _comparison_evidence_groups(question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    annotated: list[dict[str, Any]] = []
    for context in contexts:
        item = dict(context)
        label = str(item.get("comparison_target") or "").strip()
        if label not in {TARGET_A, TARGET_B, SHARED, UNKNOWN}:
            label = classify_comparison_target(question, item).label
            item["comparison_target"] = label
        annotated.append(item)
    return group_comparison_evidence(question, annotated)


def _claims_for_context(context: dict[str, Any]) -> list[str]:
    claims = _context_claims(context)
    if claims:
        return list(dict.fromkeys(claims))
    return _candidate_sentences(str(context.get("text") or ""))[:2]


def _claim_lines(items: list[dict[str, Any]], *, max_claims_per_evidence: int = 2) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    seen_claims: set[str] = set()
    for context in items:
        evidence_id = str(context.get("chunk_id") or "").strip()
        if not evidence_id:
            continue
        for claim in _claims_for_context(context)[:max_claims_per_evidence]:
            claim = claim.strip()
            key = _normalize_text(claim)
            if not claim or key in seen_claims:
                continue
            seen_claims.add(key)
            lines.append((evidence_id, claim))
    return lines


class HistoryAnswererAgent:
    """Active Qwen3 History role, matching the canonical History SFT contract."""

    def __init__(self, *, model_runtime: RoleLLMBackend):
        if model_runtime is None:
            raise ValueError("Active HistoryAnswererAgent requires a role model runtime.")
        self.model_runtime = model_runtime

    @staticmethod
    def _retrieval_payload(
        *,
        question: str,
        contexts: list[dict[str, Any]],
        analysis: dict[str, Any],
        tool_trace: list[str],
        is_ood: bool,
        ood_reason: str,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "final_context": contexts,
            "analysis": analysis,
            "tool_trace": tool_trace,
            "is_ood": is_ood,
            "ood_reason": ood_reason,
            "global_context_count": sum(
                item.get("source_kind") != "attachment" for item in contexts
            ),
            "temporary_context_count": sum(
                item.get("source_kind") == "attachment" for item in contexts
            ),
            "temporary_context_relevant": any(
                item.get("source_kind") == "attachment" for item in contexts
            ),
        }

    @staticmethod
    def _guard_result(
        *,
        question: str,
        retrieval: dict[str, Any],
        analysis: dict[str, Any],
        tool_trace: list[str],
        answer: str,
        status: str,
        guard_name: str,
        answer_depth: str,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": answer,
            "status": status,
            "source_ids": [],
            "source_chunks": [],
            "model_source_ids": [],
            "invalid_source_ids": [],
            "unsupported_years": [],
            "format_ok": True,
            "raw_output": "",
            "retrieval": retrieval,
            "analysis": analysis,
            "tool_trace": [*tool_trace, f"history_guard:{guard_name}"],
            "history_message_count": 0,
            "rewrite_used": False,
            "repair_attempted": False,
            "structured_expansion_used": False,
            "answer_provenance": {
                "source": "deterministic_guard",
                "history_adapter_called": False,
                "history_generation_calls": 0,
                "history_retry_used": False,
                "history_retry_reason": None,
                "history_first_answer_chars": 0,
                "history_first_answer_words": 0,
                "history_final_answer_chars": 0,
                "history_final_answer_words": 0,
                "history_first_quality_issues": [],
                "history_final_quality_issues": [],
                "history_first_latency_ms": 0.0,
                "history_retry_latency_ms": 0.0,
                "history_total_latency_ms": 0.0,
                "guard_short_circuit": True,
                "guard_name": guard_name,
                "guard_override": False,
                "answer_depth": answer_depth,
            },
            "history_debug": {
                "generation_calls": 0,
                "input_evidence_ids": [],
                "input_claim_count": 0,
                "input_source_kind_counts": {},
                "input_evidence_preview": [],
                "cited_ids": [],
                "conversation_history_used": False,
                "answer_depth": answer_depth,
                "question_type": _history_question_type(question),
                "history_retry_used": False,
                "history_retry_reason": None,
                "first_answer_chars": 0,
                "first_answer_words": 0,
                "final_answer_chars": 0,
                "final_answer_words": 0,
                "first_quality_issues": [],
                "final_quality_issues": [],
                "first_latency_ms": 0.0,
                "retry_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "initial_quality_issues": [],
                "quality_warnings": [],
                "unsupported_years": [],
            },
        }

    def _generate_once(
        self,
        *,
        question: str,
        contexts: list[dict[str, Any]],
        input_ids: list[str],
        answer_depth: str,
        avoid_generic_source_prefix: bool,
        retry_reason: str | None = None,
        previous_quality_issues: list[str] | None = None,
    ) -> _HistoryGeneration:
        messages = build_history_answerer_messages(
            question,
            contexts,
            answer_depth=answer_depth,
            avoid_generic_source_prefix=avoid_generic_source_prefix,
            retry_reason=retry_reason,
            previous_quality_issues=previous_quality_issues,
        )
        generation_started = time.perf_counter()
        raw_output = self.model_runtime.generate_text(
            adapter="history",
            messages=messages,
        )
        latency_ms = (time.perf_counter() - generation_started) * 1000
        parsed = parse_history_answer_output(raw_output, allowed_source_ids=input_ids)
        answer_text = (
            _remove_generic_source_prefix(parsed.answer)
            if avoid_generic_source_prefix
            else parsed.answer
        )
        quality_issues = _deep_answer_quality_issues(
            question,
            contexts,
            answer_text,
            answer_depth=answer_depth,
        )
        unsupported_years = _unsupported_years(answer_text, contexts, parsed.source_ids)
        if unsupported_years:
            quality_issues = [
                *quality_issues,
                *[f"unsupported_year:{year}" for year in unsupported_years],
            ]
        return _HistoryGeneration(
            parsed=parsed,
            messages=messages,
            answer_text=answer_text,
            quality_issues=list(dict.fromkeys(quality_issues)),
            unsupported_years=unsupported_years,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _should_retry_history(
        *,
        inference_mode: str | None,
        answer_depth: str,
        question_type: str,
        contexts: list[dict[str, Any]],
        first_quality_issues: list[str],
        question: str,
    ) -> bool:
        if inference_mode not in {"three_llm", "agentic_rag"}:
            return False
        if answer_depth != "deep" or question_type not in {"analysis", "cause", "significance", "compare"}:
            return False
        if not any(
            issue in RETRYABLE_DEEP_ISSUES
            or issue.startswith("missing_compare_target:")
            for issue in first_quality_issues
        ):
            return False
        if question_type == "compare":
            groups = _comparison_evidence_groups(question, contexts)
            return bool(
                groups
                and (
                    groups[TARGET_A]["evidence"]
                    and groups[TARGET_B]["evidence"]
                )
                and len(contexts) >= 2
            )
        important_claims = _important_context_claims(
            question,
            contexts,
            question_type=question_type,
        )
        return len(important_claims) >= 3 or (_history_input_claim_count(contexts) >= 3 and len(contexts) >= 2)

    @staticmethod
    def _retry_reason(first_quality_issues: list[str]) -> str | None:
        for issue in first_quality_issues:
            if issue == "comparison_target_leakage":
                return issue
        for issue in first_quality_issues:
            if issue == "shallow_comparison":
                return issue
        for issue in first_quality_issues:
            if issue == "deep_answer_collapse":
                return issue
        for issue in first_quality_issues:
            if issue in RETRYABLE_DEEP_ISSUES or issue.startswith("missing_compare_target:"):
                return issue
        return None

    @staticmethod
    def _retry_generation_is_valid(generation: _HistoryGeneration) -> bool:
        if generation.unsupported_years:
            return False
        if "internal_meta_language" in generation.quality_issues:
            return False
        if "comparison_target_leakage" in generation.quality_issues:
            return False
        return True

    @staticmethod
    def _retry_is_better(first: _HistoryGeneration, retry: _HistoryGeneration) -> bool:
        first_penalty = sum(
            2 if issue in {"deep_answer_collapse", "shallow_comparison"} else 1
            for issue in first.quality_issues
        )
        retry_penalty = sum(
            2 if issue in {"deep_answer_collapse", "shallow_comparison"} else 1
            for issue in retry.quality_issues
        )
        if retry_penalty < first_penalty:
            return True
        if retry_penalty > first_penalty:
            return False
        return (
            retry.answer_words > first.answer_words
            or len(retry.parsed.source_ids) > len(first.parsed.source_ids)
        )

    def answer(
        self,
        *,
        question: str,
        contexts: list[dict[str, Any]],
        analysis: dict[str, Any],
        tool_trace: list[str],
        is_ood: bool = False,
        ood_reason: str = "",
        history: list[dict[str, str]] | None = None,
        request_id: str | None = None,
        answer_depth: str = "standard",
        avoid_generic_source_prefix: bool = False,
        inference_mode: str | None = None,
    ) -> dict[str, Any]:
        del history  # History SFT has no conversation-history input.
        started = time.perf_counter()
        question = str(question).strip()
        contexts = [
            dict(item)
            for item in contexts
            if str(item.get("chunk_id") or "").strip() and str(item.get("text") or "").strip()
        ]
        retrieval = self._retrieval_payload(
            question=question,
            contexts=contexts,
            analysis=analysis,
            tool_trace=tool_trace,
            is_ood=is_ood,
            ood_reason=ood_reason,
        )
        input_ids = [str(item["chunk_id"]) for item in contexts]
        input_claim_count = _history_input_claim_count(contexts)
        input_source_kind_counts = dict(Counter(_context_source_kind(item) for item in contexts))
        question_type = _history_question_type(question)
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.history_input_evidence_count = len(contexts)
            telemetry.history_input_claim_count = input_claim_count
            telemetry.history_input_source_kind_counts = input_source_kind_counts
        log_event(
            "HISTORY_START",
            request_id=request_id,
            input_evidence_count=len(contexts),
            input_evidence_ids=input_ids,
            input_claim_count=input_claim_count,
            input_source_kind_counts=input_source_kind_counts,
            answer_depth=answer_depth,
            question_type=question_type,
        )

        if is_ood and not contexts:
            elapsed_ms = (time.perf_counter() - started) * 1000
            telemetry = current_request_telemetry()
            if telemetry is not None:
                telemetry.history_ms += elapsed_ms
            log_event(
                "HISTORY_COMPLETE",
                request_id=request_id,
                actual_llm_calls=0,
                elapsed_ms=elapsed_ms,
                cited_count=0,
                invalid_citation_count=0,
            )
            return self._guard_result(
                question=question,
                retrieval=retrieval,
                analysis=analysis,
                tool_trace=tool_trace,
                answer=SAFE_OOD_ANSWER,
                status="blocked_off_topic",
                guard_name="off_topic_no_evidence",
                answer_depth=answer_depth,
            )
        if not contexts:
            elapsed_ms = (time.perf_counter() - started) * 1000
            telemetry = current_request_telemetry()
            if telemetry is not None:
                telemetry.history_ms += elapsed_ms
            log_event(
                "HISTORY_COMPLETE",
                request_id=request_id,
                actual_llm_calls=0,
                elapsed_ms=elapsed_ms,
                cited_count=0,
                invalid_citation_count=0,
            )
            return self._guard_result(
                question=question,
                retrieval=retrieval,
                analysis=analysis,
                tool_trace=tool_trace,
                answer=SAFE_INSUFFICIENT_ANSWER,
                status="blocked_no_context",
                guard_name="no_selected_evidence",
                answer_depth=answer_depth,
            )

        by_id = {str(item["chunk_id"]): item for item in contexts}
        first_generation = self._generate_once(
            question=question,
            contexts=contexts,
            input_ids=input_ids,
            answer_depth=answer_depth,
            avoid_generic_source_prefix=avoid_generic_source_prefix,
        )
        final_generation = first_generation
        retry_generation: _HistoryGeneration | None = None
        retry_error: str | None = None
        retry_reason = self._retry_reason(first_generation.quality_issues)
        history_retry_used = False
        retry_selected = False
        retry_eligible = bool(
            retry_reason
            and self._should_retry_history(
                inference_mode=inference_mode,
                answer_depth=answer_depth,
                question_type=question_type,
                contexts=contexts,
                first_quality_issues=first_generation.quality_issues,
                question=question,
            )
        )

        if retry_eligible:
            history_retry_used = True
            log_event(
                "HISTORY_RETRY_START",
                request_id=request_id,
                reason=retry_reason,
                first_quality_issues=first_generation.quality_issues,
                first_answer_words=first_generation.answer_words,
            )
            try:
                retry_generation = self._generate_once(
                    question=question,
                    contexts=contexts,
                    input_ids=input_ids,
                    answer_depth=answer_depth,
                    avoid_generic_source_prefix=avoid_generic_source_prefix,
                    retry_reason=retry_reason,
                    previous_quality_issues=first_generation.quality_issues,
                )
                if self._retry_generation_is_valid(retry_generation) and self._retry_is_better(
                    first_generation,
                    retry_generation,
                ):
                    final_generation = retry_generation
                    retry_selected = True
                elif not self._retry_generation_is_valid(retry_generation):
                    retry_error = "retry_output_failed_validation"
            except HistoryAnswerContractError as exc:
                retry_error = f"{type(exc).__name__}:{exc}"
            log_event(
                "HISTORY_RETRY_END",
                request_id=request_id,
                reason=retry_reason,
                selected=retry_selected,
                retry_error=retry_error,
                retry_quality_issues=retry_generation.quality_issues if retry_generation else [],
                retry_answer_words=retry_generation.answer_words if retry_generation else 0,
            )

        answer_text = final_generation.answer_text
        source_ids = final_generation.parsed.source_ids
        final_quality_issues = final_generation.quality_issues
        unsupported_years = final_generation.unsupported_years
        source_chunks = [by_id[source_id] for source_id in source_ids]
        status = "ok" if source_ids else "insufficient"

        elapsed_ms = (time.perf_counter() - started) * 1000
        generation_calls = 1 + int(history_retry_used)
        invalid_citation_count = 0
        if telemetry is not None:
            telemetry.history_ms += elapsed_ms
            telemetry.history_generation_calls += generation_calls
            telemetry.history_retry_used = telemetry.history_retry_used or history_retry_used
            telemetry.history_retry_reason = retry_reason if history_retry_used else telemetry.history_retry_reason
            telemetry.history_first_answer_chars = first_generation.answer_chars
            telemetry.history_first_answer_words = first_generation.answer_words
            telemetry.history_final_answer_chars = final_generation.answer_chars
            telemetry.history_final_answer_words = final_generation.answer_words
            telemetry.history_first_quality_issues = list(first_generation.quality_issues)
            telemetry.history_final_quality_issues = list(final_generation.quality_issues)
            telemetry.history_first_latency_ms += first_generation.latency_ms
            telemetry.history_retry_latency_ms += retry_generation.latency_ms if retry_generation else 0.0
            telemetry.history_total_latency_ms += elapsed_ms
        log_event(
            "HISTORY_COMPLETE",
            request_id=request_id,
            actual_llm_calls=generation_calls,
            elapsed_ms=elapsed_ms,
            cited_count=len(source_ids),
            invalid_citation_count=invalid_citation_count,
            structured_expansion_used=False,
            history_retry_used=history_retry_used,
            history_retry_reason=retry_reason if history_retry_used else None,
            quality_issues=final_quality_issues,
        )
        return {
            "question": question,
            "answer": answer_text,
            "status": status,
            "source_ids": source_ids,
            "source_chunks": source_chunks,
            "model_source_ids": final_generation.parsed.source_ids,
            "invalid_source_ids": [],
            "unsupported_years": unsupported_years,
            "format_ok": True,
            "raw_output": final_generation.parsed.raw_output,
            "retrieval": retrieval,
            "analysis": analysis,
            "prompt_budget": None,
            "support_score": None,
            "quality_warnings": final_quality_issues,
            "rewrite_used": False,
            "repair_attempted": False,
            "structured_expansion_used": False,
            "initial_quality_issues": first_generation.quality_issues,
            "history_message_count": 0,
            "tool_trace": [
                *tool_trace,
                "history:adapter",
                "history:citation_validation",
                *(["history:retry"] if history_retry_used else []),
                *(["history:retry_selected"] if retry_selected else []),
                *(["history:retry_fallback_first"] if history_retry_used and not retry_selected else []),
            ],
            "latency_sec": elapsed_ms / 1000,
            "answer_provenance": {
                "source": "history_adapter",
                "history_adapter_called": True,
                "history_generation_calls": generation_calls,
                "history_retry_used": history_retry_used,
                "history_retry_reason": retry_reason if history_retry_used else None,
                "history_retry_decision": "retry" if retry_eligible else "skip",
                "history_retry_selected": retry_selected,
                "history_retry_error": retry_error,
                "history_first_answer_chars": first_generation.answer_chars,
                "history_first_answer_words": first_generation.answer_words,
                "history_final_answer_chars": final_generation.answer_chars,
                "history_final_answer_words": final_generation.answer_words,
                "history_first_quality_issues": first_generation.quality_issues,
                "history_final_quality_issues": final_quality_issues,
                "history_first_latency_ms": first_generation.latency_ms,
                "history_retry_latency_ms": retry_generation.latency_ms if retry_generation else 0.0,
                "history_total_latency_ms": elapsed_ms,
                "guard_short_circuit": False,
                "guard_name": None,
                "guard_override": False,
                "answer_depth": answer_depth,
                "structured_expansion_used": False,
            },
            "history_debug": {
                "generation_calls": generation_calls,
                "input_evidence_ids": input_ids,
                "input_claim_count": input_claim_count,
                "input_source_kind_counts": input_source_kind_counts,
                "input_evidence_preview": [
                    {
                        "evidence_id": str(item["chunk_id"]),
                        "text_preview": str(item.get("text") or "")[:220],
                        "claim_count": len(_context_claims(item)),
                        **(
                            {"comparison_target": item.get("comparison_target")}
                            if item.get("comparison_target")
                            else {}
                        ),
                    }
                    for item in contexts
                ],
                "comparison_evidence_groups": _comparison_evidence_groups(question, contexts)
                if question_type == "compare"
                else {},
                "comparison_dimension_coverage": comparison_dimension_coverage(question, contexts)
                if question_type == "compare"
                else {},
                "cited_ids": source_ids,
                "model_cited_ids": final_generation.parsed.source_ids,
                "conversation_history_used": False,
                "answer_depth": answer_depth,
                "question_type": question_type,
                "structured_expansion_used": False,
                "history_retry_used": history_retry_used,
                "history_retry_reason": retry_reason if history_retry_used else None,
                "history_retry_decision": "retry" if retry_eligible else "skip",
                "history_retry_selected": retry_selected,
                "history_retry_error": retry_error,
                "first_answer_chars": first_generation.answer_chars,
                "first_answer_words": first_generation.answer_words,
                "final_answer_chars": final_generation.answer_chars,
                "final_answer_words": final_generation.answer_words,
                "first_quality_issues": first_generation.quality_issues,
                "final_quality_issues": final_quality_issues,
                "first_latency_ms": first_generation.latency_ms,
                "retry_latency_ms": retry_generation.latency_ms if retry_generation else 0.0,
                "total_latency_ms": elapsed_ms,
                "initial_quality_issues": first_generation.quality_issues,
                "quality_warnings": final_quality_issues,
                "unsupported_years": unsupported_years,
            },
        }
