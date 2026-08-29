from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from app.agents.history_contract import (
    SAFE_INSUFFICIENT_ANSWER,
    SAFE_OOD_ANSWER,
    build_history_answerer_messages,
    parse_history_answer_output,
)
from app.agents.model_runtime import RoleLLMBackend
from app.agents.comparison import (
    SHARED,
    TARGET_A,
    TARGET_B,
    UNKNOWN,
    classify_comparison_target,
    group_comparison_evidence,
)
from app.rag.retrieval import extract_comparison_targets, text_matches_target
from app.telemetry import current_request_telemetry, log_event

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
YEAR_RE = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
GENERIC_SOURCE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"theo\s+(?:các\s+)?(?:tài liệu|nguồn)(?:\s+(?:được cung cấp|tham khảo))?"
    r"|dựa\s+trên\s+(?:các\s+)?(?:tài liệu|nguồn)(?:\s+(?:được cung cấp|tham khảo))?"
    r"|từ\s+(?:các\s+)?(?:tài liệu|nguồn)(?:\s+(?:được cung cấp|tham khảo))?"
    r")\s*,\s*",
    re.I,
)
HISTORY_STOPWORDS = {
    "ai", "bao", "bi", "cai", "cho", "co", "cua", "da", "duoc", "gi",
    "khi", "la", "mot", "nam", "nao", "nhu", "nhung", "o", "ra", "sau",
    "tai", "the", "thi", "theo", "trong", "tu", "va", "ve", "voi",
}
DEEP_ANALYTICAL_CUES = {
    "nguyen nhan", "vi sao", "tai sao", "dan den", "suy yeu", "y nghia",
    "vai tro", "so sanh", "phan tich", "danh gia", "he qua", "tac dong",
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
    return GENERIC_SOURCE_PREFIX_RE.sub("", str(answer), count=1)


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


def _supported_years(question: str, contexts: list[dict[str, Any]]) -> set[str]:
    text = " ".join([question, *[str(context.get("text") or "") for context in contexts]])
    return set(YEAR_RE.findall(_normalize_text(text)))


def _unsupported_years(answer: str, question: str, contexts: list[dict[str, Any]]) -> list[str]:
    answer_years = set(YEAR_RE.findall(_normalize_text(answer)))
    return sorted(answer_years - _supported_years(question, contexts))


def _sentence_count(answer: str) -> int:
    return len(_candidate_sentences(answer))


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


def _deep_answer_quality_issues(
    question: str,
    contexts: list[dict[str, Any]],
    answer: str,
    *,
    answer_depth: str,
) -> list[str]:
    if not answer.strip():
        return ["empty_answer"]
    if answer_depth != "deep":
        return []

    issues: list[str] = []
    question_type = _history_question_type(question)
    validated_claim_count = _history_input_claim_count(contexts)
    breadth = max(validated_claim_count, len(contexts))
    sentence_count = _sentence_count(answer)
    answer_is_short = len(answer) < 520
    if question_type == "compare":
        missing_targets = _answer_mentions_supported_compare_targets(question, contexts, answer)
        issues.extend(f"missing_compare_target:{target}" for target in missing_targets)
        normalized_answer = _normalize_text(answer)
        has_compare_structure = any(
            marker in normalized_answer
            for marker in ("khai quat", "diem giong", "diem khac", "nhan xet")
        )
        if (
            (breadth >= 2 and sentence_count <= 1 and len(answer) < 320)
            or (breadth >= 4 and answer_is_short and (sentence_count <= 2 or not has_compare_structure))
        ):
            issues.append("deep_answer_collapse")
    elif question_type == "analysis":
        if (
            (breadth >= 2 and sentence_count <= 1 and len(answer) < 320)
            or (breadth >= 4 and answer_is_short and sentence_count <= 2)
        ):
            issues.append("deep_answer_collapse")
    return list(dict.fromkeys(issues))


def _deep_evidence_claims(
    question: str,
    contexts: list[dict[str, Any]],
    *,
    max_claims: int = 5,
) -> list[tuple[str, str]]:
    question_terms = _content_terms(question)
    scored: list[tuple[float, int, str, str]] = []
    for context_index, context in enumerate(contexts):
        evidence_id = str(context.get("chunk_id") or "").strip()
        if not evidence_id:
            continue
        for sentence_index, sentence in enumerate(_candidate_sentences(str(context.get("text") or ""))):
            terms = _content_terms(sentence)
            if not terms:
                continue
            overlap = len(question_terms & terms) / max(len(question_terms), 1)
            if overlap <= 0:
                continue
            score = overlap + min(len(sentence), 260) / 2000
            scored.append((score, -context_index * 100 - sentence_index, evidence_id, sentence))
    scored.sort(reverse=True)
    claims: list[tuple[str, str]] = []
    seen_text: set[str] = set()
    seen_ids: set[str] = set()
    for _, _, evidence_id, sentence in scored:
        key = _normalize_text(sentence)
        if key in seen_text:
            continue
        if evidence_id in seen_ids and len(claims) >= 2:
            continue
        seen_text.add(key)
        seen_ids.add(evidence_id)
        claims.append((evidence_id, sentence))
        if len(claims) >= max_claims:
            break
    return claims


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


def _expand_deep_compare_answer_from_evidence(
    question: str,
    contexts: list[dict[str, Any]],
) -> tuple[str, list[str]] | None:
    groups = _comparison_evidence_groups(question, contexts)
    if not groups:
        return None

    target_a_name = groups[TARGET_A]["name"]
    target_b_name = groups[TARGET_B]["name"]
    target_a_claims = _claim_lines(groups[TARGET_A]["evidence"])
    target_b_claims = _claim_lines(groups[TARGET_B]["evidence"])
    shared_claims = _claim_lines(groups["shared_evidence"])
    if not target_a_claims and not target_b_claims and not shared_claims:
        return None

    used_source_ids: list[str] = []

    def remember(claims: list[tuple[str, str]]) -> None:
        used_source_ids.extend(evidence_id for evidence_id, _ in claims)

    lines = [
        "Khái quát",
        f"{target_a_name} và {target_b_name} cần được đặt cạnh nhau theo đúng nhóm bằng chứng đã kiểm chứng.",
        "",
        "Điểm giống nhau",
    ]
    if shared_claims:
        lines.extend(f"- {claim}" for _, claim in shared_claims)
        remember(shared_claims)
    else:
        lines.append(
            "- Các bằng chứng đã chọn chưa nêu một điểm giống nhau đủ rõ cho cả hai đối tượng; phần so sánh chắc hơn nằm ở các khác biệt được chứng cứ riêng hỗ trợ."
        )

    lines.extend(["", "Điểm khác nhau"])
    if target_a_claims:
        lines.append(f"- {target_a_name}:")
        lines.extend(f"  - {claim}" for _, claim in target_a_claims)
        remember(target_a_claims)
    if target_b_claims:
        lines.append(f"- {target_b_name}:")
        lines.extend(f"  - {claim}" for _, claim in target_b_claims)
        remember(target_b_claims)

    if not target_a_claims or not target_b_claims:
        missing = target_a_name if not target_a_claims else target_b_name
        lines.append(f"- Bằng chứng riêng cho {missing} còn thiếu hoặc chưa đủ chắc để đối chiếu thêm chiều cạnh.")

    lines.extend(["", "Nhận xét"])
    if target_a_claims and target_b_claims:
        lines.append(
            f"Các nguồn cho phép so sánh trực tiếp ở phần khác nhau: {target_a_name} được mô tả bằng các dữ kiện riêng của nó, còn {target_b_name} được mô tả bằng một nhóm dữ kiện khác; vì vậy không được hoán đổi hoặc lặp một claim một phía sang phía còn lại."
        )
    else:
        lines.append(
            "Nhận xét nên giữ mức thận trọng vì nhóm bằng chứng hiện có chưa cân bằng hoàn toàn giữa hai đối tượng."
        )

    source_ids = list(dict.fromkeys(used_source_ids))
    return "\n".join(lines), source_ids


def _expand_deep_answer_from_evidence(
    question: str,
    contexts: list[dict[str, Any]],
    current_answer: str,
) -> tuple[str, list[str]] | None:
    question_type = _history_question_type(question)
    if question_type not in {"analysis", "compare"} or len(current_answer) >= 620:
        return None
    if question_type == "compare":
        return _expand_deep_compare_answer_from_evidence(question, contexts)
    claims = _deep_evidence_claims(question, contexts)
    if len(claims) < 2:
        return None
    selected_ids = list(dict.fromkeys(evidence_id for evidence_id, _ in claims))

    lines = ["Kết luận trực tiếp"]
    lines.append(claims[0][1])
    remaining = claims[1:]
    if remaining:
        lines.append("\nCác khía cạnh được tài liệu hỗ trợ")
        lines.extend(f"- {claim}" for _, claim in remaining)
    lines.append("\nTổng hợp")
    lines.append("Từ các dữ kiện này, câu trả lời nên được hiểu theo nhiều khía cạnh được nguồn nêu, thay vì chỉ dừng ở một kết luận ngắn.")
    return "\n".join(lines), selected_ids


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
                "initial_quality_issues": [],
                "quality_warnings": [],
                "unsupported_years": [],
            },
        }

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

        calls_before = telemetry.total_llm_calls if telemetry is not None else 0
        messages = build_history_answerer_messages(
            question,
            contexts,
            answer_depth=answer_depth,
            avoid_generic_source_prefix=avoid_generic_source_prefix,
        )
        raw_output = self.model_runtime.generate_text(
            adapter="history",
            messages=messages,
        )
        telemetry = current_request_telemetry()
        actual_calls = (telemetry.total_llm_calls - calls_before) if telemetry is not None else 1
        parsed = parse_history_answer_output(raw_output, allowed_source_ids=input_ids)
        by_id = {str(item["chunk_id"]): item for item in contexts}
        initial_quality_issues = _deep_answer_quality_issues(
            question,
            contexts,
            parsed.answer,
            answer_depth=answer_depth,
        )
        structured_expansion = (
            _expand_deep_answer_from_evidence(question, contexts, parsed.answer)
            if answer_depth == "deep" and "deep_answer_collapse" in initial_quality_issues
            else None
        )
        answer_text = (
            _remove_generic_source_prefix(parsed.answer)
            if avoid_generic_source_prefix
            else parsed.answer
        )
        source_ids = parsed.source_ids
        structured_expansion_used = False
        if structured_expansion is not None:
            answer_text, expanded_source_ids = structured_expansion
            source_ids = expanded_source_ids
            structured_expansion_used = True
        final_quality_issues = _deep_answer_quality_issues(
            question,
            contexts,
            answer_text,
            answer_depth=answer_depth,
        )
        unsupported_years = _unsupported_years(answer_text, question, contexts)
        if unsupported_years:
            final_quality_issues = [
                *final_quality_issues,
                *[f"unsupported_year:{year}" for year in unsupported_years],
            ]
        final_quality_issues = list(dict.fromkeys(final_quality_issues))
        source_chunks = [by_id[source_id] for source_id in source_ids]
        status = "ok" if source_ids else "insufficient"

        elapsed_ms = (time.perf_counter() - started) * 1000
        invalid_citation_count = len(set(parsed.source_ids) - set(input_ids))
        if telemetry is not None:
            telemetry.history_ms += elapsed_ms
            telemetry.history_generation_calls += actual_calls
        log_event(
            "HISTORY_COMPLETE",
            request_id=request_id,
            actual_llm_calls=actual_calls,
                elapsed_ms=elapsed_ms,
                cited_count=len(source_ids),
                invalid_citation_count=invalid_citation_count,
                structured_expansion_used=structured_expansion_used,
                quality_issues=final_quality_issues,
            )
        return {
            "question": question,
            "answer": answer_text,
            "status": status,
            "source_ids": source_ids,
            "source_chunks": source_chunks,
            "model_source_ids": parsed.source_ids,
            "invalid_source_ids": [],
            "unsupported_years": unsupported_years,
            "format_ok": True,
            "raw_output": parsed.raw_output,
            "retrieval": retrieval,
            "analysis": analysis,
            "prompt_budget": None,
            "support_score": None,
            "quality_warnings": final_quality_issues,
            "rewrite_used": False,
            "repair_attempted": False,
            "structured_expansion_used": structured_expansion_used,
            "initial_quality_issues": initial_quality_issues,
            "history_message_count": 0,
            "tool_trace": [
                *tool_trace,
                "history:adapter",
                "history:citation_validation",
                *(["history:deep_structured_expansion"] if structured_expansion_used else []),
            ],
            "latency_sec": elapsed_ms / 1000,
            "answer_provenance": {
                "source": "history_adapter",
                "history_adapter_called": True,
                "history_generation_calls": 1,
                "guard_short_circuit": False,
                "guard_name": None,
                "guard_override": False,
                "answer_depth": answer_depth,
                "structured_expansion_used": structured_expansion_used,
            },
            "history_debug": {
                "generation_calls": 1,
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
                "cited_ids": source_ids,
                "model_cited_ids": parsed.source_ids,
                "conversation_history_used": False,
                "answer_depth": answer_depth,
                "question_type": question_type,
                "structured_expansion_used": structured_expansion_used,
                "initial_quality_issues": initial_quality_issues,
                "quality_warnings": final_quality_issues,
                "unsupported_years": unsupported_years,
            },
        }
