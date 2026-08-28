from __future__ import annotations

import re
import time
from typing import Any

from app.agents.history_contract import (
    SAFE_INSUFFICIENT_ANSWER,
    SAFE_OOD_ANSWER,
    build_history_answerer_messages,
    parse_history_answer_output,
)
from app.agents.model_runtime import RoleLLMBackend
from app.telemetry import current_request_telemetry, log_event

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
HISTORY_STOPWORDS = {
    "ai", "bao", "bi", "cai", "cho", "co", "cua", "da", "duoc", "gi",
    "khi", "la", "mot", "nam", "nao", "nhu", "nhung", "o", "ra", "sau",
    "tai", "the", "thi", "theo", "trong", "tu", "va", "ve", "voi",
}
DEEP_ANALYTICAL_CUES = {
    "nguyen nhan", "vi sao", "tai sao", "dan den", "suy yeu", "y nghia",
    "vai tro", "so sanh", "phan tich", "danh gia", "he qua", "tac dong",
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


def _candidate_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(str(text)):
        sentence = sentence.strip(" \t\r\n-•")
        if sentence:
            sentences.append(sentence)
    return sentences


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


def _expand_deep_answer_from_evidence(
    question: str,
    contexts: list[dict[str, Any]],
    current_answer: str,
) -> tuple[str, list[str]] | None:
    if not _is_deep_analytical_question(question) or len(contexts) < 2 or len(current_answer) >= 420:
        return None
    claims = _deep_evidence_claims(question, contexts)
    if len(claims) < 2:
        return None
    selected_ids = list(dict.fromkeys(evidence_id for evidence_id, _ in claims))
    lines = ["Trả lời có thể triển khai sâu hơn từ các bằng chứng đã chọn:"]
    for index, (_, claim) in enumerate(claims, start=1):
        lines.append(f"{index}. {claim}")
    lines.append("Tổng hợp lại, ý nghĩa chính cần rút ra nằm ở các hệ quả trực tiếp được những bằng chứng trên nêu ra.")
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
                "input_evidence_preview": [],
                "cited_ids": [],
                "conversation_history_used": False,
                "answer_depth": answer_depth,
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
        log_event(
            "HISTORY_START",
            request_id=request_id,
            input_evidence_count=len(contexts),
            input_evidence_ids=input_ids,
            answer_depth=answer_depth,
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

        telemetry = current_request_telemetry()
        calls_before = telemetry.total_llm_calls if telemetry is not None else 0
        messages = build_history_answerer_messages(
            question,
            contexts,
            answer_depth=answer_depth,
        )
        raw_output = self.model_runtime.generate_text(
            adapter="history",
            messages=messages,
        )
        telemetry = current_request_telemetry()
        actual_calls = (telemetry.total_llm_calls - calls_before) if telemetry is not None else 1
        parsed = parse_history_answer_output(raw_output, allowed_source_ids=input_ids)
        by_id = {str(item["chunk_id"]): item for item in contexts}
        structured_expansion = (
            _expand_deep_answer_from_evidence(question, contexts, parsed.answer)
            if answer_depth == "deep"
            else None
        )
        answer_text = parsed.answer
        source_ids = parsed.source_ids
        structured_expansion_used = False
        if structured_expansion is not None:
            answer_text, expanded_source_ids = structured_expansion
            source_ids = expanded_source_ids
            structured_expansion_used = True
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
            )
        return {
            "question": question,
            "answer": answer_text,
            "status": status,
            "source_ids": source_ids,
            "source_chunks": source_chunks,
            "model_source_ids": parsed.source_ids,
            "invalid_source_ids": [],
            "unsupported_years": [],
            "format_ok": True,
            "raw_output": parsed.raw_output,
            "retrieval": retrieval,
            "analysis": analysis,
            "prompt_budget": None,
            "support_score": None,
            "quality_warnings": [],
            "rewrite_used": False,
            "repair_attempted": False,
            "structured_expansion_used": structured_expansion_used,
            "initial_quality_issues": [],
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
                "input_evidence_preview": [
                    {
                        "evidence_id": str(item["chunk_id"]),
                        "text_preview": str(item.get("text") or "")[:220],
                    }
                    for item in contexts
                ],
                "cited_ids": source_ids,
                "model_cited_ids": parsed.source_ids,
                "conversation_history_used": False,
                "answer_depth": answer_depth,
                "structured_expansion_used": structured_expansion_used,
            },
        }
