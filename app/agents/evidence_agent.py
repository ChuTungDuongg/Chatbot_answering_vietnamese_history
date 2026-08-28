from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agents.model_runtime import RoleLLMBackend
from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.schemas import EvidenceAgentRequest, EvidenceCritique, EvidenceModelOutput, SelectedEvidence
from app.agents.evidence_validation import (
    compressed_derived_from_own_claims,
    grounded_in_source,
    normalize_grounding,
    referenced_evidence_ids,
)
from app.telemetry import current_request_telemetry, log_event


WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
QUESTION_STOPWORDS = {
    "ai", "bao", "bi", "cai", "cho", "co", "cua", "da", "duoc", "gi",
    "khi", "la", "mot", "nam", "nao", "nhu", "nhung", "o", "ra", "sau",
    "tai", "the", "thi", "theo", "trong", "tu", "va", "ve", "voi",
}
RECOVERY_STOPWORDS = QUESTION_STOPWORDS | {
    "ay", "bang", "cau", "chi", "do", "nay", "nen", "thuoc",
}
EVIDENCE_TEXT_BUDGET = 14_000
MAX_EVIDENCE_ITEM_CHARS = 3_200
MIN_EVIDENCE_ITEM_CHARS = 1_200
MAX_RECOVERY_SPAN_CHARS = 700
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
                    raise EvidenceModelContractError(f"Evidence model output failed canonical schema validation: {exc}") from exc
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
    ) -> tuple[EvidenceAgentRequest, dict[str, dict[str, Any]]]:
        # Canonical Evidence SFT contains at most seven candidates. Keep the
        # production pool close to that distribution and preserve retrieval order.
        available: dict[str, dict[str, Any]] = {}
        for item in evidence:
            chunk_id = str(item.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in available:
                continue
            available[chunk_id] = item
            if len(available) >= self.max_contexts:
                break
        per_item_limit = min(
            MAX_EVIDENCE_ITEM_CHARS,
            max(MIN_EVIDENCE_ITEM_CHARS, EVIDENCE_TEXT_BUDGET // max(len(available), 1)),
        )
        request = EvidenceAgentRequest.model_validate({
            "question": question,
            "max_selected": min(max(final_k, 1), self.max_contexts),
            "evidence": [
                {
                    "evidence_id": chunk_id,
                    "source_type": item.get("source_kind", "local"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "chunk_id": chunk_id,
                    "text": question_relevant_excerpt(
                        str(item.get("text", "")),
                        question,
                        max_chars=per_item_limit,
                    ),
                    "retrieval_score": item.get("score") or item.get("reranker_score"),
                }
                for chunk_id, item in available.items()
            ],
        })
        return request, available

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
            )
        try:
            return EvidenceModelOutput.model_validate(output)
        except ValidationError as exc:
            raise EvidenceModelContractError(
                f"Evidence model returned invalid canonical output: {exc}",
                code="invalid_evidence_schema",
                repair_attempted=repair_attempted,
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
        question: str,
        generation_calls: int,
        repair_used: bool,
        repair_path: str | None,
    ) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        selected = model_output.selected_evidence[: self.max_contexts]
        contexts: list[dict[str, Any]] = []
        for item in selected:
            context = dict(available[item.evidence_id])
            context["text"] = item.compressed_text or str(context.get("text", ""))
            contexts.append(context)
        selected_ids = [item.evidence_id for item in selected]
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
        request, available = self._build_evidence_request(question, evidence, final_k=final_k)
        visible_sources = {item.evidence_id: item.text for item in request.evidence}
        selected_candidate_count = min(max(final_k, 1), self.max_contexts)
        generation_calls = 1
        repair_used = False
        repair_path: str | None = None
        log_event(
            "EVIDENCE_START",
            request_id=request_id,
            attempt=telemetry.evidence_attempts if telemetry is not None else None,
            candidate_count=len(evidence),
            model_visible_count=len(request.evidence),
            final_k=final_k,
            per_item_limit=len(request.evidence[0].text) if request.evidence else 0,
            text_budget=EVIDENCE_TEXT_BUDGET,
        )

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
        output = self.model_runtime.generate_json(
            adapter="evidence",
            messages=self._evidence_messages(request),
            max_new_tokens=768,
            repair=False,
        )
        log_event(
            "EVIDENCE_GENERATION_END",
            request_id=request_id,
            generation_number=1,
            elapsed_ms=(time.perf_counter() - generation_started) * 1000,
        )
        model_output = self._parse_model_output(output)
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
            if not all(issue.recoverable for issue in issues):
                self._raise_contract_error(issues, repair_attempted=False)

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
            else:
                generation_calls = 2
                repair_used = True
                repair_path = "model"
                if telemetry is not None:
                    telemetry.evidence_recovery_used = True
                    telemetry.evidence_repair_used = True
                repair_started = time.perf_counter()
                log_event(
                    "EVIDENCE_MODEL_REPAIR_START",
                    request_id=request_id,
                    issue_codes=[issue.code for issue in recovered_issues],
                )
                logger.info(
                    "evidence_repair_generation_start",
                    extra={
                        "request_id": request_id,
                        "generation_call": generation_calls,
                        "input_evidence_count": len(request.evidence),
                        "selected_candidate_count": selected_candidate_count,
                    },
                )
                repair_output = self.model_runtime.generate_json(
                    adapter="evidence",
                    messages=self._repair_messages(
                        request=request,
                        invalid_output=output,
                        validation_errors=[issue.as_dict() for issue in recovered_issues],
                    ),
                    max_new_tokens=768,
                    repair=False,
                )
                logger.info(
                    "evidence_repair_generation_end",
                    extra={
                        "request_id": request_id,
                        "elapsed_ms": (time.perf_counter() - repair_started) * 1000,
                        "output_tokens": len(json.dumps(repair_output, ensure_ascii=False).split()),
                    },
                )
                log_event(
                    "EVIDENCE_MODEL_REPAIR_END",
                    request_id=request_id,
                    elapsed_ms=(time.perf_counter() - repair_started) * 1000,
                )
                model_output = self._parse_model_output(repair_output, repair_attempted=True)
                repair_issues = self._contract_issues(model_output, visible_sources)
                if repair_issues:
                    log_event(
                        "EVIDENCE_REPAIR_FAILED",
                        request_id=request_id,
                        final_issue_codes=[issue.code for issue in repair_issues],
                    )
                    self._log_validation_failed(request_id, repair_issues, "failed_after_model_repair")
                    self._raise_contract_error(repair_issues, repair_attempted=True)
                log_event("EVIDENCE_REPAIR_SUCCESS", request_id=request_id)

        critique, contexts = self._critique_from_output(
            model_output,
            available=available,
            request=request,
            question=question,
            generation_calls=generation_calls,
            repair_used=repair_used,
            repair_path=repair_path,
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
            telemetry.evidence_repair_used = telemetry.evidence_repair_used or repair_path == "model"
            telemetry.evidence_recovery_used = telemetry.evidence_recovery_used or repair_path in {"deterministic", "model"}
        log_event(
            "EVIDENCE_COMPLETE",
            request_id=request_id,
            attempt=telemetry.evidence_attempts if telemetry is not None else None,
            status=critique.status,
            actual_llm_calls=actual_calls,
            recovery_used=repair_path in {"deterministic", "model"},
            repair_used=repair_path == "model",
            selected_count=len(contexts),
            elapsed_ms=elapsed_ms,
        )
        return critique, contexts
