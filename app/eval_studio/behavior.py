"""Deterministic observations, not historical correctness or semantic grading.

Lexical answer scoring includes an O(n*m) LCS implementation. It is skipped
(null metrics, never a prefix score) above 8,192 characters or 512 tokens per
answer/reference. Evidence inspection is bounded to 500 rows/65,536 characters;
metrics needing an incomplete input remain null. Stored artifacts are unchanged.
"""

from collections import Counter
import json
import re
import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from evaluation.metrics.answer import score_answer, tokens
from evaluation.metrics.citations import score_citations
from evaluation.metrics.grounding import score_grounding

SCHEMA_VERSION = "1.2"
METRIC_VERSION = "behavior-v1"
MAX_METRIC_CHARS = 8192
MAX_METRIC_TOKENS = 512
MAX_EVIDENCE_ROWS = 500
MAX_EVIDENCE_CHARS = 65536
ToolName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
EvidenceId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Term = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
ToolGroup = Annotated[list[ToolName], Field(min_length=1, max_length=30)]


class BehaviorSpec(BaseModel):
    """Declarative expectations without an imposed tool-call sequence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    required_tools: list[ToolName] = Field(default_factory=list, max_length=100)
    any_tool_groups: list[ToolGroup] = Field(default_factory=list, max_length=30)
    forbidden_tools: list[ToolName] = Field(default_factory=list, max_length=100)
    allowed_tools: list[ToolName] | None = Field(default=None, max_length=100)
    max_tool_calls: int | None = Field(default=None, ge=0, le=10000)
    max_repeated_calls: int | None = Field(default=None, ge=0, le=10000)
    expected_evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=100)
    evidence_terms: list[Term] = Field(default_factory=list, max_length=100)
    expected_exit_reasons: list[ToolName] = Field(default_factory=list, max_length=30)
    output_format: Literal["text", "json", "markdown"] | None = None
    required_json_keys: list[EvidenceId] = Field(default_factory=list, max_length=100)
    require_citations: bool | None = None
    answerable: bool | None = None
    in_domain: bool | None = None


def _normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _rate(values):
    return sum(values) / len(values) if values else None


def _json(answer):
    def invalid_constant(value):
        raise ValueError(value)

    try:
        return True, json.loads(answer, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError):
        return False, None


def evaluate_behavior(question: dict, run: dict, evidence: list, diagnostics: dict) -> dict:
    """Score only observed metadata; every absent expectation/check is null.

    Capture flags are authoritative. In particular, legacy aggregate tool counts
    cannot stand in for a captured tool trace. Argument/success rates report only
    explicitly observed labels and include their denominators. Repetition needs
    every fingerprint, and counts duplicate (name, fingerprint) calls after the
    first occurrence, independently of call order.
    """
    spec = BehaviorSpec.model_validate(question.get("expected_behavior") or {})
    diagnostics = diagnostics or {}
    flags = diagnostics.get("capture") or {}
    capture = {key: flags.get(key) is True for key in ("tool_trace", "retrieved_evidence", "final_context")}
    calls = diagnostics.get("tool_calls")
    tool_known = capture["tool_trace"] and isinstance(calls, list)
    calls = calls if tool_known else []
    names = {item.get("name") for item in calls}
    valid = [item["arguments_valid"] for item in calls if type(item.get("arguments_valid")) is bool]
    successes = [
        item.get("status") in ("ok", "success", "completed")
        for item in calls
        if item.get("status") in ("ok", "success", "completed", "error", "failed", "rejected", "cancelled")
    ]
    fingerprints_known = tool_known and all(
        isinstance(item.get("argument_fingerprint"), str) and item["argument_fingerprint"] for item in calls
    )
    repeated = None
    if fingerprints_known:
        counts = Counter((item.get("name"), item["argument_fingerprint"]) for item in calls)
        repeated = sum(count - 1 for count in counts.values())
    tools = {
        "call_count": len(calls) if tool_known else None,
        "argument_validity_rate": _rate(valid) if tool_known else None,
        "argument_validity_observed": len(valid) if tool_known else None,
        "success_rate": _rate(successes) if tool_known else None,
        "success_observed": len(successes) if tool_known else None,
        "repeated_call_count": repeated,
        "repeated_call_rate": repeated / len(calls) if repeated is not None and calls else None,
    }
    checks = {
        "required_tools": set(spec.required_tools) <= names if tool_known and spec.required_tools else None,
        "any_tool_groups": (
            all(names.intersection(group) for group in spec.any_tool_groups)
            if tool_known and spec.any_tool_groups
            else None
        ),
        "forbidden_tools": (
            not bool(names.intersection(spec.forbidden_tools))
            if tool_known and spec.forbidden_tools
            else None
        ),
        "allowed_tools": (
            names <= set(spec.allowed_tools) if tool_known and spec.allowed_tools is not None else None
        ),
        "max_tool_calls": (
            len(calls) <= spec.max_tool_calls if tool_known and spec.max_tool_calls is not None else None
        ),
        "max_repeated_calls": (
            repeated <= spec.max_repeated_calls
            if repeated is not None and spec.max_repeated_calls is not None
            else None
        ),
    }
    exit_reason = (diagnostics.get("planning") or {}).get("exit_reason")
    checks["expected_exit_reason"] = (
        exit_reason in spec.expected_exit_reasons
        if isinstance(exit_reason, str) and exit_reason.strip() and spec.expected_exit_reasons
        else None
    )

    rows = []
    evidence_chars = 0
    evidence_truncated = len(evidence) > MAX_EVIDENCE_ROWS
    for item in evidence[:MAX_EVIDENCE_ROWS]:
        text = item.get("excerpt") or item.get("text") or ""
        if evidence_chars + len(text) > MAX_EVIDENCE_CHARS:
            evidence_truncated = True
            break
        evidence_chars += len(text)
        rows.append({**item, "text": text})
    retrieved_known = capture["retrieved_evidence"] and not evidence_truncated
    final_known = capture["final_context"] and not evidence_truncated
    selected = [item for item in rows if item.get("selected") is True]
    if selected and all(type(item.get("final_rank")) is int for item in selected):
        selected.sort(key=lambda item: item["final_rank"])
    ids = {str(item[key]) for item in rows for key in ("chunk_id", "source_id") if item.get(key)}
    expected_ids = set(spec.expected_evidence_ids)
    id_recall = len(ids & expected_ids) / len(expected_ids) if expected_ids and retrieved_known else None
    text = _normal("\n".join(item["text"] for item in rows))
    term_recall = _rate([_normal(term) in text for term in spec.evidence_terms]) if retrieved_known else None
    evidence_metrics = {
        "retrieved_count": len(rows) if retrieved_known else None,
        "final_context_count": len(selected) if final_known else None,
        "expected_id_recall": id_recall,
        "term_recall": term_recall,
    }
    checks["evidence_ids"] = id_recall == 1 if id_recall is not None else None
    checks["evidence_terms"] = term_recall == 1 if term_recall is not None else None

    answer = run.get("answer")
    answer_known = isinstance(answer, str) and (
        bool(answer.strip()) or run.get("completion_status", "completed") == "completed"
    )
    answer = answer if answer_known else ""
    gold = question.get("reference_answer") or None
    if isinstance(gold, str) and not gold.strip():
        gold = None
    answer_within_limit = len(answer) <= MAX_METRIC_CHARS
    lexical_within_limit = answer_within_limit and (gold is None or len(gold) <= MAX_METRIC_CHARS)
    if lexical_within_limit:
        lexical_within_limit = len(tokens(answer)) <= MAX_METRIC_TOKENS and (
            gold is None or len(tokens(gold)) <= MAX_METRIC_TOKENS
        )
    answer_metrics = (
        score_answer(answer, gold) if answer_known and lexical_within_limit else score_answer("", None)
    )
    citation_metrics = score_citations(
        answer if answer_within_limit else "",
        selected if final_known else [],
        gold_source_ids=spec.expected_evidence_ids or None,
    )
    if not final_known or not answer_known or not answer_within_limit:
        citation_metrics.update(
            {
                key: None
                for key in (
                    "citation_validity_rate",
                    "cited_retrieved_rate",
                    "citation_precision",
                    "citation_recall",
                    "source_id_existence_rate",
                    "answer_citation_coverage",
                    "resolved_source_ids",
                    "invalid_citation_ids",
                )
            }
        )
    facts = question.get("expected_facts") or question.get("must_cover_points") or []
    facts_within_limit = len(facts) <= 100 and all(
        isinstance(fact, str) and len(fact) <= 1000 for fact in facts
    )
    grounding = score_grounding(
        answer if answer_within_limit else "",
        str(question.get("question") or "")[:MAX_METRIC_CHARS],
        selected if final_known else [],
        required_facts=facts if facts_within_limit else None,
        answerable=spec.answerable,
        in_domain=spec.in_domain,
    )
    if not answer_known or not answer_within_limit:
        grounding = dict.fromkeys(grounding)
    elif not final_known:
        grounding.update(unverified_years=None, unverified_year_rate=None)

    json_ok, parsed = _json(answer) if answer_known and answer_within_limit else (None, None)
    checks["output_format"] = None
    if spec.output_format and answer_known and answer_within_limit:
        checks["output_format"] = {
            "text": bool(answer.strip()),
            "json": json_ok,
            # Markdown/plain text are not disjoint formats: this is a surface heuristic.
            "markdown": bool(
                re.search(r"(?m)^\s*(?:#{1,6}\s|[-*+]\s|\d+\.\s|```)|\*\*.+?\*\*|\[[^]]+\]\([^)]+\)", answer)
            ),
        }[spec.output_format]
    checks["required_json_keys"] = (
        (bool(json_ok) and isinstance(parsed, dict) and set(spec.required_json_keys) <= set(parsed))
        if spec.required_json_keys and json_ok is not None
        else None
    )
    checks["citations"] = None
    if spec.require_citations is not None and answer_known and answer_within_limit and final_known:
        aliases = citation_metrics["cited_aliases"]
        checks["citations"] = (
            bool(aliases) and not citation_metrics["invalid_citation_ids"]
            if spec.require_citations
            else not bool(aliases)
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "metric_labels": {
            "tools": "observed diagnostics; rates use explicitly labeled calls",
            "answer_metrics": "lexical heuristic, not historical correctness",
            "citation_metrics": "citation ID heuristic, not source support or correctness",
            "grounding_metrics": "phrase/year/refusal heuristic, not semantic entailment",
            "evidence": "ID/phrase heuristic, not evidence relevance",
            "checks.output_format": "shape/Markdown surface heuristic",
        },
        "capture": capture,
        "checks": checks,
        "tools": tools,
        "evidence": evidence_metrics,
        "answer_metrics": answer_metrics,
        "citation_metrics": citation_metrics,
        "grounding_metrics": grounding,
        "semantic_checks": dict.fromkeys(
            ("historical_correctness", "reasoning_quality", "evidence_entailment")
        ),
        "human_required": ["historical_correctness", "reasoning_quality", "evidence_entailment"],
        "limits": {
            "max_metric_chars": MAX_METRIC_CHARS,
            "max_metric_tokens": MAX_METRIC_TOKENS,
            "max_evidence_rows": MAX_EVIDENCE_ROWS,
            "max_evidence_chars": MAX_EVIDENCE_CHARS,
            "answer_metrics_skipped": not answer_known or not lexical_within_limit,
            "evidence_metrics_skipped": evidence_truncated,
            "required_facts_skipped": not facts_within_limit,
        },
    }
