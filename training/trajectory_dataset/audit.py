from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Iterable

from .dedup import first_user_question, normalized_question
from .preprocess import analyze_truncation


def _describe(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {
        "p50": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.50) - 1)],
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "max": ordered[-1],
    }


def _tool_payloads(row: dict[str, Any]) -> list[list[dict[str, Any]]]:
    payloads = []
    for message in row.get("messages") or []:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            payload = []
        payloads.append(payload if isinstance(payload, list) else [])
    return payloads


def audit_rows(rows: Iterable[dict[str, Any]], *, strict_custom: bool = False) -> dict[str, Any]:
    materialized = list(rows)
    tasks = Counter(str(row.get("task_type") or "unknown") for row in materialized)
    sources = Counter(str(row.get("source_dataset") or "unknown") for row in materialized)
    subjects = Counter(str((row.get("provenance") or {}).get("subject_type") or "unknown") for row in materialized)
    observation_chars: list[int] = []
    empty_results = total_results = 0
    max_tool_calls = 0
    issue_counts: Counter[str] = Counter()
    issue_rows: dict[str, list[str]] = {}
    seen_questions: dict[str, str] = {}
    seen_compare_pairs: set[str] = set()

    def issue(name: str, row: dict[str, Any]) -> None:
        issue_counts[name] += 1
        issue_rows.setdefault(name, []).append(str(row.get("id") or "<missing>"))

    for row in materialized:
        provenance = row.get("provenance") or {}
        task = str(row.get("task_type") or "")
        subject = str(provenance.get("subject_type") or "unknown")
        question_key = normalized_question(first_user_question(row))
        if question_key in seen_questions:
            issue("duplicate_normalized_questions", row)
        elif question_key:
            seen_questions[question_key] = str(row.get("id") or "")
        tool_calls = sum(len(message.get("tool_calls") or []) for message in row.get("messages") or [])
        max_tool_calls = max(max_tool_calls, tool_calls)
        if task == "multihop" and tool_calls < 2:
            issue("multihop_fewer_than_2_tool_calls", row)
        if task == "cause" and subject in {"person", "location", "date", "topic"}:
            issue("cause_invalid_subject", row)
        if task in {"significance", "multihop"} and subject in {"person", "location", "date", "topic"}:
            issue("analytical_invalid_subject", row)
        if task == "summary" and subject in {"location", "date", "topic"}:
            issue("analytical_invalid_subject", row)
        if task == "compare":
            secondary_type = str(provenance.get("secondary_subject_type") or "")
            if not secondary_type or secondary_type != subject:
                issue("compare_type_mismatch", row)
            first = normalized_question(str(provenance.get("primary_title") or ""))
            second = normalized_question(str(provenance.get("secondary_title") or ""))
            if not first or first == second:
                issue("compare_identical_titles", row)
            pair = str(provenance.get("compare_pair_key") or "||".join(sorted((first, second))))
            if pair in seen_compare_pairs:
                issue("duplicate_compare_pair", row)
            elif first and second:
                seen_compare_pairs.add(pair)
        if task == "verification":
            claim = str(provenance.get("concrete_claim") or "").strip()
            if len(claim) < 20 or "claim about" in claim.casefold() or "nhận định về" in claim.casefold():
                issue("missing_concrete_verification_claim", row)
            if subject not in {"person", "event", "organization", "state", "dynasty", "document"}:
                issue("verification_invalid_subject", row)
        if task == "insufficient_evidence":
            claim = str(provenance.get("synthetic_claim") or "")
            if "Z-1901" not in claim:
                issue("missing_concrete_synthetic_claim", row)
            if subject not in {"person", "event", "organization", "state", "dynasty", "document"}:
                issue("insufficient_invalid_subject", row)

        observed_ids: set[str] = set()
        payloads = _tool_payloads(row)
        for payload in payloads:
            total_results += 1
            if not payload:
                empty_results += 1
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            observation_chars.append(len(encoded))
            observed_ids.update(
                str(result.get("chunk_id") or result.get("evidence_id"))
                for result in payload
                if isinstance(result, dict) and (result.get("chunk_id") or result.get("evidence_id"))
            )
        final_answer = str((row.get("messages") or [{}])[-1].get("content") or "")
        citations = set(re.findall(r"\[([^\[\]]+)\]", final_answer))
        declared = {str(value) for value in provenance.get("evidence_ids", [])}
        if not citations.issubset(observed_ids) or not declared.issubset(observed_ids):
            issue("grounded_answer_invalid_observed_citations", row)
        explicitly_insufficient = "chưa đủ bằng chứng" in final_answer.casefold()
        if (declared or (provenance.get("grounded") and observed_ids)) and not citations and not explicitly_insufficient:
            issue("grounded_answer_missing_citation", row)

    strict_issue_names = {
        "cause_invalid_subject", "analytical_invalid_subject", "compare_type_mismatch",
        "compare_identical_titles", "duplicate_compare_pair",
        "missing_concrete_verification_claim", "missing_concrete_synthetic_claim",
        "verification_invalid_subject", "insufficient_invalid_subject",
        "multihop_fewer_than_2_tool_calls", "duplicate_normalized_questions",
        "grounded_answer_invalid_observed_citations", "grounded_answer_missing_citation",
    }
    strict_violations = sum(issue_counts[name] for name in strict_issue_names)
    return {
        "rows": len(materialized),
        "task_counts": dict(sorted(tasks.items())),
        "source_counts": dict(sorted(sources.items())),
        "subject_type_counts": dict(sorted(subjects.items())),
        "issues": dict(sorted(issue_counts.items())),
        "issue_row_ids": {key: values[:20] for key, values in sorted(issue_rows.items())},
        "empty_tool_result_rate": round(empty_results / total_results, 6) if total_results else 0.0,
        "observation_chars": _describe(observation_chars),
        "max_tool_calls": max_tool_calls,
        "strict_custom": strict_custom,
        "strict_violation_count": strict_violations if strict_custom else 0,
        "valid": not strict_custom or strict_violations == 0,
    }


def tokenizer_audit(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    token_counts: list[int] = []
    too_long = user_lost = tool_calls_lost = assistant_lost = final_lost = all_lost = 0
    for row in rows:
        report = analyze_truncation(tokenizer, row, max_length=max_seq_length)
        token_counts.append(int(report["total_tokens"]))
        too_long += int(report["truncated"])
        user_lost += int(report["initial_user_lost"])
        tool_calls_lost += int(report["lost_tool_call_targets"] > 0)
        assistant_lost += int(report["lost_assistant_targets"] > 0)
        final_lost += int(report["final_assistant_lost"])
        all_lost += int(report["all_assistant_supervision_lost"])
    return {
        "rendered_tokens": _describe(token_counts),
        "rows_over_max_seq_length": too_long,
        "rows_initial_user_lost": user_lost,
        "rows_any_tool_call_supervision_lost": tool_calls_lost,
        "rows_any_assistant_supervision_lost": assistant_lost,
        "rows_final_assistant_supervision_lost": final_lost,
        "rows_all_assistant_supervision_lost": all_lost,
        "max_seq_length": max_seq_length,
    }
