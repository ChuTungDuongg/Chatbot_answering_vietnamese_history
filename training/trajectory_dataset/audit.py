from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Iterable

from .builders.custom_history import (
    is_result_facet_relevant,
    is_result_relevant_to_target,
    is_vietnam_history_relevant,
    result_text_mentions_target,
    title_implied_subject_type,
)
from .citations import extract_evidence_citations
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


def _empty_breakdown(groups: dict[str, Counter[str]]) -> dict[str, dict[str, int | float]]:
    report: dict[str, dict[str, int | float]] = {}
    for key, counts in sorted(groups.items()):
        values: dict[str, int | float] = dict(sorted(counts.items()))
        total = counts["tool_results_total"]
        values["empty_tool_result_rate"] = round(
            counts["empty_tool_results_total"] / total, 6,
        ) if total else 0.0
        report[key] = values
    return report


def audit_rows(rows: Iterable[dict[str, Any]], *, strict_custom: bool = False) -> dict[str, Any]:
    materialized = list(rows)
    tasks = Counter(str(row.get("task_type") or "unknown") for row in materialized)
    sources = Counter(str(row.get("source_dataset") or "unknown") for row in materialized)
    subjects = Counter(str((row.get("provenance") or {}).get("subject_type") or "unknown") for row in materialized)
    observation_chars: list[int] = []
    empty_results = expected_empty_results = unexpected_empty_results = total_results = 0
    empty_by_task: dict[str, Counter[str]] = {}
    empty_by_role: dict[str, Counter[str]] = {}
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
        source_dataset = str(row.get("source_dataset") or "")
        primary_title = str(provenance.get("primary_title") or "")
        secondary_title = str(provenance.get("secondary_title") or "")
        primary_aliases = provenance.get("primary_aliases") or []
        secondary_aliases = provenance.get("secondary_aliases") or []
        if not isinstance(primary_aliases, list):
            primary_aliases = []
        if not isinstance(secondary_aliases, list):
            secondary_aliases = []
        if source_dataset.startswith("custom_history") and primary_title:
            implied_type = title_implied_subject_type(primary_title)
            if implied_type is not None and implied_type != subject:
                issue("subject_type_mismatch", row)
            secondary_type = str(provenance.get("secondary_subject_type") or "")
            secondary_implied_type = title_implied_subject_type(secondary_title)
            if secondary_title and secondary_implied_type is not None and secondary_implied_type != secondary_type:
                issue("subject_type_mismatch", row)
            primary_signals = provenance.get("vietnam_history_relevance_signals") or []
            primary_domain_ok = (
                provenance.get("vietnam_history_relevant") is True
                and isinstance(primary_signals, list)
                and bool(primary_signals)
            )
            if not primary_domain_ok and not is_vietnam_history_relevant({
                "title": primary_title,
                "metadata": {"subject_type": subject},
            }):
                issue("domain_mismatch", row)
            if secondary_title:
                secondary_signals = provenance.get("secondary_vietnam_history_relevance_signals") or []
                secondary_domain_ok = (
                    provenance.get("secondary_vietnam_history_relevant") is True
                    and isinstance(secondary_signals, list)
                    and bool(secondary_signals)
                )
                if not secondary_domain_ok and not is_vietnam_history_relevant({
                    "title": secondary_title,
                    "metadata": {"subject_type": secondary_type},
                }):
                    issue("domain_mismatch", row)
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
        retrieval_queries = provenance.get("retrieval_queries")
        query_metadata = retrieval_queries if isinstance(retrieval_queries, list) else []
        roles = [
            str(query_metadata[index].get("role") or "unknown")
            if index < len(query_metadata) and isinstance(query_metadata[index], dict)
            else str(payload[0].get("retrieval_role") or "unknown")
            if payload and isinstance(payload[0], dict)
            else "unknown"
            for index, payload in enumerate(payloads)
        ]
        results_by_role = {role: payload for role, payload in zip(roles, payloads)}
        direct_verification_present = bool(results_by_role.get("claim_support"))
        trajectory_chars = 0
        semantic_entity_by_id: dict[str, bool] = {}
        semantic_facet_by_id: dict[str, bool] = {}
        semantic_role_by_id: dict[str, str] = {}
        observation_target_mismatch = False
        observation_facet_mismatch = False
        compare_target_contamination = False
        for index, payload in enumerate(payloads):
            total_results += 1
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            payload_chars = len(encoded)
            trajectory_chars += payload_chars
            observation_chars.append(payload_chars)
            observed_ids.update(
                str(result.get("chunk_id") or result.get("evidence_id"))
                for result in payload
                if isinstance(result, dict) and (result.get("chunk_id") or result.get("evidence_id"))
            )
            role = roles[index]
            metadata = query_metadata[index] if index < len(query_metadata) and isinstance(query_metadata[index], dict) else {}
            target_title = secondary_title if role == "target_b" else primary_title
            target_type = str(provenance.get("secondary_subject_type") or "") if role == "target_b" else subject
            target_aliases = secondary_aliases if role == "target_b" else primary_aliases
            other_title = primary_title if role == "target_b" else secondary_title if role == "target_a" else ""
            other_type = subject if role == "target_b" else str(provenance.get("secondary_subject_type") or "")
            other_aliases = primary_aliases if role == "target_b" else secondary_aliases
            for result in payload:
                if not isinstance(result, dict) or not target_title:
                    continue
                result_id = str(result.get("chunk_id") or result.get("evidence_id") or "")
                entity_relevant = is_result_relevant_to_target(
                    result,
                    target_title=target_title,
                    target_subject_type=target_type,
                    retrieval_role=role,
                    target_aliases=target_aliases,
                )
                if target_type == "person" and role in {
                    "factual", "biography_timeline", "role_contribution", "target_a", "target_b",
                    "claim_support",
                }:
                    entity_relevant = entity_relevant and result_text_mentions_target(
                        result, target_title=target_title, target_aliases=target_aliases,
                    )
                facet_relevant = is_result_facet_relevant(
                    result,
                    query=str(metadata.get("query") or ""),
                    task_type=task,
                    retrieval_role=role,
                    target_subject_type=target_type,
                    target_title=target_title,
                    target_aliases=target_aliases,
                )
                semantic_entity_by_id[result_id] = entity_relevant
                semantic_facet_by_id[result_id] = facet_relevant
                semantic_role_by_id[result_id] = role
                observation_target_mismatch = observation_target_mismatch or not entity_relevant
                observation_facet_mismatch = observation_facet_mismatch or not facet_relevant
                if task == "compare" and other_title:
                    contaminates = is_result_relevant_to_target(
                        result,
                        target_title=other_title,
                        target_subject_type=other_type,
                        retrieval_role=role,
                        target_aliases=other_aliases,
                    )
                    compare_target_contamination = compare_target_contamination or contaminates or not entity_relevant
            required = bool(metadata.get("required", False))
            expected_empty = bool(metadata.get("expected_empty", False))
            if "required" not in metadata and "expected_empty" not in metadata:
                expected_empty = (
                    task == "insufficient_evidence"
                    or (task == "hard_negative" and role == "wrong_facet")
                    or (
                        task == "verification" and role in {"corroboration", "external_corroboration"}
                        and direct_verification_present
                    )
                )
                required = not expected_empty
            if task == "verification" and role in {"corroboration", "external_corroboration"}:
                expected_empty = expected_empty and direct_verification_present
            task_counter = empty_by_task.setdefault(task or "unknown", Counter())
            role_counter = empty_by_role.setdefault(role, Counter())
            task_counter["tool_results_total"] += 1
            role_counter["tool_results_total"] += 1
            if not payload:
                empty_results += 1
                task_counter["empty_tool_results_total"] += 1
                role_counter["empty_tool_results_total"] += 1
                if expected_empty and not required:
                    expected_empty_results += 1
                    task_counter["expected_empty_tool_results"] += 1
                    role_counter["expected_empty_tool_results"] += 1
                else:
                    unexpected_empty_results += 1
                    task_counter["unexpected_empty_tool_results"] += 1
                    role_counter["unexpected_empty_tool_results"] += 1
                    if str(row.get("source_dataset") or "").startswith("custom_history"):
                        issue("unexpected_empty_tool_results", row)
        if source_dataset.startswith("custom_history") and observation_target_mismatch:
            issue("observation_target_mismatch", row)
        if source_dataset.startswith("custom_history") and observation_facet_mismatch:
            issue("observation_facet_mismatch", row)
        if source_dataset.startswith("custom_history") and compare_target_contamination:
            issue("compare_target_contamination", row)
        configured_budget = provenance.get("trajectory_observation_char_budget")
        if isinstance(configured_budget, int) and trajectory_chars > configured_budget:
            issue("trajectory_observation_budget_exceeded", row)
        final_answer = str((row.get("messages") or [{}])[-1].get("content") or "")
        parsed_citations = extract_evidence_citations(final_answer, observed_ids)
        citations = set(parsed_citations.citations)
        declared = {str(value) for value in provenance.get("evidence_ids", [])}
        if parsed_citations.unknown_ids or not citations.issubset(observed_ids) or not declared.issubset(observed_ids):
            issue("grounded_answer_invalid_observed_citations", row)
        answer_evidence_ids = citations | declared
        final_target_mismatch = source_dataset.startswith("custom_history") and any(
            not semantic_entity_by_id.get(evidence_id, False) for evidence_id in answer_evidence_ids
        )
        final_facet_mismatch = source_dataset.startswith("custom_history") and any(
            not semantic_facet_by_id.get(evidence_id, False) for evidence_id in answer_evidence_ids
        )
        if (
            source_dataset.startswith("custom_history")
            and task == "compare"
            and not "chưa đủ bằng chứng" in final_answer.casefold()
            and {semantic_role_by_id.get(evidence_id) for evidence_id in answer_evidence_ids} != {"target_a", "target_b"}
        ):
            final_target_mismatch = True
        final_target_mismatch = final_target_mismatch or (
            source_dataset.startswith("custom_history")
            and subject == "person"
            and task != "insufficient_evidence"
            and not "chưa đủ bằng chứng" in final_answer.casefold()
            and primary_title
            and not result_text_mentions_target(
                {"text": final_answer}, target_title=primary_title, target_aliases=primary_aliases,
            )
        )
        if final_target_mismatch:
            issue("final_answer_target_mismatch", row)
        if final_facet_mismatch:
            issue("final_answer_facet_mismatch", row)
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
        "unexpected_empty_tool_results", "trajectory_observation_budget_exceeded",
        "subject_type_mismatch", "observation_target_mismatch",
        "final_answer_target_mismatch", "compare_target_contamination",
        "observation_facet_mismatch", "final_answer_facet_mismatch",
        "domain_mismatch",
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
        "empty_tool_results_total": empty_results,
        "expected_empty_tool_results": expected_empty_results,
        "unexpected_empty_tool_results": unexpected_empty_results,
        "empty_tool_results_by_task_type": _empty_breakdown(empty_by_task),
        "empty_tool_results_by_retrieval_role": _empty_breakdown(empty_by_role),
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
