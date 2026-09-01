from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Iterable

from .adapters.agent_flan import contains_agent_flan_action_syntax, contains_agent_flan_thought_target
from .adapters.vietnam_history import canonical_analysis_messages_remaining
from .builders.custom_history import (
    canonical_subject_identity,
    classify_subject,
    compare_subjects_compatible,
    is_custom_history_eligible,
    is_result_facet_relevant,
    is_result_relevant_to_target,
    result_text_mentions_target,
)
from .citations import extract_evidence_citations
from .dedup import first_user_question, normalized_question
from .preprocess import (
    IGNORE_INDEX,
    analyze_truncation,
    assistant_labeled_token_counts,
    build_canonical_sft_example,
)


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
            continue
        # Expected-empty semantics belong to the compact retrieval contract.
        # Other canonical tools legitimately return plain text or JSON objects;
        # treating those as an empty retrieval result creates false failures.
        if isinstance(payload, list):
            payloads.append(payload)
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
        if source_dataset == "vietnam_history_200k" and canonical_analysis_messages_remaining(row):
            issue("vietnam_history_analysis_message_remaining", row)
        if source_dataset == "agent_flan":
            assistant_targets = [
                str(message.get("content") or "")
                for message in row.get("messages") or []
                if isinstance(message, dict)
                and message.get("role") == "assistant"
                and not message.get("tool_calls")
            ]
            if any(contains_agent_flan_action_syntax(content) for content in assistant_targets):
                issue("agent_flan_literal_action_without_tools", row)
            if any(contains_agent_flan_thought_target(content) for content in assistant_targets):
                issue("agent_flan_thought_target", row)

        def audit_subject_record(title: str, subject_type: str, *, secondary: bool) -> dict[str, Any]:
            identity_key = "secondary_subject_identity" if secondary else "primary_subject_identity"
            identity = provenance.get(identity_key)
            if isinstance(identity, dict):
                canonical = canonical_subject_identity(identity)
                if canonical is not None:
                    return canonical
            evidence = [
                result
                for role, payload in zip(roles, payloads)
                if (role == "target_b") == secondary
                for result in payload
                if isinstance(result, dict)
            ]
            metadata: dict[str, Any] = {"subject_type": subject_type}
            for field in (
                "people", "events", "documents", "organizations", "states",
                "dynasties", "locations", "dates", "topics", "aliases",
                "alternative_names", "other_names",
            ):
                values: list[Any] = []
                for result in evidence:
                    result_metadata = result.get("metadata")
                    if not isinstance(result_metadata, dict):
                        continue
                    value = result_metadata.get(field)
                    if isinstance(value, list):
                        values.extend(value)
                    elif value not in (None, ""):
                        values.append(value)
                if values:
                    metadata[field] = list(dict.fromkeys(str(value) for value in values))
            reconstructed = {
                "title": title,
                "text": " ".join(str(result.get("text") or "") for result in evidence),
                "history_score": max(
                    (result.get("history_score") for result in evidence if isinstance(result.get("history_score"), (int, float))),
                    default=None,
                ),
                "metadata": metadata,
            }
            return canonical_subject_identity(reconstructed) or reconstructed

        def builder_confirmed_domain(*, secondary: bool) -> bool:
            key = (
                "secondary_custom_history_eligibility_signals"
                if secondary else "custom_history_eligibility_signals"
            )
            signals = provenance.get(key)
            return isinstance(signals, list) and any(
                str(signal).startswith("language:") for signal in signals
            ) and any(str(signal).startswith("history:strong") for signal in signals)

        if source_dataset.startswith("custom_history") and primary_title:
            primary_record = audit_subject_record(primary_title, subject, secondary=False)
            if classify_subject(primary_record) != subject:
                issue("subject_type_mismatch", row)
            secondary_type = str(provenance.get("secondary_subject_type") or "")
            secondary_record = audit_subject_record(secondary_title, secondary_type, secondary=True)
            if secondary_title and classify_subject(secondary_record) != secondary_type:
                issue("subject_type_mismatch", row)
            if not is_custom_history_eligible(primary_record) and not builder_confirmed_domain(secondary=False):
                issue("domain_mismatch", row)
            if secondary_title:
                if not is_custom_history_eligible(secondary_record) and not builder_confirmed_domain(secondary=True):
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
            primary_compare_record = audit_subject_record(primary_title, subject, secondary=False)
            secondary_compare_record = audit_subject_record(secondary_title, secondary_type, secondary=True)
            if not secondary_type or not compare_subjects_compatible(
                primary_compare_record,
                secondary_compare_record,
            ):
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
                    require_selected_person_text=True,
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
                    title_only = {"title": result.get("title"), "text": "", "metadata": {}}
                    assigned_title_match = is_result_relevant_to_target(
                        title_only,
                        target_title=target_title,
                        target_subject_type=target_type,
                        retrieval_role=role,
                        target_aliases=target_aliases,
                    )
                    other_title_match = is_result_relevant_to_target(
                        title_only,
                        target_title=other_title,
                        target_subject_type=other_type,
                        retrieval_role=role,
                        target_aliases=other_aliases,
                    )
                    assigned_text_match = result_text_mentions_target(
                        result, target_title=target_title, target_aliases=target_aliases,
                    )
                    other_text_match = result_text_mentions_target(
                        result, target_title=other_title, target_aliases=other_aliases,
                    )
                    contaminates = (
                        other_title_match and not assigned_title_match
                    ) or (
                        other_text_match and not assigned_text_match and not assigned_title_match
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
        "vietnam_history_analysis_message_remaining",
        "agent_flan_literal_action_without_tools", "agent_flan_thought_target",
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
    zero_supervised = preprocessing_errors = 0
    preprocessing_error_counts: Counter[str] = Counter()
    preprocessing_error_row_ids: list[str] = []
    preprocessing_error_examples: list[dict[str, str]] = []

    def record_preprocessing_error(row: dict[str, Any], stage: str, error: Exception) -> None:
        nonlocal preprocessing_errors
        preprocessing_errors += 1
        reason = f"{type(error).__name__}: {error}"
        preprocessing_error_counts[reason] += 1
        row_id = str(row.get("id") or "<missing>")
        if len(preprocessing_error_row_ids) < 50:
            preprocessing_error_row_ids.append(row_id)
        if len(preprocessing_error_examples) < 10:
            preprocessing_error_examples.append({"row_id": row_id, "stage": stage, "error": reason})

    for row in rows:
        try:
            report = analyze_truncation(tokenizer, row, max_length=max_seq_length)
        except Exception as exc:
            record_preprocessing_error(row, "analyze_truncation", exc)
            continue
        token_counts.append(int(report["total_tokens"]))
        too_long += int(report["truncated"])
        user_lost += int(report["initial_user_lost"])
        tool_calls_lost += int(report["lost_tool_call_targets"] > 0)
        assistant_lost += int(report["lost_assistant_targets"] > 0)
        final_lost += int(report["final_assistant_lost"])
        all_lost += int(report["all_assistant_supervision_lost"])
        try:
            feature = build_canonical_sft_example(tokenizer, row, max_length=max_seq_length)
        except Exception as exc:
            record_preprocessing_error(row, "build_canonical_sft_example", exc)
        else:
            zero_supervised += int(not any(label != IGNORE_INDEX for label in feature["labels"]))
    return {
        "rendered_tokens": _describe(token_counts),
        "rows_over_max_seq_length": too_long,
        "rows_initial_user_lost": user_lost,
        "rows_any_tool_call_supervision_lost": tool_calls_lost,
        "rows_any_assistant_supervision_lost": assistant_lost,
        "rows_final_assistant_supervision_lost": final_lost,
        "rows_all_assistant_supervision_lost": all_lost,
        "rows_zero_supervised_tokens": zero_supervised,
        "preprocessing_errors": preprocessing_errors,
        "preprocessing_error_counts": dict(sorted(preprocessing_error_counts.items())),
        "preprocessing_error_row_ids": preprocessing_error_row_ids,
        "preprocessing_error_examples": preprocessing_error_examples,
        "max_seq_length": max_seq_length,
    }


def central_v2_audit(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    max_seq_length: int = 4096,
) -> dict[str, Any]:
    """Behavior and labeled-token audit for the Central V2 two-source mix."""
    materialized = list(rows)
    source_counts = Counter(str(row.get("source_dataset") or "unknown") for row in materialized)
    task_counts = Counter(str(row.get("task_type") or "unknown") for row in materialized)
    rows_using_tools = 0
    first_tool = 0
    direct_first_answer = 0
    tool_calls_per_row: list[int] = []
    lengths: list[int] = []
    answerability: Counter[str] = Counter()
    prefixes: Counter[str] = Counter()
    suspicious = Counter({"Vào năm": 0, "Ý nghĩa:": 0, "Action:": 0, "Thought:": 0})
    total_labeled = tool_labeled = final_labeled = 0
    trajectory_tokens: list[int] = []
    truncation_risk = 0
    token_errors = 0

    for row in materialized:
        messages = [message for message in row.get("messages") or [] if isinstance(message, dict)]
        assistants = [message for message in messages if message.get("role") == "assistant"]
        calls = sum(len(message.get("tool_calls") or []) for message in assistants)
        tool_calls_per_row.append(calls)
        lengths.append(len(messages))
        if calls:
            rows_using_tools += 1
        if assistants and assistants[0].get("tool_calls"):
            first_tool += 1
        elif assistants and str(assistants[0].get("content") or "").strip():
            direct_first_answer += 1
        provenance = row.get("provenance") or {}
        if provenance.get("answerable") is True:
            answerability["answerable"] += 1
        elif provenance.get("is_impossible") is True or provenance.get("answerable") is False:
            answerability["unanswerable"] += 1
        for message in assistants:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            normalized_prefix = " ".join(content.split())[:48]
            prefixes[normalized_prefix] += 1
            for needle in suspicious:
                suspicious[needle] += content.count(needle)

        if tokenizer is not None:
            try:
                counts = assistant_labeled_token_counts(tokenizer, row)
                tool_labeled += counts["assistant_tool_call_labeled_tokens"]
                final_labeled += counts["assistant_final_answer_labeled_tokens"]
                total_labeled += counts["total_assistant_labeled_tokens"]
                trajectory_tokens.append(counts["trajectory_tokens"])
                truncation_risk += int(counts["trajectory_tokens"] > max_seq_length)
            except Exception:
                token_errors += 1
        else:
            # Clearly marked estimate for cheap offline inspection; the Colab
            # workflow supplies the Qwen tokenizer for exact labeled tokens.
            for message in assistants:
                content = json.dumps(message.get("tool_calls"), ensure_ascii=False) if message.get("tool_calls") else str(message.get("content") or "")
                count = len(content.split())
                total_labeled += count
                if message.get("tool_calls"):
                    tool_labeled += count
                else:
                    final_labeled += count

    row_count = len(materialized)
    return {
        "rows": row_count,
        "rows_by_source": dict(sorted(source_counts.items())),
        "rows_by_task": dict(sorted(task_counts.items())),
        "rows_using_tools": rows_using_tools,
        "total_assistant_labeled_tokens": total_labeled,
        "assistant_tool_call_labeled_tokens": tool_labeled,
        "assistant_final_answer_labeled_tokens": final_labeled,
        "tool_call_tokens_per_all_labeled_tokens": round(tool_labeled / total_labeled, 6) if total_labeled else 0.0,
        "token_metrics_exact": tokenizer is not None,
        "token_metric_errors": token_errors,
        "first_assistant_tool_call_rate": round(first_tool / row_count, 6) if row_count else 0.0,
        "direct_first_assistant_answer_rate": round(direct_first_answer / row_count, 6) if row_count else 0.0,
        "tool_calls_per_trajectory": _describe(tool_calls_per_row),
        "answerable_unanswerable": dict(sorted(answerability.items())),
        "trajectory_turns": _describe(lengths),
        "trajectory_tokens": _describe(trajectory_tokens),
        "truncation_risk_rows": truncation_risk,
        "max_seq_length": max_seq_length,
        "top_repeated_assistant_prefixes": [
            {"prefix": prefix, "count": count} for prefix, count in prefixes.most_common(20)
        ],
        "suspicious_string_frequency": dict(suspicious),
    }
