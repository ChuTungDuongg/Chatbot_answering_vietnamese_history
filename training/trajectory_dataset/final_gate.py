from __future__ import annotations

from typing import Any

from .audit import audit_rows, tokenizer_audit
from .split import TrajectorySplits, split_coverage_report
from .validate import validate_rows


def _reason_count(rejected: list[dict[str, Any]], fragments: tuple[str, ...]) -> int:
    return sum(
        any(fragment in str(item.get("reason") or "").casefold() for fragment in fragments)
        for item in rejected
    )


def final_dataset_gate(
    splits: TrajectorySplits,
    *,
    tokenizer: Any | None,
    max_seq_length: int = 4096,
) -> dict[str, Any]:
    """Evaluate the deterministic, all-split GO-TRAIN contract.

    Token safety is deliberately not guessed: a fully passing gate requires a
    tokenizer audit using the exact canonical preprocessing path.
    """
    split_rows = {
        name: list(getattr(splits, name))
        for name in ("train", "validation", "test")
    }
    all_rows = [row for name in ("train", "validation", "test") for row in split_rows[name]]
    validation = {name: validate_rows(rows) for name, rows in split_rows.items()}
    rejected = [item for result in validation.values() for item in result.rejected]
    audit = audit_rows(all_rows, strict_custom=True)
    issues = audit.get("issues") or {}
    coverage = split_coverage_report(splits)
    token_report = (
        tokenizer_audit(all_rows, tokenizer, max_seq_length=max_seq_length)
        if tokenizer is not None else None
    )
    token_failures = None if token_report is None else sum(
        int(token_report.get(key) or 0)
        for key in (
            "rows_over_max_seq_length",
            "rows_initial_user_lost",
            "rows_any_tool_call_supervision_lost",
            "rows_any_assistant_supervision_lost",
            "rows_final_assistant_supervision_lost",
            "rows_all_assistant_supervision_lost",
            "rows_zero_supervised_tokens",
            "preprocessing_errors",
        )
    )
    counts: dict[str, int | None] = {
        "vietnam_history_analysis_messages_remaining": int(
            issues.get("vietnam_history_analysis_message_remaining", 0)
        ),
        "agent_flan_literal_action_without_tools": int(
            issues.get("agent_flan_literal_action_without_tools", 0)
        ),
        "agent_flan_thought_targets": int(issues.get("agent_flan_thought_target", 0)),
        "custom_subject_type_mismatch": int(issues.get("subject_type_mismatch", 0)),
        "compare_type_mismatch": int(issues.get("compare_type_mismatch", 0)),
        "unexpected_empty_tool_results": int(audit.get("unexpected_empty_tool_results", 0)),
        "canonical_validation_rejected": len(rejected),
        "tool_call_linkage_errors": _reason_count(rejected, (
            "tool_call_id", "tool calls without results", "pending tool results",
            "undefined tool", "tool result references", "does not match",
        )),
        "custom_citation_errors": _reason_count(rejected, (
            "evidence_ids are absent", "unknown evidence ids", "evidence citation",
        )) + int(issues.get("grounded_answer_invalid_observed_citations", 0)),
        "source_group_leakage": int(coverage["source_group_leakage_count"]),
        "token_supervision_failures": token_failures,
    }
    gates = {
        "analysis_leakage": counts["vietnam_history_analysis_messages_remaining"] == 0,
        "agent_action_without_tools": counts["agent_flan_literal_action_without_tools"] == 0,
        "agent_thought_targets": counts["agent_flan_thought_targets"] == 0,
        "subject_type": counts["custom_subject_type_mismatch"] == 0,
        "compare_type": counts["compare_type_mismatch"] == 0,
        "unexpected_empty_results": counts["unexpected_empty_tool_results"] == 0,
        "canonical_validation": counts["canonical_validation_rejected"] == 0,
        "tool_linkage": counts["tool_call_linkage_errors"] == 0,
        "citations": counts["custom_citation_errors"] == 0,
        "group_leakage": counts["source_group_leakage"] == 0,
        "token_safety": token_failures == 0 if token_failures is not None else False,
    }
    return {
        "valid": all(gates.values()),
        "token_safety_evaluated": token_report is not None,
        "gates": gates,
        "counts": counts,
        "split_validation_rejected": {
            name: len(result.rejected) for name, result in validation.items()
        },
        "coverage": coverage,
        "audit": audit,
        "tokenizer": token_report,
    }
