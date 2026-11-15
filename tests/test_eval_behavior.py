"""Behavior metrics use captured observations and never infer historical truth."""

import pytest
from pydantic import ValidationError


def evaluate(spec=None, *, answer="Năm 938 [S1]", calls=None, capture=None, evidence=None, **run):
    from app.eval_studio.behavior import evaluate_behavior

    diagnostics = {
        "tool_calls": calls if calls is not None else [],
        "planning": {"exit_reason": "completed"},
        "capture": (
            capture
            if capture is not None
            else {
                "tool_trace": True,
                "retrieved_evidence": True,
                "final_context": True,
            }
        ),
    }
    return evaluate_behavior(
        {
            "question": "Năm nào?",
            "reference_answer": "Năm 938",
            "expected_facts": ["938"],
            "expected_behavior": spec or {},
        },
        {"answer": answer, "completion_status": "completed", **run},
        (
            evidence
            if evidence is not None
            else [
                {"chunk_id": "c1", "excerpt": "Chiến thắng năm 938", "selected": True},
            ]
        ),
        diagnostics,
    )


def call(name, fingerprint="a", valid=True, status="ok"):
    return {
        "name": name,
        "call_id": "call-1",
        "argument_fingerprint": fingerprint,
        "arguments_valid": valid,
        "status": status,
        "error_code": None,
        "duration_ms": 10,
        "result_count": 1,
    }


def test_tools_allow_alternative_groups_without_enforcing_sequence():
    result = evaluate(
        {
            "required_tools": ["search"],
            "any_tool_groups": [["lookup", "read"]],
            "allowed_tools": ["search", "read"],
            "max_tool_calls": 3,
        },
        calls=[call("read"), call("search")],
    )
    assert result["checks"]["required_tools"] is True
    assert result["checks"]["any_tool_groups"] is True
    assert result["checks"]["allowed_tools"] is True
    assert result["checks"]["max_tool_calls"] is True
    assert result["tools"]["success_rate"] == 1.0


def test_missing_trace_is_unknown_even_with_legacy_count_or_fallback_calls():
    result = evaluate(
        {
            "required_tools": ["search"],
            "forbidden_tools": ["web"],
            "max_tool_calls": 10,
            "max_repeated_calls": 0,
        },
        calls=[call("search")],
        capture={},
        tool_calls=1,
    )
    for key in ("required_tools", "forbidden_tools", "max_tool_calls", "max_repeated_calls"):
        assert result["checks"][key] is None
    assert result["tools"]["call_count"] is None
    assert result["tools"]["argument_validity_rate"] is None
    assert result["tools"]["success_rate"] is None
    assert result["evidence"]["expected_id_recall"] is None
    assert result["citation_metrics"]["citation_validity_rate"] is None
    assert result["grounding_metrics"]["unverified_year_rate"] is None


def test_captured_empty_trace_can_fail_required_tools_and_pass_zero_budget():
    result = evaluate({"required_tools": ["search"], "max_tool_calls": 0}, calls=[])
    assert result["checks"]["required_tools"] is False
    assert result["checks"]["max_tool_calls"] is True
    assert result["tools"]["call_count"] == 0
    assert result["tools"]["argument_validity_rate"] is None
    assert result["tools"]["success_rate"] is None


def test_tool_violations_and_repeated_arguments_use_name_and_fingerprint():
    result = evaluate(
        {
            "required_tools": ["missing"],
            "forbidden_tools": ["web"],
            "allowed_tools": ["search"],
            "max_tool_calls": 2,
            "max_repeated_calls": 0,
        },
        calls=[call("search"), call("search", valid=False, status="error"), call("web")],
    )
    assert result["checks"]["required_tools"] is False
    assert result["checks"]["forbidden_tools"] is False
    assert result["checks"]["allowed_tools"] is False
    assert result["checks"]["max_tool_calls"] is False
    assert result["checks"]["max_repeated_calls"] is False
    assert result["tools"]["repeated_call_count"] == 1
    assert result["tools"]["repeated_call_rate"] == pytest.approx(1 / 3)
    assert result["tools"]["argument_validity_rate"] == pytest.approx(2 / 3)
    assert result["tools"]["success_rate"] == pytest.approx(2 / 3)


def test_unknown_argument_metadata_does_not_mean_invalid_or_repeated():
    result = evaluate({"max_repeated_calls": 0}, calls=[call("search", None, None, None)])
    assert result["tools"]["argument_validity_rate"] is None
    assert result["tools"]["success_rate"] is None
    assert result["tools"]["repeated_call_rate"] is None
    assert result["checks"]["max_repeated_calls"] is None


def test_policy_rejected_calls_count_as_observed_failures():
    result = evaluate(calls=[call("search"), call("web", status="rejected")])
    assert result["tools"]["success_rate"] == 0.5
    assert result["tools"]["success_observed"] == 2


def test_failed_run_with_no_answer_does_not_get_zero_lexical_scores():
    result = evaluate(answer="", completion_status="failed")
    assert result["answer_metrics"]["token_f1"] is None
    assert result["grounding_metrics"]["required_fact_phrase_recall"] is None


def test_blank_reference_or_planning_reason_is_unknown():
    from app.eval_studio.behavior import evaluate_behavior

    result = evaluate_behavior(
        {"reference_answer": "   ", "expected_behavior": {"expected_exit_reasons": ["completed"]}},
        {"answer": "938", "completion_status": "completed"},
        [],
        {"capture": {"tool_trace": True}, "tool_calls": [], "planning": {"exit_reason": ""}},
    )
    assert result["checks"]["expected_exit_reason"] is None
    assert result["answer_metrics"]["token_f1"] is None


def test_expected_evidence_can_identify_source_or_chunk():
    result = evaluate(
        {"expected_evidence_ids": ["source-a", "c1"]},
        evidence=[{"chunk_id": "c1", "source_id": "source-a", "excerpt": "938", "selected": True}],
    )
    assert result["checks"]["evidence_ids"] is True
    assert result["evidence"]["expected_id_recall"] == 1.0


def test_capped_evidence_never_produces_false_missing_ids_or_grounding():
    result = evaluate(
        {"expected_evidence_ids": ["c1"], "evidence_terms": ["938"], "require_citations": True},
        evidence=[{"chunk_id": "c1", "excerpt": "938 " * 100_000, "selected": True}],
    )
    assert result["checks"]["evidence_ids"] is None
    assert result["checks"]["evidence_terms"] is None
    assert result["checks"]["citations"] is None
    assert result["grounding_metrics"]["unverified_year_rate"] is None
    assert result["limits"]["evidence_metrics_skipped"] is True


def test_evidence_and_grounding_use_distinct_retrieved_and_final_context():
    result = evaluate(
        {
            "expected_evidence_ids": ["c1", "c2"],
            "evidence_terms": ["Bạch Đằng"],
            "expected_exit_reasons": ["completed"],
            "require_citations": True,
        },
        evidence=[
            {"chunk_id": "c1", "excerpt": "Năm 938", "selected": False},
            {"chunk_id": "c2", "excerpt": "Bạch Đằng năm 1288", "selected": True},
        ],
    )
    assert result["evidence"]["expected_id_recall"] == 1.0
    assert result["evidence"]["term_recall"] == 1.0
    assert result["checks"]["evidence_ids"] is True
    assert result["checks"]["evidence_terms"] is True
    assert result["checks"]["expected_exit_reason"] is True
    assert result["checks"]["citations"] is True
    assert result["grounding_metrics"]["unverified_years"] == ["938"]


def test_citation_mapping_uses_final_context_order_and_rejects_invalid_alias():
    result = evaluate(
        {"require_citations": True},
        answer="Năm 938 [S1] [missing]",
        evidence=[
            {"chunk_id": "c0", "excerpt": "Năm 1945", "selected": False},
            {"chunk_id": "c1", "excerpt": "Năm 938", "selected": True},
        ],
    )
    assert result["citation_metrics"]["resolved_source_ids"] == ["c1"]
    assert result["citation_metrics"]["invalid_citation_ids"] == ["missing"]
    assert result["citation_metrics"]["citation_validity_rate"] == 0.5
    assert result["checks"]["citations"] is False


@pytest.mark.parametrize(
    "spec,answer,format_ok,keys_ok",
    [
        ({"output_format": "json", "required_json_keys": ["year"]}, '{"year":938}', True, True),
        ({"output_format": "json", "required_json_keys": ["year"]}, '{"text":"938"}', True, False),
        ({"output_format": "json"}, '```json\n{"year":938}\n```', False, None),
        ({"output_format": "markdown"}, "# Năm 938\n\n- Bạch Đằng", True, None),
        ({"output_format": "text"}, "", False, None),
    ],
)
def test_output_contract_checks_shape(spec, answer, format_ok, keys_ok):
    result = evaluate(spec, answer=answer)
    assert result["checks"]["output_format"] is format_ok
    assert result["checks"]["required_json_keys"] is keys_ok


def test_lexical_metrics_are_labeled_and_semantic_claims_require_human():
    result = evaluate({"answerable": False, "in_domain": False}, answer="Không đủ thông tin")
    assert result["schema_version"] == "1.2"
    assert result["metric_version"]
    assert result["grounding_metrics"]["insufficient_answer_behavior"] == 1.0
    assert result["grounding_metrics"]["ood_refusal_behavior"] == 1.0
    assert result["semantic_checks"]["historical_correctness"] is None
    assert "historical_correctness" in result["human_required"]
    assert "heuristic" in result["metric_labels"]["answer_metrics"]


def test_oversized_lexical_inputs_are_unknown_and_limits_are_exported():
    result = evaluate(answer="x " * 100_000)
    assert result["answer_metrics"]["token_f1"] is None
    assert result["answer_metrics"]["rouge_l_f1"] is None
    assert result["limits"]["answer_metrics_skipped"] is True
    assert result["limits"]["max_metric_tokens"] <= 1024


@pytest.mark.parametrize(
    "spec",
    [
        {"sequence": ["search"]},
        {"max_tool_calls": -1},
        {"max_tool_calls": True},
        {"required_tools": ["x" * 101]},
        {"required_tools": ["search"] * 101},
        {"any_tool_groups": [[]]},
        {"output_format": "xml"},
        {"answerable": "yes"},
    ],
)
def test_behavior_spec_rejects_unbounded_or_unsupported_contract(spec):
    from app.eval_studio.behavior import BehaviorSpec

    with pytest.raises(ValidationError):
        BehaviorSpec.model_validate(spec)
