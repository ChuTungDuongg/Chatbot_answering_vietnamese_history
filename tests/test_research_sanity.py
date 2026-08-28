from __future__ import annotations

import json

import pytest

from app.agents.policy_schema import ResearchPolicyState
from training.common.jsonl import read_jsonl
from training.research_agent.evaluate import evaluate_rows
from training.research_agent.sanity import (
    SANITY_CATEGORIES,
    build_sanity_suite,
    evaluate_sanity_predictions,
    inference_messages,
)


@pytest.fixture(scope="module")
def sanity_suite():
    rows = read_jsonl("datasets/research_agent/history_trajectories.jsonl")
    return build_sanity_suite(rows)


def _case(suite, category):
    return next(row for row in suite if row["sanity_category"] == category)


def test_sanity_categories_remain_separate_and_use_expected_sample_counts(sanity_suite):
    counts = {
        category: sum(row["sanity_category"] == category for row in sanity_suite)
        for category in SANITY_CATEGORIES
    }
    assert counts == {
        "no_tool": 5,
        "factual_history_search": 5,
        "conversational_prefix_history_search": 5,
        "search_to_inspect": 20,
        "inspect_to_finish": 20,
    }


def test_step2_replay_preserves_real_canonical_runtime_payload(sanity_suite):
    row = _case(sanity_suite, "search_to_inspect")
    state = ResearchPolicyState.model_validate(row["training_prompt"])
    gold_ids = row["training_target"]["arguments"]["ids"]

    assert row["source_dataset"] == "vn_history_phase6"
    assert row["synthetic"] is False
    assert state.step == 2
    assert len(state.observations) == 1
    assert state.observations[0].tool == "search_history"
    assert state.observations[0].result_count is not None
    assert state.observations[0].evidence_ids
    assert state.evidence_ids == state.observations[0].evidence_ids
    assert set(gold_ids).issubset(set(state.evidence_ids))
    assert json.loads(inference_messages(row)[1]["content"]) == row["training_prompt"]


def test_sanity_report_keeps_policy_and_state_categories_separate(sanity_suite):
    predictions = [{"prediction": row["training_target"]} for row in sanity_suite]
    report = evaluate_sanity_predictions(predictions, sanity_suite)

    assert set(report["by_sanity_category"]) == set(SANITY_CATEGORIES)
    assert all(
        report["by_sanity_category"][category]["action_accuracy"] == 1.0
        for category in SANITY_CATEGORIES
    )
    step2 = report["by_sanity_category"]["search_to_inspect"]
    assert step2["step2_action_tool_transition_accuracy"] == 1.0
    assert step2["step2_evidence_id_exact_match"] == 1.0
    assert step2["step2_evidence_id_precision"] == 1.0
    assert step2["step2_evidence_id_recall"] == 1.0


def test_repeated_search_is_only_a_transition_failure(sanity_suite):
    gold = _case(sanity_suite, "search_to_inspect")
    prediction = {
        "prediction": {
            "action": "tool",
            "tool_name": "search_history",
            "arguments": {"query": gold["training_prompt"]["question"], "top_k": 8},
        }
    }
    report = evaluate_rows([prediction], [gold])

    assert report["step2_action_tool_transition_accuracy"] == 0.0
    assert report["step2_evidence_id_scored_rows"] == 0
    assert report["step2_evidence_id_exact_match"] is None
    assert report["step2_evidence_id_precision"] is None
    assert report["step2_evidence_id_recall"] is None


def test_wrong_ids_do_not_turn_correct_transition_into_action_failure(sanity_suite):
    gold = _case(sanity_suite, "search_to_inspect")
    prediction = {
        "prediction": {
            "action": "tool",
            "tool_name": "inspect_evidence",
            "arguments": {"ids": ["wrong-evidence-id"]},
        }
    }
    report = evaluate_rows([prediction], [gold])

    assert report["step2_action_tool_transition_accuracy"] == 1.0
    assert report["step2_evidence_id_scored_rows"] == 1
    assert report["step2_evidence_id_exact_match"] == 0.0
    assert report["step2_evidence_id_precision"] == 0.0
    assert report["step2_evidence_id_recall"] == 0.0


def test_extra_id_lowers_precision_but_preserves_recall_and_transition(sanity_suite):
    gold = _case(sanity_suite, "search_to_inspect")
    gold_ids = list(gold["training_target"]["arguments"]["ids"])
    prediction = {
        "prediction": {
            "action": "tool",
            "tool_name": "inspect_evidence",
            "arguments": {"ids": [*gold_ids, "extra-evidence-id"]},
        }
    }
    report = evaluate_rows([prediction], [gold])

    assert report["step2_action_tool_transition_accuracy"] == 1.0
    assert report["step2_evidence_id_exact_match"] == 0.0
    assert report["step2_evidence_id_precision"] == pytest.approx(
        len(set(gold_ids)) / (len(set(gold_ids)) + 1)
    )
    assert report["step2_evidence_id_recall"] == 1.0
