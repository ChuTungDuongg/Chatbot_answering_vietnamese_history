from __future__ import annotations

import copy
import json

import pytest

from training.trajectory_dataset.adapters.common import AdapterError
from training.central.normalization.hermes import normalize_hermes_function_calling
from training.central.normalization.viquad import (
    answer_sentence,
    history_relevance,
    normalize_uit_viquad2,
)
from training.trajectory_dataset.audit import central_v2_audit
from training.trajectory_dataset.dedup import deduplicate
from training.trajectory_dataset.mix import mix_capacity_report, mix_sources
from training.trajectory_dataset.preprocess import build_canonical_sft_example
from training.trajectory_dataset.split import split_coverage_report, split_trajectories
from training.trajectory_dataset.validate import validate_rows, validate_trajectory


TOOLS = [{
    "type": "function",
    "function": {
        "name": "weather",
        "description": "weather fixture",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def hermes_row(conversations, *, row_id="hermes-1", tools=TOOLS):
    return {"id": row_id, "tools": json.dumps(tools), "conversations": conversations}


def test_hermes_single_tool_call_only_is_canonical_and_valid():
    row = hermes_row([
        {"from": "system", "value": "<tools>ignored because explicit tools exist</tools>"},
        {"from": "human", "value": "Weather in Hanoi?"},
        {"from": "gpt", "value": '<tool_call>{"name":"weather","arguments":{"city":"Hanoi"}}</tool_call>'},
    ])
    normalized = normalize_hermes_function_calling(
        row, index=0, source_file="func-calling-singleturn.json",
    )

    call = normalized["messages"][-1]["tool_calls"][0]
    assert call["id"] == "call_0001"
    assert call["function"]["arguments"] == {"city": "Hanoi"}
    assert normalized["provenance"]["terminal_tool_call_only"] is True
    assert validate_trajectory(normalized) == []
    assert "<tool_call>" not in json.dumps(normalized["messages"])


def test_hermes_multiple_tools_and_multi_turn_results_are_paired():
    row = hermes_row([
        {"from": "human", "value": "Compare Hanoi and Hue weather."},
        {"from": "gpt", "value": (
            '<tool_call>{"name":"weather","arguments":{"city":"Hanoi"}}</tool_call>'
            '<tool_call>{"name":"weather","arguments":{"city":"Hue"}}</tool_call>'
        )},
        {"from": "tool", "value": '<tool_response>{"temp":30}</tool_response>'},
        {"from": "tool", "value": '<tool_response>{"temp":29}</tool_response>'},
        {"from": "gpt", "value": "Hanoi is one degree warmer."},
        {"from": "human", "value": "Which result was first?"},
        {"from": "gpt", "value": "The Hanoi result was first."},
    ], row_id="multi")
    normalized = normalize_hermes_function_calling(row, index=1, source_file="func-calling.json")

    calls = normalized["messages"][2]["tool_calls"]
    results = [message for message in normalized["messages"] if message["role"] == "tool"]
    assert [call["id"] for call in calls] == ["call_0001", "call_0002"]
    assert [result["tool_call_id"] for result in results] == ["call_0001", "call_0002"]
    assert validate_trajectory(normalized) == []


@pytest.mark.parametrize(
    "row,reason",
    [
        (
            hermes_row([
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": '<tool_call>{"name":"weather","arguments":"not-json"}</tool_call>'},
            ], row_id="bad-args"),
            "malformed",
        ),
        (
            hermes_row([
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": '<tool_call>{"name":"undefined","arguments":{}}</tool_call>'},
            ], row_id="undefined"),
            "undefined tool",
        ),
        (
            hermes_row([
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "Thought: plan\nAction: weather"},
            ], row_id="thought"),
            "Thought/Action",
        ),
    ],
)
def test_hermes_rejects_malformed_undefined_and_scratchpad(row, reason):
    with pytest.raises(AdapterError, match=reason):
        normalize_hermes_function_calling(row, index=0, source_file="func-calling.json")


def test_hermes_json_mode_is_excluded_by_default():
    with pytest.raises(AdapterError, match="JSON-mode"):
        normalize_hermes_function_calling(
            hermes_row([{"from": "human", "value": "json"}, {"from": "gpt", "value": "{}"}]),
            index=0,
            source_file="json-mode-agentic.json",
        )


def test_hermes_pending_call_is_only_allowed_in_singleturn_subset():
    row = hermes_row([
        {"from": "human", "value": "Weather in Hanoi?"},
        {"from": "gpt", "value": '<tool_call>{"name":"weather","arguments":{"city":"Hanoi"}}</tool_call>'},
    ])

    with pytest.raises(AdapterError, match="where a result is required"):
        normalize_hermes_function_calling(row, index=0, source_file="func-calling.json")


def viquad_row(*, impossible=False, row_id="v1"):
    context = "Năm 938, Ngô Quyền lãnh đạo quân dân đánh bại quân Nam Hán trên sông Bạch Đằng. Chiến thắng mở ra thời kỳ tự chủ."
    answer = "Ngô Quyền"
    return {
        "id": row_id,
        "uit_id": "uit-" + row_id,
        "title": "Trận Bạch Đằng năm 938",
        "context": context,
        "question": "Ai lãnh đạo trận Bạch Đằng năm 938?",
        "answers": [] if impossible else {"text": [answer], "answer_start": [context.index(answer)]},
        "is_impossible": impossible,
        "plausible_answers": [{"text": "Ngô Quyền", "answer_start": context.index(answer)}] if impossible else [],
    }


def test_viquad_answerable_becomes_grounded_search_history_trajectory():
    raw = viquad_row()
    normalized = normalize_uit_viquad2(raw, index=0, split="train")
    assistants = [message for message in normalized["messages"] if message["role"] == "assistant"]
    source_id = normalized["provenance"]["evidence_ids"][0]

    assert assistants[0]["tool_calls"][0]["function"]["name"] == "search_history"
    assert assistants[0]["tool_calls"][0]["function"]["arguments"]["query"] == raw["question"]
    assert "Ngô Quyền" in assistants[-1]["content"]
    assert f"[{source_id}]" in assistants[-1]["content"]
    observation = json.loads(normalized["messages"][3]["content"])
    assert observation["observation_origin"] == "uit_viquad2_ground_truth_context"
    assert observation["results"][0]["chunk_id"] == source_id
    assert validate_trajectory(normalized) == []


def test_viquad_impossible_teaches_insufficient_evidence_without_false_citation():
    normalized = normalize_uit_viquad2(viquad_row(impossible=True, row_id="v2"), index=1, split="validation")
    final = normalized["messages"][-1]["content"]
    assert "không đủ" in final
    assert "[viquad_" not in final
    assert normalized["provenance"]["plausible_answers"]
    assert validate_trajectory(normalized) == []


def test_viquad_history_filter_is_conservative_and_year_alone_is_not_enough():
    accepted = history_relevance(viquad_row())
    rejected = history_relevance({
        "title": "Giải bóng đá 2024",
        "question": "Đội nào vô địch năm 2024?",
        "context": "Giải đấu thể thao diễn ra trong năm 2024 với nhiều bàn thắng.",
    })
    assert accepted.accepted is True
    assert rejected.accepted is False
    with pytest.raises(AdapterError, match="non-history"):
        normalize_uit_viquad2({
            "id": "sport",
            "title": "Giải bóng đá 2024",
            "question": "Đội nào vô địch năm 2024?",
            "context": "Đội A vô địch giải bóng đá năm 2024.",
            "answers": {"text": ["Đội A"], "answer_start": [0]},
            "is_impossible": False,
        }, index=0)


def test_answer_span_extracts_the_evidence_sentence_only():
    context = "Câu đầu không liên quan. Ngô Quyền chiến thắng trên sông Bạch Đằng năm 938. Câu cuối."
    sentence = answer_sentence(context, "Ngô Quyền", context.index("Ngô Quyền"))
    assert sentence == "Ngô Quyền chiến thắng trên sông Bạch Đằng năm 938."


def test_mix_ratio_capacity_shortfall_dedup_and_no_duplication():
    hermes = [copy.deepcopy(normalize_hermes_function_calling(
        hermes_row([
            {"from": "human", "value": f"q{i}"},
            {"from": "gpt", "value": '<tool_call>{"name":"weather","arguments":{"city":"Hanoi"}}</tool_call>'},
        ], row_id=f"h{i}"), index=i, source_file="func-calling-singleturn.json",
    )) for i in range(13)]
    viquad = [normalize_uit_viquad2(viquad_row(row_id=f"v{i}"), index=i) for i in range(7)]
    for index, row in enumerate(viquad):
        row["id"] += f"-{index}"
        row["messages"][1]["content"] += f" bản {index}"
    sources = {"hermes_function_calling": hermes, "uit_viquad2_grounded": viquad}
    ratios = {"hermes_function_calling": 0.65, "uit_viquad2_grounded": 0.35}
    mixed = mix_sources(sources, ratios, max_total=20, seed=7)
    assert len(mixed) == 20
    assert len({row["id"] for row in mixed}) == 20

    short = mix_capacity_report(
        {"hermes_function_calling": hermes[:1], "uit_viquad2_grounded": viquad},
        ratios,
        requested_total=20,
    )
    assert short["insufficient_sources"] == ["hermes_function_calling"]
    assert short["duplicates_fabricated"] is False
    duplicated = copy.deepcopy(hermes[0])
    result = deduplicate([hermes[0], duplicated])
    assert len(result.rows) == 1


def test_split_keeps_identical_normalized_questions_together():
    rows = []
    for index in range(8):
        row = normalize_uit_viquad2(viquad_row(row_id=f"split-{index}"), index=index)
        row["id"] += f"-{index}"
        row["provenance"]["source_group"] = f"group-{index}"
        if index > 1:
            row["messages"][1]["content"] += f" biến thể {index}"
        rows.append(row)
    splits = split_trajectories(rows, train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25, seed=2)
    report = split_coverage_report(splits)
    assert report["normalized_question_leakage_count"] == 0


class FakeQwenTokenizer:
    def __init__(self):
        self.enable_thinking_values = []

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, enable_thinking=True):
        assert tokenize is False
        self.enable_thinking_values.append(enable_thinking)
        prefix = "TOOLS=" + json.dumps(tools or [], sort_keys=True) + "\n"
        body = "".join(json.dumps(message, sort_keys=True, ensure_ascii=False) + "\n" for message in messages)
        return prefix + body + ("ASSISTANT\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(text.encode("utf-8"))}


def test_token_audit_and_qwen_train_inference_template_parity():
    rows = [
        normalize_hermes_function_calling(
            hermes_row([
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": '<tool_call>{"name":"weather","arguments":{"city":"Hue"}}</tool_call>'},
            ]),
            index=0,
            source_file="func-calling-singleturn.json",
        ),
        normalize_uit_viquad2(viquad_row(), index=0),
    ]
    tokenizer = FakeQwenTokenizer()
    feature = build_canonical_sft_example(tokenizer, rows[1], max_length=100_000)
    report = central_v2_audit(rows, tokenizer=tokenizer, max_seq_length=100_000)

    assert any(label != -100 for label in feature["labels"])
    assert tokenizer.enable_thinking_values and set(tokenizer.enable_thinking_values) == {False}
    assert report["token_metrics_exact"] is True
    assert report["total_assistant_labeled_tokens"] > 0
    assert report["assistant_tool_call_labeled_tokens"] > 0
    assert report["first_assistant_tool_call_rate"] == 1.0
    assert report["direct_first_assistant_answer_rate"] == 0.0
    assert report["suspicious_string_frequency"]["Thought:"] == 0


def test_strict_central_v2_validation_gates():
    valid = normalize_uit_viquad2(viquad_row(), index=0)
    bad_thought = copy.deepcopy(valid)
    bad_thought["messages"][-1]["content"] = "Thought: hidden plan"
    assert any("reasoning supervision" in error for error in validate_trajectory(bad_thought))

    bad_first = copy.deepcopy(valid)
    bad_first["messages"][2] = {"role": "assistant", "content": "Direct answer"}
    assert any("must begin" in error for error in validate_trajectory(bad_first))

    duplicate_result = validate_rows([valid, copy.deepcopy(valid)])
    assert len(duplicate_result.rejected) == 1

    bad_contract = copy.deepcopy(valid)
    bad_contract["provenance"]["chat_template_contract"]["enable_thinking"] = True
    assert any("train/inference" in error for error in validate_trajectory(bad_contract))
