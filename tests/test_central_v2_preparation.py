import copy
import json

import pytest

from training.central.data.prepare import normalize_rows
from training.central.mixing.mix import assistant_token_share, load_mix_config, mix_v2
from training.central.normalization.validation import require_v2_trajectory
from training.central.train.config import parse_args
from training.central.train.constants import DEFAULT_RUN_NAME, DEFAULT_TRAIN_FILE
from tests.trajectory_dataset.test_central_v2 import hermes_row


def normalized():
    raw = hermes_row([
        {"from": "human", "value": "Weather in Alpha?"},
        {"from": "gpt", "value": '<tool_call>{"name":"weather","arguments":{"city":"Alpha"}}</tool_call>'},
        {"from": "tool", "value": '<tool_response>{"temperature":20}</tool_response>'},
        {"from": "gpt", "value": "The reported temperature is 20."},
    ])
    rows, rejected = normalize_rows([raw], source="hermes")
    assert not rejected
    return rows[0]


def test_v2_schema_checks_argument_types_required_fields_and_no_network_refs():
    valid = normalized()
    assert require_v2_trajectory(valid) is valid
    for args in ({"city": 42}, {}):
        row = copy.deepcopy(valid)
        next(message for message in row["messages"] if message.get("tool_calls"))["tool_calls"][0]["function"]["arguments"] = args
        with pytest.raises(ValueError):
            require_v2_trajectory(row)
    row = copy.deepcopy(valid)
    row["tools"][0]["function"]["parameters"]["$ref"] = "https://invalid.example/schema.json"
    with pytest.raises(ValueError, match="external schema"):
        require_v2_trajectory(row)


def test_source_gates_and_single_config_no_legacy_mix(tmp_path):
    row = normalized()
    row["source_dataset"] = "agent_flan"
    with pytest.raises(ValueError, match="only Hermes"):
        require_v2_trajectory(row)
    config = load_mix_config()
    assert config["ratios"] == {"hermes_function_calling": .65, "uit_viquad2_grounded": .35}
    config["ratios"]["agent_flan"] = .1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="two intended sources"):
        load_mix_config(path)
    assert DEFAULT_RUN_NAME == "central-v2"
    assert DEFAULT_TRAIN_FILE.startswith("training/central/")
    assert parse_args([]).model_id == "Qwen/Qwen3-8B"


def test_mix_is_seeded_and_rejects_duplicate_ids():
    pools = {"hermes_function_calling": [], "uit_viquad2_grounded": []}
    # Use real tiny normalizer output for both source kinds.
    for index in range(8):
        row = normalized()
        row["id"] = f"hermes-{index}"
        pools["hermes_function_calling"].append(row)
    raw = {"id": "v", "title": "Lịch sử", "context": "Triều đại Alpha tiến hành cải cách hành chính. Vua tổ chức quân đội bảo vệ lãnh thổ sau chiến tranh.",
           "question": "Triều đại nào tiến hành cải cách hành chính?", "answers": {"text": ["Triều đại Alpha"], "answer_start": [0]}, "is_impossible": False}
    rows, rejected = normalize_rows([raw], source="viquad")
    assert not rejected
    for index in range(5):
        row = copy.deepcopy(rows[0]); row["id"] = f"viquad-{index}"
        pools["uit_viquad2_grounded"].append(row)
    assert mix_v2(pools, seed=7) == mix_v2(pools, seed=7)
    pools["hermes_function_calling"].append(pools["hermes_function_calling"][0])
    with pytest.raises(ValueError, match="duplicate"):
        mix_v2(pools)


def test_assistant_token_share_uses_supervised_tokens(monkeypatch):
    import training.central.mixing.mix as mixing
    monkeypatch.setattr(mixing, "assistant_labeled_token_counts", lambda tokenizer, row: {"total_assistant_labeled_tokens": row["count"]})
    result = assistant_token_share([{"source_dataset": "hermes_function_calling", "count": 6},
                                    {"source_dataset": "uit_viquad2_grounded", "count": 4}], object())
    assert result["hermes_function_calling"]["share"] == .6
