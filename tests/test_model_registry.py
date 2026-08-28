from __future__ import annotations

import json

import pytest

from app.agents.model_runtime import VLLMOpenAIBackend
from app.agents.model_registry import (
    LEGACY_HISTORY_BASE_MODEL_ID,
    ROLE_MODELS,
    SHARED_BASE_MODEL_ID,
    registry_manifest,
    validate_role_adapter,
)
from training.history_answerer.merge_adapter import merge_lora_adapter


def test_all_active_roles_use_one_qwen3_base_and_unique_adapters():
    assert set(ROLE_MODELS) == {"research", "evidence", "history"}
    assert {item.expected_base_model_id for item in ROLE_MODELS.values()} == {SHARED_BASE_MODEL_ID}
    assert len({item.model_name for item in ROLE_MODELS.values()}) == 3
    assert len({item.adapter_path for item in ROLE_MODELS.values()}) == 3
    manifest = registry_manifest()
    assert manifest["legacy_models"]["qwen25_history"]["legacy_only"] is True


def test_adapter_base_mismatch_fails_early(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": LEGACY_HISTORY_BASE_MODEL_ID}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="adapter/base mismatch"):
        validate_role_adapter("history", adapter)


def test_matching_adapter_metadata_is_accepted(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": SHARED_BASE_MODEL_ID}), encoding="utf-8"
    )
    assert validate_role_adapter("research", adapter) == SHARED_BASE_MODEL_ID


def test_merge_rejects_qwen25_adapter_for_qwen3_before_loading_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": LEGACY_HISTORY_BASE_MODEL_ID}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cannot merge adapter"):
        merge_lora_adapter(
            model_id=SHARED_BASE_MODEL_ID,
            adapter=adapter,
            output_dir=tmp_path / "merged",
        )


def test_vllm_openai_payload_routes_each_role_to_its_lora_name():
    backend = VLLMOpenAIBackend.__new__(VLLMOpenAIBackend)
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"choices": [{"message": {"content": "ok"}}]}

    backend._post = fake_post
    for role in ("research", "evidence", "history"):
        assert backend.generate_text(
            adapter=role,
            messages=[{"role": "user", "content": "fixture"}],
        ) == "ok"

    assert [path for path, _ in calls] == ["/chat/completions"] * 3
    assert [payload["model"] for _, payload in calls] == [
        ROLE_MODELS[role].model_name for role in ("research", "evidence", "history")
    ]
