from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.agents.common.model_runtime import VLLMOpenAIBackend
from app.agents.common.model_registry import (
    CENTRAL_BASE_MODEL_ID,
    CENTRAL_MODEL,
    ROLE_MODELS,
    SHARED_BASE_MODEL_ID,
    registry_manifest,
    validate_role_adapter,
    validate_central_adapter,
)
from training.history_answerer.merge_adapter import merge_lora_adapter


WRONG_BASE_MODEL_ID = "Example/NonCanonical-History-Base"


def test_all_active_roles_use_one_qwen3_base_and_unique_adapters():
    assert set(ROLE_MODELS) == {"research", "evidence", "history"}
    assert {item.expected_base_model_id for item in ROLE_MODELS.values()} == {SHARED_BASE_MODEL_ID}
    assert len({item.model_name for item in ROLE_MODELS.values()}) == 3
    assert len({item.adapter_path for item in ROLE_MODELS.values()}) == 3
    manifest = registry_manifest()
    assert manifest["legacy_models"] == {}
    assert manifest["central"]["expected_base_model_id"] == CENTRAL_BASE_MODEL_ID
    assert CENTRAL_MODEL.adapter_path is None
    assert manifest["central"]["adapter_path"] is None
    assert manifest["central"]["adapter_configured"] is False
    assert manifest["central"]["adapter_source"] == "none"


def test_active_settings_reject_legacy_merged_backend():
    with pytest.raises(ValidationError):
        Settings(llm_backend="legacy-merged")


def test_adapter_base_mismatch_fails_early(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": WRONG_BASE_MODEL_ID}), encoding="utf-8"
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


def test_central_adapter_accepts_8b_and_rejects_4b_role_base(tmp_path):
    adapter = tmp_path / "central"
    adapter.mkdir()
    config = adapter / "adapter_config.json"
    config.write_text(json.dumps({"base_model_name_or_path": CENTRAL_BASE_MODEL_ID}), encoding="utf-8")
    assert validate_central_adapter(adapter) == CENTRAL_BASE_MODEL_ID

    config.write_text(json.dumps({"base_model_name_or_path": SHARED_BASE_MODEL_ID}), encoding="utf-8")
    with pytest.raises(ValueError, match="central adapter/base mismatch"):
        validate_central_adapter(adapter)


def test_central_only_settings_do_not_require_any_4b_role_adapter(tmp_path):
    root = tmp_path / "artifacts"
    settings = Settings(
        app_mode="full",
        artifact_root=root,
        enable_hybrid_mode=False,
        enable_three_llm_mode=False,
        enable_central_mode=True,
        central_agent_adapter_path=root / "adapters" / "central",
        research_agent_adapter_path=None,
        evidence_agent_adapter_path=None,
        history_agent_adapter_path=None,
    )

    required = settings.required_full_paths()

    assert root / "adapters" / "central" in required
    assert all(root / "adapters" / role not in required for role in ("research", "evidence", "history"))


def test_central_base_only_settings_require_no_adapter_and_blank_env_is_none(tmp_path):
    root = tmp_path / "artifacts"
    settings = Settings(
        app_mode="full",
        artifact_root=root,
        enable_hybrid_mode=False,
        enable_three_llm_mode=False,
        enable_central_mode=True,
        central_agent_adapter_path="   ",
    )

    assert settings.central_adapter_path is None
    assert settings.required_full_paths() == settings.required_retrieval_paths()


def test_registry_supports_future_optional_central_v2_adapter():
    manifest = registry_manifest(central_adapter_path="adapters/central-v2")

    assert manifest["central"]["adapter_path"] == "adapters/central-v2"
    assert manifest["central"]["adapter_configured"] is True
    assert manifest["central"]["adapter_source"] == "peft"


def test_merge_rejects_qwen25_adapter_for_qwen3_before_loading_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": WRONG_BASE_MODEL_ID}), encoding="utf-8"
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
