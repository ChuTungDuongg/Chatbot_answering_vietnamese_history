from __future__ import annotations

import json
import shutil

import pytest

from app.agents.model_registry import CENTRAL_BASE_MODEL_ID, SHARED_BASE_MODEL_ID, registry_manifest
from app.artifact_contract import (
    build_artifact_lock,
    diff_artifact_locks,
    inference_config_payload,
    manifest_payload,
    validate_artifact_lock,
    write_artifact_lock,
)


def _artifact_root(tmp_path, *, central_adapter_path: str | None = None):
    root = tmp_path / "deployment"
    for role in ("research", "evidence", "history"):
        adapter = root / "adapters" / role
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": SHARED_BASE_MODEL_ID}),
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(f"{role}-weights".encode("ascii"))
    # Keep an unreferenced V1 directory in base-only fixtures. The V2 contract is
    # driven exclusively by explicit config, so its bytes must not be loaded or
    # included in the deployment identity.
    central = root / (central_adapter_path or "adapters/central")
    central.mkdir(parents=True)
    (central / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": CENTRAL_BASE_MODEL_ID}), encoding="utf-8",
    )
    (central / "adapter_model.safetensors").write_bytes(b"central-weights")
    (root / "corpus").mkdir()
    (root / "corpus" / "vn_history_rag_chunks_enriched.jsonl").write_text(
        '{"chunk_id":"c1"}\n{"chunk_id":"c2"}\n',
        encoding="utf-8",
    )
    (root / "retrieval" / "faiss").mkdir(parents=True)
    (root / "retrieval" / "faiss" / "chunks.index").write_bytes(b"fake-index")
    (root / "retrieval" / "faiss" / "manifest.json").write_text('{"count":2}', encoding="utf-8")
    (root / "retrieval" / "bm25s_index").mkdir()
    (root / "retrieval" / "bm25s_index" / "phase9_manifest.json").write_text('{"count":2}', encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / "inference_config.json").write_text(
        json.dumps(inference_config_payload(central_adapter_path=central_adapter_path), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "config" / "model_registry.json").write_text(
        json.dumps(registry_manifest(central_adapter_path=central_adapter_path), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    provisional = build_artifact_lock(root)
    (root / "manifest.json").write_text(
        json.dumps(
            manifest_payload(
                corpus_count=2,
                deployment_id=provisional["deployment_id"],
                central_adapter_path=central_adapter_path,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_artifact_lock(root)
    return root


def test_artifact_lock_is_deterministic(tmp_path):
    root = _artifact_root(tmp_path)

    first = build_artifact_lock(root)
    second = build_artifact_lock(root)

    assert first == second
    assert first["deployment_id"].startswith("qwen3-")


def test_adapter_base_mismatch_fails(tmp_path):
    root = _artifact_root(tmp_path)
    (root / "adapters" / "evidence" / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="adapter/base mismatch"):
        build_artifact_lock(root)


def test_artifact_hash_mismatch_fails(tmp_path):
    root = _artifact_root(tmp_path)
    (root / "adapters" / "history" / "adapter_model.safetensors").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="artifact_lock"):
        validate_artifact_lock(root)


def test_central_adapter_byte_change_reports_exact_stale_field(tmp_path):
    root = _artifact_root(tmp_path, central_adapter_path="adapters/central-v2")
    (root / "adapters" / "central-v2" / "adapter_model.safetensors").write_bytes(b"changed-central")

    with pytest.raises(RuntimeError) as exc_info:
        validate_artifact_lock(root)

    assert "central.adapter_model_sha256" in str(exc_info.value)
    assert "lock=" in str(exc_info.value) and "actual=" in str(exc_info.value)


def test_base_only_lock_ignores_retained_unconfigured_v1_adapter_bytes(tmp_path):
    root = _artifact_root(tmp_path)
    before = validate_artifact_lock(root)
    (root / "adapters" / "central" / "adapter_model.safetensors").write_bytes(b"obsolete-v1-changed")

    after = validate_artifact_lock(root)

    assert before == after
    assert after["central_adapter_present"] is False
    assert after["central"] is None


def test_future_configured_central_v2_adapter_is_valid(tmp_path):
    root = _artifact_root(tmp_path, central_adapter_path="adapters/central-v2")

    lock = validate_artifact_lock(root)

    assert lock["central_adapter_present"] is True
    assert lock["central"]["path"] == "adapters/central-v2"


def test_config_consistency_validation_fails(tmp_path):
    root = _artifact_root(tmp_path)
    config_path = root / "config" / "inference_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["generation"]["research"]["max_new_tokens"] = 123
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="inference_config_sha256"):
        validate_artifact_lock(root)


def test_model_registry_change_reports_exact_stale_field(tmp_path):
    root = _artifact_root(tmp_path)
    registry = root / "config" / "model_registry.json"
    registry.write_text(registry.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="model_registry_sha256"):
        validate_artifact_lock(root)


def test_lock_diff_is_bounded_and_field_level():
    differences = diff_artifact_locks(
        {"central": {"adapter_model_sha256": "old"}, "same": 1},
        {"central": {"adapter_model_sha256": "new"}, "same": 1},
    )

    assert differences == ['- central.adapter_model_sha256: lock="old" actual="new"']


def test_manifest_deployment_id_must_match_lock(tmp_path):
    root = _artifact_root(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deployment_id"] = "qwen3-stale-manifest"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest deployment_id.*qwen3-stale-manifest"):
        validate_artifact_lock(root)


def test_manifest_cannot_retain_a_stale_central_v1_reference(tmp_path):
    root = _artifact_root(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["central"]["adapter_path"] = "adapters/central"
    manifest["central"]["adapter_configured"] = True
    manifest["central"]["adapter_source"] = "peft"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest Central adapter contract"):
        validate_artifact_lock(root)


def test_central_only_artifact_lock_does_not_require_three_role_adapters(tmp_path):
    root = _artifact_root(tmp_path)
    for role in ("research", "evidence", "history"):
        shutil.rmtree(root / "adapters" / role)
    provisional = build_artifact_lock(root)
    (root / "manifest.json").write_text(
        json.dumps(manifest_payload(corpus_count=2, deployment_id=provisional["deployment_id"])),
        encoding="utf-8",
    )
    write_artifact_lock(root)

    lock = validate_artifact_lock(root)

    assert lock["roles"] == {}
    assert lock["central"] is None
    assert lock["central_adapter_present"] is False
