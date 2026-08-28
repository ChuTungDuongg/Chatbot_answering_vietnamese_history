from __future__ import annotations

import json

import pytest

from app.agents.model_registry import SHARED_BASE_MODEL_ID
from app.artifact_contract import (
    build_artifact_lock,
    inference_config_payload,
    manifest_payload,
    validate_artifact_lock,
    write_artifact_lock,
)


def _artifact_root(tmp_path):
    root = tmp_path / "deployment"
    for role in ("research", "evidence", "history"):
        adapter = root / "adapters" / role
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": SHARED_BASE_MODEL_ID}),
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(f"{role}-weights".encode("ascii"))
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
        json.dumps(inference_config_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "config" / "model_registry.json").write_text(
        json.dumps(__import__("app.agents.model_registry", fromlist=["registry_manifest"]).registry_manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    provisional = build_artifact_lock(root)
    (root / "manifest.json").write_text(
        json.dumps(manifest_payload(corpus_count=2, deployment_id=provisional["deployment_id"]), ensure_ascii=False, indent=2),
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


def test_config_consistency_validation_fails(tmp_path):
    root = _artifact_root(tmp_path)
    config_path = root / "config" / "inference_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["generation"]["research"]["max_new_tokens"] = 123
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact_lock"):
        validate_artifact_lock(root)
