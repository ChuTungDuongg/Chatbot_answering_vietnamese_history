from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.agents.common.model_registry import (
    CENTRAL_BASE_MODEL_ID,
    CENTRAL_MODEL,
    ROLE_MODELS,
    SHARED_BASE_MODEL_ID,
    registry_manifest,
    validate_central_adapter,
    validate_role_adapter,
)


LOCK_FILENAME = "artifact_lock.json"
CORPUS_RELATIVE_PATH = "corpus/vn_history_rag_chunks_enriched.jsonl"
FAISS_RELATIVE_PATH = "retrieval/faiss/chunks.index"
FAISS_MANIFEST_RELATIVE_PATH = "retrieval/faiss/manifest.json"
BM25_MANIFEST_RELATIVE_PATH = "retrieval/bm25s_index/phase9_manifest.json"
INFERENCE_CONFIG_RELATIVE_PATH = "config/inference_config.json"
MODEL_REGISTRY_RELATIVE_PATH = "config/model_registry.json"
DEPLOYMENT_ID_PREFIX = "qwen3"
_MISSING = object()


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def count_jsonl(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def faiss_metadata(path: str | Path) -> dict[str, int | None]:
    try:
        import faiss

        index = faiss.read_index(str(path))
        return {"ntotal": int(index.ntotal), "dimension": int(index.d)}
    except Exception:
        return {"ntotal": None, "dimension": None}


def bm25_count(manifest_path: str | Path) -> int | None:
    try:
        value = load_json(manifest_path).get("count")
        return int(value) if value is not None else None
    except Exception:
        return None


def inference_config_payload(*, central_adapter_path: str | None = None) -> dict[str, Any]:
    registry = registry_manifest(central_adapter_path=central_adapter_path)
    central = registry["central"]
    return {
        "llm": {
            "backend": "transformers",
            "shared_base_model_id": SHARED_BASE_MODEL_ID,
            "tokenizer_model_id": SHARED_BASE_MODEL_ID,
            "role_models": {role: spec.model_name for role, spec in ROLE_MODELS.items()},
            "central": {
                "model_id": CENTRAL_BASE_MODEL_ID,
                "adapter_path": central["adapter_path"],
                "adapter_configured": central["adapter_configured"],
                "adapter_source": central["adapter_source"],
            },
            "vllm_base_url": "http://127.0.0.1:8001/v1",
        },
        "retrieval": {
            "embedding_model_id": "intfloat/multilingual-e5-base",
            "reranker_model_id": "BAAI/bge-reranker-v2-m3",
        },
        "generation": {role: spec.generation for role, spec in ROLE_MODELS.items()},
        "central_generation": CENTRAL_MODEL.generation,
    }


def manifest_payload(
    *,
    corpus_count: int,
    deployment_id: str | None = None,
    central_adapter_path: str | None = None,
) -> dict[str, Any]:
    payload = {
        **registry_manifest(central_adapter_path=central_adapter_path),
        "corpus": {"path": CORPUS_RELATIVE_PATH, "count": corpus_count},
        "retrieval": {
            "faiss": "retrieval/faiss",
            "bm25s": "retrieval/bm25s_index",
        },
        "base_weights_bundled": False,
        "legacy_history_model": None,
    }
    if deployment_id is not None:
        payload["deployment_id"] = deployment_id
    return payload


def _lock_payload_without_deployment_id(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    roles: dict[str, Any] = {}
    for role, spec in ROLE_MODELS.items():
        adapter_dir = root_path / spec.adapter_path
        if not adapter_dir.exists():
            continue
        validate_role_adapter(role, adapter_dir)
        config_path = adapter_dir / "adapter_config.json"
        model_path = adapter_dir / "adapter_model.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing {role} adapter weights: {model_path}")
        config = load_json(config_path)
        roles[role] = {
            "path": spec.adapter_path,
            "base_model_name_or_path": str(config.get("base_model_name_or_path") or ""),
            "adapter_config_sha256": sha256_file(config_path),
            "adapter_model_sha256": sha256_file(model_path),
            "adapter_model_size": model_path.stat().st_size,
        }

    registry = load_json(root_path / MODEL_REGISTRY_RELATIVE_PATH)
    central_config = registry.get("central") if isinstance(registry.get("central"), dict) else {}
    central_relative_path = str(central_config.get("adapter_path") or "").strip() or None
    central = None
    if central_relative_path is not None:
        central_dir = root_path / central_relative_path
        if not central_dir.is_dir():
            raise FileNotFoundError(f"Configured Central adapter directory is missing: {central_dir}")
        validate_central_adapter(central_dir)
        central_config_path = central_dir / "adapter_config.json"
        central_model_path = central_dir / "adapter_model.safetensors"
        if not central_model_path.is_file():
            raise FileNotFoundError(f"Missing central adapter weights: {central_model_path}")
        adapter_config = load_json(central_config_path)
        central = {
            "path": central_relative_path,
            "base_model_name_or_path": str(adapter_config.get("base_model_name_or_path") or ""),
            "adapter_config_sha256": sha256_file(central_config_path),
            "adapter_model_sha256": sha256_file(central_model_path),
            "adapter_model_size": central_model_path.stat().st_size,
        }
    corpus_path = root_path / CORPUS_RELATIVE_PATH
    faiss_path = root_path / FAISS_RELATIVE_PATH
    bm25_manifest_path = root_path / BM25_MANIFEST_RELATIVE_PATH
    inference_config_path = root_path / INFERENCE_CONFIG_RELATIVE_PATH
    model_registry_path = root_path / MODEL_REGISTRY_RELATIVE_PATH
    faiss_info = faiss_metadata(faiss_path)
    return {
        "schema_version": 2,
        "deployment_id_derivation": (
            "sha256 over canonical artifact_lock payload excluding deployment_id, "
            "then prefix qwen3- and truncate to 16 hex chars"
        ),
        "shared_base_model_id": SHARED_BASE_MODEL_ID,
        "roles": roles,
        "central_base_model_id": CENTRAL_BASE_MODEL_ID,
        "central_adapter_present": central is not None,
        "central": central,
        "corpus": {
            "path": CORPUS_RELATIVE_PATH,
            "sha256": sha256_file(corpus_path),
            "count": count_jsonl(corpus_path),
        },
        "faiss": {
            "path": FAISS_RELATIVE_PATH,
            "sha256": sha256_file(faiss_path),
            **faiss_info,
        },
        "bm25": {
            "manifest_path": BM25_MANIFEST_RELATIVE_PATH,
            "sha256": sha256_file(bm25_manifest_path),
            "count": bm25_count(bm25_manifest_path),
        },
        "inference_config_sha256": sha256_file(inference_config_path),
        "model_registry_sha256": sha256_file(model_registry_path),
    }


def derive_deployment_id(payload_without_deployment_id: dict[str, Any]) -> str:
    digest = hashlib.sha256(stable_json_dumps(payload_without_deployment_id).encode("utf-8")).hexdigest()
    return f"{DEPLOYMENT_ID_PREFIX}-{digest[:16]}"


def build_artifact_lock(root: str | Path) -> dict[str, Any]:
    payload = _lock_payload_without_deployment_id(root)
    return {"deployment_id": derive_deployment_id(payload), **payload}


def write_artifact_lock(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    lock = build_artifact_lock(root_path)
    (root_path / LOCK_FILENAME).write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return lock


def _safe_diff_value(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    rendered = stable_json_dumps(value)
    return rendered if len(rendered) <= 180 else rendered[:177] + "..."


def diff_artifact_locks(
    locked: dict[str, Any],
    computed: dict[str, Any],
    *,
    max_differences: int = 20,
) -> list[str]:
    """Return a bounded field-level diff between persisted and computed locks."""

    differences: list[str] = []

    def visit(path: str, lock_value: Any, actual_value: Any) -> None:
        if len(differences) >= max_differences or lock_value == actual_value:
            return
        if isinstance(lock_value, dict) and isinstance(actual_value, dict):
            for key in sorted(set(lock_value) | set(actual_value)):
                visit(
                    f"{path}.{key}" if path else str(key),
                    lock_value.get(key, _MISSING),
                    actual_value.get(key, _MISSING),
                )
                if len(differences) >= max_differences:
                    break
            return
        differences.append(
            f"- {path}: lock={_safe_diff_value(lock_value)} actual={_safe_diff_value(actual_value)}"
        )

    visit("", locked, computed)
    if not differences:
        differences.append("- lock payload differs, but no scalar field diff could be rendered")
    return differences


def validate_artifact_lock(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    lock_path = root_path / LOCK_FILENAME
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing deployment artifact lock: {lock_path}")
    actual_lock = load_json(lock_path)
    expected_lock = build_artifact_lock(root_path)
    if actual_lock != expected_lock:
        details = "\n".join(diff_artifact_locks(actual_lock, expected_lock))
        raise RuntimeError(
            "Deployment artifact_lock.json does not match artifact bytes/configs.\n"
            "artifact lock mismatch:\n"
            f"{details}"
        )
    if actual_lock.get("shared_base_model_id") != SHARED_BASE_MODEL_ID:
        raise RuntimeError("Artifact lock shared base does not match canonical Qwen3.")
    if actual_lock.get("central_base_model_id") != CENTRAL_BASE_MODEL_ID:
        raise RuntimeError("Artifact lock Central base does not match canonical Qwen3-8B.")
    if bool(actual_lock.get("central_adapter_present")) != (actual_lock.get("central") is not None):
        raise RuntimeError("Artifact lock Central adapter presence flag is inconsistent.")

    manifest = load_json(root_path / "manifest.json")
    inference_config = load_json(root_path / INFERENCE_CONFIG_RELATIVE_PATH)
    model_registry = load_json(root_path / MODEL_REGISTRY_RELATIVE_PATH)
    if manifest.get("deployment_id") != actual_lock["deployment_id"]:
        raise RuntimeError(
            "manifest deployment_id does not match artifact_lock.json: "
            f"manifest={manifest.get('deployment_id')!r} lock={actual_lock['deployment_id']!r}"
        )
    if manifest.get("shared_base_model_id") != SHARED_BASE_MODEL_ID:
        raise RuntimeError("manifest shared base does not match canonical Qwen3.")
    registry_central = model_registry.get("central") if isinstance(model_registry.get("central"), dict) else {}
    configured_path = registry_central.get("adapter_path")
    if model_registry != registry_manifest(central_adapter_path=configured_path):
        raise RuntimeError("model_registry.json does not match app.agents.common.model_registry.")
    expected_central_manifest = registry_manifest(central_adapter_path=configured_path)["central"]
    if manifest.get("central") != expected_central_manifest:
        raise RuntimeError("manifest Central adapter contract does not match model_registry.json.")
    llm = inference_config.get("llm", {})
    if llm.get("backend") != "transformers":
        raise RuntimeError("inference_config llm.backend must be transformers.")
    if llm.get("shared_base_model_id") != SHARED_BASE_MODEL_ID:
        raise RuntimeError("inference_config shared base does not match canonical Qwen3.")
    if llm.get("central") != inference_config_payload(central_adapter_path=configured_path)["llm"]["central"]:
        raise RuntimeError("inference_config Central adapter contract does not match registry.")
    if inference_config.get("generation") != {role: spec.generation for role, spec in ROLE_MODELS.items()}:
        raise RuntimeError("inference_config generation budgets do not match registry.")
    if inference_config.get("central_generation") != CENTRAL_MODEL.generation:
        raise RuntimeError("inference_config Central generation budget does not match registry.")
    return actual_lock
