from __future__ import annotations

import os
from pathlib import Path
from typing import Any


CENTRAL_REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
)
INFERENCE_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "*.safetensors.index.json",
    "model*.json",
)


def resolve_hf_hub_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None and str(cache_dir).strip():
        return Path(cache_dir)
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    transformers_cache = os.getenv("TRANSFORMERS_CACHE")
    if transformers_cache:
        return Path(transformers_cache)
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_status(model_id: str, *, cache_dir: str | Path | None = None) -> dict[str, Any]:
    root = resolve_hf_hub_cache_dir(cache_dir)
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=model_id,
            cache_dir=str(root),
            local_files_only=True,
            allow_patterns=INFERENCE_ALLOW_PATTERNS,
        )
        snapshot_path = Path(snapshot)
        present = sorted(path.name for path in snapshot_path.iterdir() if path.is_file())
        required_present = {
            name: (snapshot_path / name).is_file()
            for name in CENTRAL_REQUIRED_FILES
        }
        return {
            "model_id": model_id,
            "cache_root": str(root),
            "snapshot_path": str(snapshot_path),
            "cache_hit": all(required_present.values()),
            "required_files": required_present,
            "files": present[:30],
            "error": None,
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "cache_root": str(root),
            "snapshot_path": None,
            "cache_hit": False,
            "required_files": {name: False for name in CENTRAL_REQUIRED_FILES},
            "files": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def seed_hf_cache(model_id: str, *, cache_dir: str | Path | None = None, local_files_only: bool = False) -> dict[str, Any]:
    root = resolve_hf_hub_cache_dir(cache_dir)
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=model_id,
        cache_dir=str(root),
        local_files_only=local_files_only,
        allow_patterns=INFERENCE_ALLOW_PATTERNS,
    )
    status = hf_cache_status(model_id, cache_dir=root)
    return {**status, "snapshot_path": str(snapshot)}
