"""Strict identity checks for completed Corpus V1 retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = ("count", "corpus_sha256", "ordered_chunk_id_sha256")


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Index manifest is not an object: {path}")
    return value


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_identity(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest = read_manifest(path)
    for field in IDENTITY_FIELDS:
        actual = manifest.get(field)
        if field == "count":
            actual = manifest.get("count", manifest.get("corpus_chunk_count"))
        if actual != expected[field]:
            raise RuntimeError(f"{path}: {field} differs from corpus")
    if ("corpus_chunk_count" in manifest and
            manifest["corpus_chunk_count"] != expected["count"]):
        raise RuntimeError(f"{path}: corpus_chunk_count differs from corpus")
    return manifest


def validate_v1_manifests(root: Path, expected: dict[str, Any], *,
                          include_qdrant: bool = False) -> dict[str, dict[str, Any]]:
    paths = {"faiss": root / "faiss" / "manifest.json",
             "bm25": root / "bm25s_index" / "phase9_manifest.json",
             "index": root / "index_manifest.json"}
    if include_qdrant:
        paths["qdrant"] = root / "qdrant" / "manifest.json"
    manifests = {name: validate_identity(path, expected) for name, path in paths.items()}
    faiss, index = manifests["faiss"], manifests["index"]
    for name in ("embedding_model_id", "embedding_model_resolved_revision",
                 "embedding_dimension", "query_prefix", "passage_template"):
        if faiss.get(name) != index.get(name):
            raise RuntimeError(f"FAISS/index {name} mismatch")
    if manifests["bm25"].get("bm25_configuration") != index.get("bm25_configuration"):
        raise RuntimeError("BM25/index configuration mismatch")
    if include_qdrant:
        qdrant = manifests["qdrant"]
        for name in ("embedding_model_id", "embedding_model_resolved_revision",
                     "embedding_dimension", "query_prefix", "passage_template"):
            if qdrant.get(name) != faiss.get(name):
                raise RuntimeError(f"FAISS/Qdrant {name} mismatch")
        if qdrant.get("distance") != "COSINE" or qdrant.get("quantization") != "none":
            raise RuntimeError("Qdrant manifest has incompatible vector configuration")
    return manifests
