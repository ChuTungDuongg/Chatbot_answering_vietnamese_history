"""Qdrant dense sink for the V1 ordered embedding stream.

Only a remote/persistent Qdrant server is supported for real builds. Tests inject
small fake clients; qdrant-client's in-process local mode is not used here.
"""

from __future__ import annotations

from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import Any


VECTOR_NAME = "dense_e5"
PAYLOAD_VERSION = 1
PAYLOAD_FIELDS = (
    "chunk_id", "document_id", "source_id", "title", "url",
    "source_article_id", "source_split", "historical_filter_decision",
    "historical_relevance_score", "token_count",
)


def make_client(url: str | None, api_key: str | None = None):
    if not url:
        raise ValueError("Qdrant build requires QDRANT_URL or --qdrant-url")
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key)


def point_payload(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {key: (metadata.get(key) if key == "historical_filter_decision" else row.get(key))
            for key in PAYLOAD_FIELDS}


class QdrantSink:
    def __init__(self, client, collection: str, dimension: int, out: Path):
        from qdrant_client import models

        if not collection or not collection.strip():
            raise ValueError("Qdrant collection name is required")
        self.client, self.collection, self.dimension, self.out = client, collection, dimension, out
        self.models = models
        self.stage = out / "qdrant.partial"
        if (out / "qdrant").exists() or self.stage.exists():
            raise FileExistsError("Qdrant local output or partial marker already exists")
        if client.collection_exists(collection):
            raise FileExistsError(f"Qdrant collection already exists: {collection}")
        self.stage.mkdir()
        # Server HNSW defaults are deliberate for the first full-precision baseline.
        client.create_collection(
            collection_name=collection,
            vectors_config={VECTOR_NAME: models.VectorParams(size=dimension,
                                                               distance=models.Distance.COSINE)},
        )
        self.count = 0

    def add(self, rows: list[dict[str, Any]], vectors, first_row: int) -> None:
        points = [self.models.PointStruct(
            id=first_row + offset, vector={VECTOR_NAME: vector.tolist()},
            payload=point_payload(row))
            for offset, (row, vector) in enumerate(zip(rows, vectors))]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        self.count += len(points)

    def finish(self, common: dict[str, Any], *, model_revision: str,
               requested_revision: str, batch_size: int, device: str,
               build_duration_seconds: float, embedding_stream_sha256: str,
               shared_embedding_stream: bool) -> dict[str, Any]:
        actual_count = int(self.client.count(self.collection, exact=True).count)
        if actual_count != self.count or self.count != common["corpus_chunk_count"]:
            raise RuntimeError(f"Qdrant point count mismatch: {actual_count} != {self.count}")
        info = self.client.get_collection(self.collection)
        vector = info.config.params.vectors[VECTOR_NAME]
        if int(vector.size) != self.dimension or str(vector.distance).casefold().split(".")[-1] != "cosine":
            raise RuntimeError("Qdrant vector schema differs from requested E5 COSINE schema")
        if info.config.quantization_config is not None:
            raise RuntimeError("Qdrant baseline collection unexpectedly enables quantization")
        hnsw = info.config.hnsw_config
        if not hnsw or int(hnsw.m) < 1:
            raise RuntimeError("Qdrant baseline collection has no HNSW index")
        try:
            server_version = self.client.info().version
        except Exception:
            server_version = None
        manifest = {
            **common, "count": self.count, "point_count": actual_count,
            "collection_name": self.collection,
            "point_id_rule": "zero-based chunks.jsonl row number", "vector_name": VECTOR_NAME,
            "embedding_model_id": "intfloat/multilingual-e5-base",
            "embedding_model_requested_revision": requested_revision,
            "embedding_model_resolved_revision": model_revision,
            "embedding_dimension": self.dimension,
            "normalize_embeddings": True, "distance": "COSINE", "quantization": "none",
            "passage_prefix": "passage: ", "query_prefix": "query: ",
            "passage_template": "passage: {title}\n{text}",
            "hnsw_configuration": {
                "source": "server defaults; effective values captured after collection creation",
                "m": hnsw.m, "ef_construct": hnsw.ef_construct,
                "full_scan_threshold": hnsw.full_scan_threshold,
            },
            "payload_schema_version": PAYLOAD_VERSION,
            "payload_fields": list(PAYLOAD_FIELDS), "payload_includes_text": False,
            "embedding_batch_size": batch_size, "device": device,
            "build_duration_seconds": round(build_duration_seconds, 3),
            "embedding_stream_sha256": embedding_stream_sha256,
            "shared_embedding_stream": shared_embedding_stream,
            "qdrant_client_version": version("qdrant-client"),
            "qdrant_server_version": server_version,
            "collection_status_at_finalize": str(info.status),
            "indexed_vectors_at_finalize": info.indexed_vectors_count,
        }
        (self.stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(self.stage, self.out / "qdrant")
        return manifest
