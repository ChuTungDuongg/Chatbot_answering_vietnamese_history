"""The interchangeable dense-search step of the existing hybrid retriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from app.rag.backends import DenseBackendError, QdrantSearchError


@dataclass(frozen=True)
class DenseHit:
    row_id: int
    chunk_id: str
    score: float
    rank: int
    backend: str


class ChunkIds(Sequence[str]):
    """Index corpus rows without a second list of 624k ID strings."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [str(row["chunk_id"]) for row in self.chunks[index]]
        return str(self.chunks[index]["chunk_id"])


class FaissDenseRetriever:
    name = "faiss"

    def __init__(self, index: Any, chunk_ids: Sequence[str]):
        self.index, self.chunk_ids = index, chunk_ids

    def search(self, vector: np.ndarray, k: int, *, exact: bool = False) -> list[DenseHit]:
        scores, indexes = self.index.search(np.asarray(vector, dtype="float32").reshape(1, -1),
                                             min(k, self.index.ntotal))
        return [DenseHit(row_id=int(row), chunk_id=self.chunk_ids[int(row)],
                         score=float(score), rank=rank, backend=self.name)
                for rank, (row, score) in enumerate(zip(indexes[0], scores[0]), 1)
                if int(row) >= 0]


class QdrantDenseRetriever:
    name = "qdrant"

    def __init__(self, client: Any, collection: str, chunk_ids: Sequence[str], *, hnsw_ef: int | None):
        self.client, self.collection = client, collection
        self.chunk_ids, self.hnsw_ef = chunk_ids, hnsw_ef

    def search(self, vector: np.ndarray, k: int, *, exact: bool = False) -> list[DenseHit]:
        from qdrant_client import models

        try:
            response = self.client.query_points(
                collection_name=self.collection, query=np.asarray(vector, dtype="float32").tolist(),
                using="dense_e5", limit=min(k, len(self.chunk_ids)), with_payload=["chunk_id"],
                with_vectors=False, search_params=models.SearchParams(
                    exact=exact, hnsw_ef=None if exact else self.hnsw_ef))
        except Exception:
            raise QdrantSearchError() from None
        hits = []
        for rank, point in enumerate(response.points, 1):
            try:
                row_id = int(point.id)
                score = float(point.score)
                payload_id = (point.payload or {}).get("chunk_id")
            except (ValueError, TypeError, AttributeError):
                raise QdrantSearchError() from None
            if not 0 <= row_id < len(self.chunk_ids):
                raise DenseBackendError(f"Qdrant point ID outside corpus order: {row_id}")
            chunk_id = self.chunk_ids[row_id]
            if payload_id != chunk_id:
                raise DenseBackendError(f"Qdrant point/chunk mismatch at row {row_id}")
            hits.append(DenseHit(row_id=row_id, chunk_id=chunk_id,
                                 score=score, rank=rank, backend=self.name))
        return hits
