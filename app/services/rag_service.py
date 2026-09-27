"""Load the historical corpus and selected dense backend with shared BM25S.

This service has no generation model and never builds or modifies an index.
"""

import json
import hashlib
import logging
import time
from typing import Any

from app.config import settings


logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self) -> None:
        self.loaded = False
        self.runtime_mode = settings.app_mode
        self.started_at: float | None = None
        self.startup_timings_ms: dict[str, float] = {}
        self.config: dict[str, Any] | None = None
        self.manifest: dict[str, Any] | None = None
        self.chunks: list[dict[str, Any]] = []
        self.chunk_by_id: dict[str, dict[str, Any]] = {}
        self.ordered_chunk_id_sha256: str | None = None
        self.corpus_sha256: str | None = None
        self.faiss_index = None
        self.dense_retriever = None
        self.embedding_revision: str | None = None
        self.embedding_dimension: int | None = None
        self.bm25 = None
        self.embedder = None
        self.reranker = None

    def _stage(self, name: str, action) -> None:
        start = time.perf_counter_ns()
        action()
        self.startup_timings_ms[name] = (time.perf_counter_ns() - start) / 1e6

    def load(self) -> None:
        if self.loaded:
            return
        if settings.should_load_retrieval:
            self._stage("artifact_validation", self._validate_artifacts)
            self._stage("config_load", self._load_config)
            self._stage("corpus_load", self._load_corpus)
            dense_stage = "faiss_load" if settings.retrieval_dense_backend == "faiss" else "qdrant_load"
            self._stage(dense_stage, self._load_dense)
            self._stage("bm25_load", self._load_bm25)
            self._stage("embedder_load", self._load_embedder)
            self._stage("reranker_load", self._load_reranker)
        self.loaded = True
        self.started_at = time.time()

    def shutdown(self) -> None:
        self.reranker = None
        self.embedder = None
        self.bm25 = None
        self.faiss_index = None
        self.dense_retriever = None
        self.embedding_revision = None
        self.embedding_dimension = None
        self.chunks = []
        self.chunk_by_id = {}
        self.ordered_chunk_id_sha256 = None
        self.corpus_sha256 = None
        self.loaded = False

    def _validate_artifacts(self) -> None:
        missing = [str(path) for path in settings.required_retrieval_paths() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing retrieval artifacts: " + ", ".join(missing))

    def _load_config(self) -> None:
        with settings.inference_config_path.open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        with settings.manifest_path.open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)

    def _load_corpus(self) -> None:
        chunks = []
        corpus_digest = hashlib.sha256()
        with settings.corpus_path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                corpus_digest.update(raw)
                line = raw.decode("utf-8")
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid corpus JSON at line {line_number}") from exc
                if not chunk.get("chunk_id"):
                    raise RuntimeError(f"Corpus chunk_id missing at line {line_number}")
                chunks.append(chunk)
        expected = int(self.manifest["corpus"]["count"])
        if len(chunks) != expected:
            raise RuntimeError(f"Corpus count mismatch: {len(chunks)} != {expected}")
        by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
        if len(by_id) != len(chunks):
            raise RuntimeError("Duplicate chunk_id in corpus")
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update((str(chunk["chunk_id"]) + "\n").encode("utf-8"))
        self.ordered_chunk_id_sha256 = digest.hexdigest()
        self.corpus_sha256 = corpus_digest.hexdigest()
        self.chunks, self.chunk_by_id = chunks, by_id

    def _validate_index_manifest(self, path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("count", -1)) != len(self.chunks):
            raise RuntimeError(f"Index/corpus count mismatch: {path}")
        expected = manifest.get("ordered_chunk_id_sha256")
        if expected is not None and expected != self.ordered_chunk_id_sha256:
            raise RuntimeError(f"Index/corpus row order mismatch: {path}")

        expected_bytes = manifest.get("corpus_sha256")
        if expected_bytes is not None and expected_bytes != self.corpus_sha256:
            raise RuntimeError(f"Index/corpus SHA-256 mismatch: {path}")
        return manifest

    def _load_dense(self) -> None:
        if settings.retrieval_dense_backend == "faiss":
            self._load_faiss()
        else:
            self._load_qdrant()

    def _load_faiss(self) -> None:
        import faiss

        from app.rag.dense_backend import FaissDenseRetriever

        manifest = self._validate_index_manifest(settings.faiss_manifest_path)
        model_id = manifest.get("embedding_model_id", manifest.get("embedding_model"))
        if model_id != self.config["retrieval"]["embedding_model_id"]:
            raise RuntimeError("FAISS embedding model differs from runtime config")
        self.faiss_index = faiss.read_index(str(settings.faiss_path))
        if self.faiss_index.ntotal != len(self.chunks):
            raise RuntimeError("FAISS/corpus count mismatch")
        self.embedding_revision = manifest.get("embedding_model_resolved_revision")
        self.embedding_dimension = manifest.get("embedding_dimension")
        self.dense_retriever = FaissDenseRetriever(
            self.faiss_index, [str(row["chunk_id"]) for row in self.chunks])

    def _load_qdrant(self) -> None:
        if not settings.qdrant_url:
            raise RuntimeError("RETRIEVAL_DENSE_BACKEND=qdrant requires QDRANT_URL")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Qdrant lane requires qdrant-client; install requirements.txt") from exc
        from app.rag.dense_backend import QdrantDenseRetriever

        manifest = self._validate_index_manifest(settings.qdrant_manifest_path)
        if manifest.get("collection_name") != settings.qdrant_collection:
            raise RuntimeError("Qdrant collection differs from manifest")
        if manifest.get("distance") != "COSINE" or manifest.get("quantization") != "none":
            raise RuntimeError("Qdrant manifest does not describe the full-precision COSINE baseline")
        if manifest.get("embedding_model_id") != self.config["retrieval"]["embedding_model_id"]:
            raise RuntimeError("Qdrant embedding model differs from runtime config")
        try:
            client = QdrantClient(url=settings.qdrant_url,
                                  api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None)
        except Exception:
            raise RuntimeError("Qdrant connection configuration is invalid") from None
        try:
            count = int(client.count(settings.qdrant_collection, exact=True).count)
        except Exception:
            raise RuntimeError("Qdrant collection unavailable or misconfigured") from None
        if count != len(self.chunks):
            raise RuntimeError(f"Qdrant/corpus count mismatch: {count} != {len(self.chunks)}")
        collection = client.get_collection(settings.qdrant_collection)
        if str(collection.status).casefold().split(".")[-1] != "green":
            raise RuntimeError(f"Qdrant collection is not ready: {collection.status}")
        vector = collection.config.params.vectors.get("dense_e5")
        if (vector is None or int(vector.size) != int(manifest["embedding_dimension"]) or
                str(vector.distance).casefold().split(".")[-1] != "cosine" or
                collection.config.quantization_config is not None):
            raise RuntimeError("Qdrant collection vector configuration differs from manifest")
        effective_hnsw = collection.config.hnsw_config
        saved_hnsw = manifest.get("hnsw_configuration") or {}
        if any(saved_hnsw.get(key) != getattr(effective_hnsw, key)
               for key in ("m", "ef_construct", "full_scan_threshold")):
            raise RuntimeError("Qdrant HNSW configuration differs from manifest")
        self.embedding_revision = manifest.get("embedding_model_resolved_revision")
        self.embedding_dimension = manifest.get("embedding_dimension")
        if not self.embedding_revision:
            raise RuntimeError("Qdrant manifest lacks pinned E5 revision")
        self.dense_retriever = QdrantDenseRetriever(
            client, settings.qdrant_collection, [str(row["chunk_id"]) for row in self.chunks],
            hnsw_ef=settings.qdrant_hnsw_ef)

    def _load_bm25(self) -> None:
        import bm25s

        self._validate_index_manifest(settings.bm25_manifest_path)
        self.bm25 = bm25s.BM25.load(str(settings.bm25_path), mmap=True, load_corpus=False)
        if int(self.bm25.scores["num_docs"]) != len(self.chunks):
            raise RuntimeError("BM25/corpus count mismatch")

    def _compute_device(self) -> str:
        if settings.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
        return settings.device

    def _load_embedder(self) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": self._compute_device()}
        if self.embedding_revision:
            kwargs["revision"] = self.embedding_revision
        self.embedder = SentenceTransformer(self.config["retrieval"]["embedding_model_id"], **kwargs)
        if (self.embedding_dimension is not None and
                self.embedder.get_sentence_embedding_dimension() != self.embedding_dimension):
            raise RuntimeError("Runtime E5 dimension differs from dense index manifest")
        self.embedder.max_seq_length = 512

    def _load_reranker(self) -> None:
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(
            self.config["retrieval"]["reranker_model_id"], device=self._compute_device()
        )

    def readiness(self) -> dict[str, Any]:
        retrieval_ready = all((self.loaded, bool(self.chunks), self.dense_retriever is not None,
                               self.bm25 is not None, self.embedder is not None, self.reranker is not None))
        return {
            "ready": self.loaded if settings.is_api_only else retrieval_ready,
            "corpus_loaded": bool(self.chunks),
            "faiss_loaded": self.faiss_index is not None,
            "dense_backend": settings.retrieval_dense_backend,
            "dense_loaded": self.dense_retriever is not None,
            "qdrant_hnsw_ef": settings.qdrant_hnsw_ef if settings.retrieval_dense_backend == "qdrant" else None,
            "bm25_loaded": self.bm25 is not None,
            "embedder_loaded": self.embedder is not None,
            "reranker_loaded": self.reranker is not None,
            "model_loaded": False,
            "corpus_chunks": len(self.chunks) if self.chunks else None,
            "faiss_vectors": int(self.faiss_index.ntotal) if self.faiss_index is not None else None,
            "device": settings.device if settings.should_load_retrieval else "api-only",
        }
