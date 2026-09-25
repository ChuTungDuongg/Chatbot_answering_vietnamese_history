"""Load the historical corpus and existing FAISS/BM25S retrieval stack.

This service has no generation model and never builds or modifies an index.
"""

import json
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
        self.faiss_index = None
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
            self._stage("faiss_load", self._load_faiss)
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
        self.chunks = []
        self.chunk_by_id = {}
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
        with settings.corpus_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
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
        self.chunks, self.chunk_by_id = chunks, by_id

    def _load_faiss(self) -> None:
        import faiss

        self.faiss_index = faiss.read_index(str(settings.faiss_path))
        if self.faiss_index.ntotal != len(self.chunks):
            raise RuntimeError("FAISS/corpus count mismatch")

    def _load_bm25(self) -> None:
        import bm25s

        with settings.bm25_manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("count", -1)) != len(self.chunks):
            raise RuntimeError("BM25S/corpus count mismatch")
        self.bm25 = bm25s.BM25.load(str(settings.bm25_path), mmap=True, load_corpus=False)

    def _compute_device(self) -> str:
        if settings.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
        return settings.device

    def _load_embedder(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(
            self.config["retrieval"]["embedding_model_id"], device=self._compute_device()
        )
        self.embedder.max_seq_length = 512

    def _load_reranker(self) -> None:
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(
            self.config["retrieval"]["reranker_model_id"], device=self._compute_device()
        )

    def readiness(self) -> dict[str, Any]:
        retrieval_ready = all((self.loaded, bool(self.chunks), self.faiss_index is not None,
                               self.bm25 is not None, self.embedder is not None, self.reranker is not None))
        return {
            "ready": self.loaded if settings.is_api_only else retrieval_ready,
            "corpus_loaded": bool(self.chunks),
            "faiss_loaded": self.faiss_index is not None,
            "bm25_loaded": self.bm25 is not None,
            "embedder_loaded": self.embedder is not None,
            "reranker_loaded": self.reranker is not None,
            "model_loaded": False,
            "corpus_chunks": len(self.chunks) if self.chunks else None,
            "faiss_vectors": int(self.faiss_index.ntotal) if self.faiss_index is not None else None,
            "device": settings.device if settings.should_load_retrieval else "api-only",
        }
