"""Load validated dense backends once, alongside shared corpus and BM25S.

This service has no generation model and never builds or modifies an index.
"""

import json
import hashlib
import logging
import time
from typing import Any

from app.config import settings
from app.rag.artifact_identity import fingerprint, validate_v1_manifests


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
        self.ordered_chunk_id_sha256: str | None = None
        self.corpus_sha256: str | None = None
        self.faiss_index = None
        self.dense_retriever = None
        self.dense_retrievers: dict[str, Any] = {}
        self.dense_backend_errors: dict[str, str] = {}
        self.default_dense_backend = settings.retrieval_dense_backend
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
            self._load_dense()
            self._stage("bm25_load", self._load_bm25)
            self._stage("embedder_load", self._load_embedder)
            self._stage("reranker_load", self._load_reranker)
        self.loaded = True
        self.started_at = time.time()

    def shutdown(self) -> None:
        for backend in self.dense_retrievers.values():
            client = getattr(backend, "client", None)
            self._close_client(client)
        self.dense_retrievers.clear()
        self.reranker = None
        self.embedder = None
        self.bm25 = None
        self.faiss_index = None
        self.dense_retriever = None
        self.embedding_revision = None
        self.embedding_dimension = None
        self.chunks = []
        self.ordered_chunk_id_sha256 = None
        self.corpus_sha256 = None
        self.loaded = False

    def _validate_artifacts(self) -> None:
        if settings.retrieval_dense_backend == "qdrant" and not settings.qdrant_manifest_path.is_file():
            raise RuntimeError("Qdrant V1 index is not finalized; retrieval/qdrant/manifest.json is missing.")
        missing = [str(path) for path in settings.required_retrieval_paths() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing retrieval artifacts: " + ", ".join(missing))

    @staticmethod
    def _close_client(client):
        if client is not None and callable(getattr(client, "close", None)):
            try:
                client.close()
            except Exception as exc:
                logger.warning("Qdrant client cleanup failed: type=%s", type(exc).__name__)

    def _load_config(self) -> None:
        with settings.inference_config_path.open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        with settings.manifest_path.open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if settings.retrieval_root is not None:
            if settings.retrieval_dense_backend not in self.manifest.get("available_dense_backends", []):
                raise RuntimeError(f"{settings.retrieval_dense_backend} is not listed as a finalized dense backend")
            retrieval = self.config.get("retrieval", {})
            for key in ("embedding_model_id", "reranker_model_id", "reranker_model_revision"):
                if retrieval.get(key) != self.manifest.get(key):
                    raise RuntimeError(f"Runtime {key} differs from runtime manifest")
            if {k: retrieval.get(k) for k in self.manifest.get("retrieval_settings", {})} != self.manifest.get("retrieval_settings"):
                raise RuntimeError("Retrieval settings differ from runtime manifest")

    def _load_corpus(self) -> None:
        chunks = []
        corpus_digest = hashlib.sha256()
        ids_digest = hashlib.sha256()
        seen_ids: set[str] = set()
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
                chunk_id = str(chunk["chunk_id"])
                if chunk_id in seen_ids:
                    raise RuntimeError("Duplicate chunk_id in corpus")
                seen_ids.add(chunk_id)
                ids_digest.update((chunk_id + "\n").encode("utf-8"))
                chunks.append(chunk)
        expected = int(self.manifest["corpus"]["count"])
        if len(chunks) != expected:
            raise RuntimeError(f"Corpus count mismatch: {len(chunks)} != {expected}")
        self.ordered_chunk_id_sha256 = ids_digest.hexdigest()
        self.corpus_sha256 = corpus_digest.hexdigest()
        self.chunks = chunks
        if settings.retrieval_root is not None:
            expected = {"count": len(chunks), "corpus_sha256": self.corpus_sha256,
                        "ordered_chunk_id_sha256": self.ordered_chunk_id_sha256}
            for key, actual in expected.items():
                if self.manifest["corpus"].get(key) != actual:
                    raise RuntimeError(f"Runtime corpus {key} mismatch")
            if settings.corpus_path.stat().st_size != int(self.manifest["corpus"].get("bytes", -1)):
                raise RuntimeError("Runtime corpus byte count mismatch")
            manifests = validate_v1_manifests(settings.retrieval_dir, expected,
                include_qdrant=settings.retrieval_dense_backend == "qdrant")
            for name in manifests:
                path = {"faiss": settings.faiss_manifest_path,
                        "bm25": settings.bm25_manifest_path,
                        "index": settings.index_manifest_path,
                        "qdrant": settings.qdrant_manifest_path}[name]
                if self.manifest["manifest_fingerprints"].get(name) != fingerprint(path):
                    raise RuntimeError(f"Runtime {name} manifest fingerprint mismatch; regenerate small runtime metadata")
            faiss = manifests["faiss"]
            for key in ("embedding_model_id", "embedding_model_resolved_revision", "embedding_dimension"):
                if self.manifest.get(key) != faiss.get(key):
                    raise RuntimeError(f"Runtime {key} differs from FAISS manifest")

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
        # The default is mandatory; optional lanes can fail without disabling it.
        names = (self.default_dense_backend, *(
            name for name in settings.requested_dense_backends if name != self.default_dense_backend))
        for name in names:
            try:
                if settings.retrieval_root is not None and name not in self.manifest.get("available_dense_backends", []):
                    raise RuntimeError("Dense backend is not finalized in the runtime manifest")
                self._stage(f"{name}_load", self._load_faiss if name == "faiss" else self._load_qdrant)
            except Exception as exc:
                self.dense_backend_errors[name] = type(exc).__name__
                if name == self.default_dense_backend:
                    raise
                logger.warning("Optional dense backend unavailable: backend=%s error_type=%s", name, type(exc).__name__)

    def get_dense_retriever(self, name: str | None = None):
        from app.rag.backends import DenseBackendUnavailable
        name = name or self.default_dense_backend
        if name not in self.dense_retrievers:
            raise DenseBackendUnavailable(
                "Qdrant hiện không khả dụng. Hãy chọn FAISS hoặc thử lại."
                if name == "qdrant" else "FAISS hiện không khả dụng.")
        return self.dense_retrievers[name]

    def _register_dense(self, name: str, backend, manifest: dict[str, Any], dimension: int):
        revision = manifest.get("embedding_model_resolved_revision")
        if self.dense_retrievers and (self.embedding_revision != revision or self.embedding_dimension != dimension):
            raise RuntimeError("Dense backends have different embedding revision/dimension")
        self.embedding_revision, self.embedding_dimension = revision, dimension
        self.dense_retrievers[name] = backend
        # Static compatibility alias for historical evaluators; never changed by requests.
        if name == self.default_dense_backend:
            self.dense_retriever = backend

    def _load_faiss(self) -> None:
        import faiss

        from app.rag.dense_backend import FaissDenseRetriever

        manifest = self._validate_index_manifest(settings.faiss_manifest_path)
        model_id = manifest.get("embedding_model_id", manifest.get("embedding_model"))
        if model_id != self.config["retrieval"]["embedding_model_id"]:
            raise RuntimeError("FAISS embedding model differs from runtime config")
        index = faiss.read_index(str(settings.faiss_path))
        if index.ntotal != len(self.chunks):
            raise RuntimeError("FAISS/corpus count mismatch")
        if manifest.get("embedding_dimension") is not None and index.d != int(manifest["embedding_dimension"]):
            raise RuntimeError("FAISS vector dimension differs from manifest")
        from app.rag.dense_backend import ChunkIds
        self._register_dense("faiss", FaissDenseRetriever(index, ChunkIds(self.chunks)),
                             manifest, manifest.get("embedding_dimension", getattr(index, "d", None)))
        self.faiss_index = index

    def _load_qdrant(self) -> None:
        if len(settings.requested_dense_backends) > 1 and settings.retrieval_root is None:
            raise RuntimeError("Dynamic dense backends require the validated V1 runtime layout")
        if not settings.qdrant_manifest_path.is_file():
            raise RuntimeError("Qdrant V1 index is not finalized; retrieval/qdrant/manifest.json is missing.")
        if not settings.qdrant_url:
            raise RuntimeError("RETRIEVAL_DENSE_BACKEND=qdrant requires QDRANT_URL")
        if not settings.qdrant_api_key:
            raise RuntimeError("RETRIEVAL_DENSE_BACKEND=qdrant requires QDRANT_API_KEY")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Qdrant lane requires qdrant-client; install requirements.txt") from exc
        manifest = self._validate_index_manifest(settings.qdrant_manifest_path)
        if settings.retrieval_root is not None:
            expected = {"count": len(self.chunks), "corpus_sha256": self.corpus_sha256,
                        "ordered_chunk_id_sha256": self.ordered_chunk_id_sha256}
            validate_v1_manifests(settings.retrieval_dir, expected, include_qdrant=True)
            if self.manifest["manifest_fingerprints"].get("qdrant") != fingerprint(settings.qdrant_manifest_path):
                raise RuntimeError("Runtime Qdrant manifest fingerprint mismatch")
        if manifest.get("collection_name") != settings.qdrant_collection:
            raise RuntimeError("Qdrant collection differs from manifest")
        if manifest.get("distance") != "COSINE" or manifest.get("quantization") != "none":
            raise RuntimeError("Qdrant manifest does not describe the full-precision COSINE baseline")
        if manifest.get("embedding_model_id") != self.config["retrieval"]["embedding_model_id"]:
            raise RuntimeError("Qdrant embedding model differs from runtime config")
        try:
            client = QdrantClient(url=settings.qdrant_url,
                                  api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
                                  timeout=settings.qdrant_timeout_seconds)
        except Exception:
            raise RuntimeError("Qdrant connection configuration is invalid") from None
        try:
            self._validate_qdrant_client(client, manifest)
        except BaseException:
            self._close_client(client)
            raise

    def _validate_qdrant_client(self, client, manifest):
        from app.rag.dense_backend import QdrantDenseRetriever
        try:
            count = int(client.count(settings.qdrant_collection, exact=True).count)
        except Exception:
            raise RuntimeError("Qdrant collection unavailable or misconfigured") from None
        if count != len(self.chunks):
            raise RuntimeError(f"Qdrant/corpus count mismatch: {count} != {len(self.chunks)}")
        try:
            collection = client.get_collection(settings.qdrant_collection)
        except Exception:
            raise RuntimeError("Qdrant collection unavailable or misconfigured") from None
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
        if not manifest.get("embedding_model_resolved_revision"):
            raise RuntimeError("Qdrant manifest lacks pinned E5 revision")
        from app.rag.dense_backend import ChunkIds
        backend = QdrantDenseRetriever(
            client, settings.qdrant_collection, ChunkIds(self.chunks),
            hnsw_ef=settings.qdrant_hnsw_ef)
        self._register_dense("qdrant", backend, manifest, int(manifest["embedding_dimension"]))

    def _load_bm25(self) -> None:
        import bm25s

        manifest = self._validate_index_manifest(settings.bm25_manifest_path)
        if settings.retrieval_root is not None and manifest.get("embedding_model_id") != self.config["retrieval"]["embedding_model_id"]:
            raise RuntimeError("BM25 embedding model differs from runtime config")
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

        kwargs = {"device": self._compute_device()}
        revision = self.config["retrieval"].get("reranker_model_revision")
        if revision:
            kwargs["revision"] = revision
        self.reranker = CrossEncoder(self.config["retrieval"]["reranker_model_id"], **kwargs)

    def readiness(self) -> dict[str, Any]:
        retrieval_ready = all((self.loaded, bool(self.chunks), self.dense_retriever is not None,
                               self.bm25 is not None, self.embedder is not None, self.reranker is not None))
        return {
            "ready": self.loaded if settings.is_api_only else retrieval_ready,
            "corpus_loaded": bool(self.chunks),
            "faiss_loaded": self.faiss_index is not None,
            "dense_backend": settings.retrieval_dense_backend,
            "dense_loaded": self.dense_retriever is not None,
            "qdrant_loaded": "qdrant" in self.dense_retrievers,
            "default_dense_backend": self.default_dense_backend,
            "dense_backends": {name: {"available": name in self.dense_retrievers}
                               for name in ("faiss", "qdrant")},
            "retrieval": {"default_backend": self.default_dense_backend,
                          "available_backends": list(self.dense_retrievers)},
            "corpus_identity": {"count": len(self.chunks), "corpus_sha256": self.corpus_sha256,
                                "ordered_chunk_id_sha256": self.ordered_chunk_id_sha256,
                                "embedding_model_id": (self.config or {}).get("retrieval", {}).get("embedding_model_id"),
                                "embedding_dimension": self.embedding_dimension,
                                "embedding_revision": self.embedding_revision},
            "qdrant_hnsw_ef": settings.qdrant_hnsw_ef if "qdrant" in self.dense_retrievers else None,
            "bm25_loaded": self.bm25 is not None,
            "bm25_documents": int(self.bm25.scores["num_docs"]) if self.bm25 is not None else None,
            "embedder_loaded": self.embedder is not None,
            "reranker_loaded": self.reranker is not None,
            "model_loaded": False,
            "corpus_chunks": len(self.chunks) if self.chunks else None,
            "faiss_vectors": int(self.faiss_index.ntotal) if self.faiss_index is not None else None,
            "device": settings.device if settings.should_load_retrieval else "api-only",
        }
