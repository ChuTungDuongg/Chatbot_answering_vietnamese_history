import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import bm25s
import faiss
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.artifact_contract import validate_artifact_lock
from app.config import settings
from app.telemetry import log_event


logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        # Runtime state
        self.started_at: float | None = None
        self.loaded: bool = False
        self.runtime_mode: str = settings.app_mode
        self.runtime_device: str | None = None
        self.deployment_id: str | None = None
        self.startup_timings_ms: dict[str, float] = {}
        self.artifact_lock: dict[str, Any] | None = None

        # Deployment config / manifest
        self.config: dict[str, Any] | None = None
        self.manifest: dict[str, Any] | None = None

        # Corpus
        self.chunks: list[dict[str, Any]] = []
        self.chunk_by_id: dict[str, dict[str, Any]] = {}

        # Retrieval
        self.faiss_index = None
        self.bm25 = None
        self.embedder = None
        self.reranker = None

        # Generation
        self.tokenizer = None
        self.model = None
        self.external_generation_backend = False

    # ========================================================
    # Public lifecycle
    # ========================================================

    def load(self) -> None:
        if self.loaded:
            return

        started = time.perf_counter()

        print("=" * 70)
        print("Loading Vietnamese History RAG service")
        print("=" * 70)
        print(f"Runtime mode : {settings.app_mode}")
        print(f"Target device: {settings.device}")
        self._log_gpu_profile()

        # MODE 1 — API ONLY
        if settings.is_api_only:
            self.runtime_device = "api-only"
            self.loaded = True
            self.started_at = time.time()

            print("✅ API-only mode: skipping all AI/RAG artifacts.")
            print(f"✅ API service ready in {time.perf_counter() - started:.2f}s")
            return

        # MODE 2 / 3 — retrieval-only OR full
        if settings.should_load_retrieval:
            self._time_stage("artifact_path_validation", self._validate_artifacts)
            self._time_stage("config_load", self._load_config)
            if settings.is_full and settings.llm_backend == "transformers":
                self._time_stage("artifact_lock_validation", self._validate_artifact_lock)
            self._time_stage("corpus_load", self._load_corpus)
            self._time_stage("faiss_load", self._load_faiss)
            self._time_stage("bm25_load", self._load_bm25)
            self._time_stage("embedder_loaded", self._load_embedder)
            self._time_stage("reranker_loaded", self._load_reranker)

        # MODE 3 — FULL
        if settings.should_load_model and not settings.uses_shared_backend:
            self._time_stage("legacy_generation_model_loaded", self._load_generation_model)

        self.loaded = True
        self.started_at = time.time()

        print(f"✅ RAG service ready in {time.perf_counter() - started:.2f}s")

    def shutdown(self) -> None:
        print("Shutting down Vietnamese History RAG service...")

        # Generation
        self.model = None
        self.tokenizer = None
        self.external_generation_backend = False

        # Retrieval models
        self.embedder = None
        self.reranker = None

        # Retrieval indexes
        self.faiss_index = None
        self.bm25 = None

        # Corpus
        self.chunks = []
        self.chunk_by_id = {}

        # Config
        self.config = None
        self.manifest = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        self.loaded = False
        self.started_at = None
        self.runtime_device = None
        self.deployment_id = None
        self.startup_timings_ms = {}
        self.artifact_lock = None

        print("✅ RAG service shutdown complete.")

    def _time_stage(self, stage: str, func) -> None:
        started = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.startup_timings_ms[stage] = elapsed_ms
        log_event("STARTUP_STAGE", stage=stage, elapsed_ms=elapsed_ms)
        self._log_gpu_memory_stage(stage)

    # ========================================================
    # Device helpers
    # ========================================================

    def _resolve_compute_device(self) -> str:
        requested = settings.device.strip().lower()

        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"

            # Full inference không nên âm thầm load Qwen 3B lên CPU
            # vì cấu hình cloud bị sai.
            if settings.is_full:
                raise RuntimeError(
                    "APP_MODE=full và DEVICE=cuda nhưng PyTorch không phát hiện CUDA. "
                    "Hãy kiểm tra GPU/CUDA environment."
                )

            print("⚠️ CUDA unavailable → falling back to CPU.")
            return "cpu"

        if requested == "cpu":
            return "cpu"

        raise ValueError(
            f"Unsupported DEVICE={settings.device!r}. Expected 'cpu' or 'cuda'."
        )

    def _log_gpu_profile(self) -> None:
        profile = {
            "gpu_name": None,
            "compute_capability": None,
            "total_vram_gb": None,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "bnb_4bit": True,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            profile.update({
                "gpu_name": props.name,
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "total_vram_gb": props.total_memory / 1024**3,
            })
        log_event("GPU_PROFILE", **profile)

    def _log_gpu_memory_stage(self, stage: str) -> None:
        payload = {
            "stage": stage,
            "allocated_gb": None,
            "reserved_gb": None,
            "peak_gb": None,
        }
        if torch.cuda.is_available():
            payload.update({
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            })
        log_event("GPU_MEMORY_STAGE", **payload)

    # ========================================================
    # Artifact validation
    # ========================================================

    def _validate_artifacts(self) -> None:
        if settings.is_retrieval_only:
            required = settings.required_retrieval_paths()
        elif settings.is_full:
            required = settings.required_full_paths()
        else:
            return

        missing = [str(path) for path in required if not Path(path).exists()]

        if missing:
            raise FileNotFoundError(
                f"Missing deployment artifacts for APP_MODE={settings.app_mode}:\n"
                + "\n".join(missing)
            )

        print("✅ Deployment artifacts found.")

    def _validate_artifact_lock(self) -> None:
        lock = validate_artifact_lock(settings.artifact_root)
        self.artifact_lock = lock
        self.deployment_id = str(lock["deployment_id"])
        log_event(
            "ARTIFACT_DEPLOYMENT",
            deployment_id=self.deployment_id,
            shared_base=lock["shared_base_model_id"],
            research_sha=lock["roles"]["research"]["adapter_model_sha256"][:12],
            evidence_sha=lock["roles"]["evidence"]["adapter_model_sha256"][:12],
            history_sha=lock["roles"]["history"]["adapter_model_sha256"][:12],
            corpus_count=lock["corpus"]["count"],
            faiss_count=lock["faiss"]["ntotal"],
            bm25_count=lock["bm25"]["count"],
            validation="PASS",
        )

    # ========================================================
    # Deployment config
    # ========================================================

    def _load_config(self) -> None:
        with settings.inference_config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        with settings.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        print("✅ Config + manifest loaded")

    # ========================================================
    # Expected corpus count
    # ========================================================

    def _expected_corpus_count(self) -> int:
        # Phase 10 manifest là source of truth.
        try:
            return int(self.manifest["corpus"]["count"])
        except Exception:
            # Fallback theo Phase 9 hiện tại.
            return 58_603

    # ========================================================
    # Corpus
    # ========================================================

    def _load_corpus(self) -> None:
        chunks: list[dict[str, Any]] = []

        with settings.corpus_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in corpus at line {line_no}"
                    ) from exc

                if "chunk_id" not in row:
                    raise RuntimeError(
                        f"Corpus row missing chunk_id at line {line_no}"
                    )

                chunks.append(row)

        self.chunks = chunks
        self.chunk_by_id = {
            str(chunk["chunk_id"]): chunk
            for chunk in chunks
        }

        expected = self._expected_corpus_count()

        if len(self.chunks) != expected:
            raise RuntimeError(
                f"Corpus count mismatch: {len(self.chunks):,} != {expected:,}"
            )

        if len(self.chunk_by_id) != len(self.chunks):
            raise RuntimeError(
                "Duplicate chunk_id detected in deployment corpus."
            )

        print(f"✅ Corpus loaded: {len(self.chunks):,}")

    # ========================================================
    # FAISS
    # ========================================================

    def _load_faiss(self) -> None:
        self.faiss_index = faiss.read_index(str(settings.faiss_path))
        expected = len(self.chunks)

        if self.faiss_index.ntotal != expected:
            raise RuntimeError(
                f"FAISS/corpus count mismatch: "
                f"{self.faiss_index.ntotal:,} vs {expected:,}"
            )

        print(f"✅ FAISS loaded: {self.faiss_index.ntotal:,}")

    # ========================================================
    # BM25S
    # ========================================================

    def _load_bm25(self) -> None:
        self.bm25 = bm25s.BM25.load(
            str(settings.bm25_path),
            mmap=True,
            load_corpus=False,
        )

        try:
            with settings.bm25_manifest_path.open("r", encoding="utf-8") as f:
                bm25_manifest = json.load(f)

            bm25_count = int(bm25_manifest.get("count", -1))

            if bm25_count != len(self.chunks):
                raise RuntimeError(
                    f"BM25/corpus count mismatch: "
                    f"{bm25_count:,} vs {len(self.chunks):,}"
                )

        except FileNotFoundError:
            raise RuntimeError(
                f"BM25 manifest not found: {settings.bm25_manifest_path}"
            )

        print(f"✅ BM25S loaded: {len(self.chunks):,}")

    # ========================================================
    # E5 Embedder
    # ========================================================

    def _load_embedder(self) -> None:
        if self.config is None:
            raise RuntimeError(
                "Inference config must be loaded before embedder."
            )

        model_id = self.config["retrieval"]["embedding_model_id"]
        device = self._resolve_compute_device()

        self.embedder = SentenceTransformer(
            model_id,
            device=device,
        )

        try:
            self.embedder.max_seq_length = 512
        except Exception:
            pass

        print(f"✅ Embedder loaded: {model_id} on {device}")

    # ========================================================
    # Cross-Encoder reranker
    # ========================================================

    def _load_reranker(self) -> None:
        if self.config is None:
            raise RuntimeError(
                "Inference config must be loaded before reranker."
            )

        model_id = self.config["retrieval"]["reranker_model_id"]
        device = self._resolve_compute_device()

        self.reranker = CrossEncoder(
            model_id,
            device=device,
        )

        print(f"✅ Reranker loaded: {model_id} on {device}")

    # ========================================================
    # Qwen generation model
    # ========================================================

    def _load_generation_model(self) -> None:
        device = self._resolve_compute_device()
        self.runtime_device = device

        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(0)
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(settings.model_path),
            trust_remote_code=True,
            fix_mistral_regex=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_path,
            **load_kwargs,
        )

        self.model.eval()
        self.model.config.use_cache = True

        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        print(
            f"✅ Qwen merged model loaded on {device} "
            f"with dtype={dtype}"
        )

    # ========================================================
    # Readiness
    # ========================================================

    def readiness(self) -> dict[str, Any]:
        if settings.is_api_only:
            return {
                "ready": bool(self.loaded),
                "corpus_loaded": False,
                "faiss_loaded": False,
                "bm25_loaded": False,
                "embedder_loaded": False,
                "reranker_loaded": False,
                "model_loaded": False,
                "corpus_chunks": None,
                "faiss_vectors": None,
                "device": "api-only",
            }

        if settings.is_retrieval_only:
            ready = all(
                [
                    bool(self.loaded),
                    bool(self.chunks),
                    self.faiss_index is not None,
                    self.bm25 is not None,
                    self.embedder is not None,
                    self.reranker is not None,
                ]
            )

            return {
                "ready": ready,
                "corpus_loaded": bool(self.chunks),
                "faiss_loaded": self.faiss_index is not None,
                "bm25_loaded": self.bm25 is not None,
                "embedder_loaded": self.embedder is not None,
                "reranker_loaded": self.reranker is not None,
                "model_loaded": False,
                "corpus_chunks": len(self.chunks) if self.chunks else None,
                "faiss_vectors": (
                    int(self.faiss_index.ntotal)
                    if self.faiss_index is not None
                    else None
                ),
                "device": self._resolve_compute_device(),
            }

        model_loaded = (
            self.model is not None
            or self.external_generation_backend
        )

        ready = all(
            [
                bool(self.loaded),
                bool(self.chunks),
                self.faiss_index is not None,
                self.bm25 is not None,
                self.embedder is not None,
                self.reranker is not None,
                self.tokenizer is not None,
                model_loaded,
            ]
        )

        model_device = None

        if self.model is not None:
            try:
                model_device = str(
                    self.model.get_input_embeddings().weight.device
                )
            except Exception:
                model_device = self.runtime_device

        return {
            "ready": ready,
            "corpus_loaded": bool(self.chunks),
            "faiss_loaded": self.faiss_index is not None,
            "bm25_loaded": self.bm25 is not None,
            "embedder_loaded": self.embedder is not None,
            "reranker_loaded": self.reranker is not None,
            "model_loaded": model_loaded,
            "corpus_chunks": len(self.chunks) if self.chunks else None,
            "faiss_vectors": (
                int(self.faiss_index.ntotal)
                if self.faiss_index is not None
                else None
            ),
            "device": model_device,
        }
