import json
import time
import gc
from pathlib import Path
from typing import Any

import bm25s
import faiss
import torch

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from app.config import settings


class RAGService:

    def __init__(self):

        # ====================================================
        # Runtime state
        # ====================================================

        self.started_at: float | None = None
        self.loaded: bool = False

        self.runtime_mode: str = settings.app_mode
        self.runtime_device: str | None = None

        # ====================================================
        # Deployment config / manifest
        # ====================================================

        self.config: dict[str, Any] | None = None
        self.manifest: dict[str, Any] | None = None

        # ====================================================
        # Corpus
        # ====================================================

        self.chunks: list[dict[str, Any]] = []

        self.chunk_by_id: dict[
            str,
            dict[str, Any]
        ] = {}

        # ====================================================
        # Retrieval
        # ====================================================

        self.faiss_index = None
        self.bm25 = None

        self.embedder = None
        self.reranker = None

        # ====================================================
        # Generation
        # ====================================================

        self.tokenizer = None
        self.model = None


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

        print(
            f"Runtime mode : {settings.app_mode}"
        )

        print(
            f"Target device: {settings.device}"
        )

        # ====================================================
        # MODE 1 — API ONLY
        # ====================================================
        #
        # Local FastAPI development.
        #
        # Không cần:
        # - Google Drive artifacts
        # - FAISS
        # - BM25
        # - E5
        # - reranker
        # - Qwen
        #
        # Chỉ dùng để dev:
        # /health
        # /ready
        # Swagger
        # schema
        # middleware
        # streaming contract
        # ====================================================

        if settings.is_api_only:

            self.runtime_device = "api-only"

            self.loaded = True
            self.started_at = time.time()

            print(
                "✅ API-only mode: "
                "skipping all AI/RAG artifacts."
            )

            print(
                f"✅ API service ready in "
                f"{time.perf_counter() - started:.2f}s"
            )

            return


        # ====================================================
        # MODE 2 / 3
        # retrieval-only OR full
        # ====================================================

        if settings.should_load_retrieval:

            self._validate_artifacts()

            self._load_config()
            self._load_corpus()

            self._load_faiss()
            self._load_bm25()

            self._load_embedder()
            self._load_reranker()


        # ====================================================
        # MODE 3 — FULL
        # ====================================================

        if settings.should_load_model:

            self._load_generation_model()


        self.loaded = True
        self.started_at = time.time()

        print(
            f"✅ RAG service ready in "
            f"{time.perf_counter() - started:.2f}s"
        )


    def shutdown(self) -> None:

        print(
            "Shutting down Vietnamese History "
            "RAG service..."
        )

        # ====================================================
        # Generation
        # ====================================================

        self.model = None
        self.tokenizer = None

        # ====================================================
        # Retrieval models
        # ====================================================

        self.embedder = None
        self.reranker = None

        # ====================================================
        # Retrieval indexes
        # ====================================================

        self.faiss_index = None
        self.bm25 = None

        # ====================================================
        # Corpus
        # ====================================================

        self.chunks = []
        self.chunk_by_id = {}

        # ====================================================
        # Config
        # ====================================================

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

        print(
            "✅ RAG service shutdown complete."
        )


    # ========================================================
    # Device helpers
    # ========================================================

    def _resolve_compute_device(
        self,
    ) -> str:

        requested = (
            settings.device
            .strip()
            .lower()
        )

        # ----------------------------------------------------
        # Explicit CUDA request
        # ----------------------------------------------------

        if requested == "cuda":

            if torch.cuda.is_available():
                return "cuda"

            # Full inference không nên âm thầm load Qwen 3B
            # lên CPU vì cấu hình cloud bị sai.
            if settings.is_full:

                raise RuntimeError(
                    "APP_MODE=full và DEVICE=cuda "
                    "nhưng PyTorch không phát hiện CUDA. "
                    "Hãy kiểm tra GPU/CUDA environment."
                )

            print(
                "⚠️ CUDA unavailable → "
                "falling back to CPU."
            )

            return "cpu"


        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        if requested == "cpu":
            return "cpu"


        raise ValueError(
            f"Unsupported DEVICE={settings.device!r}. "
            "Expected 'cpu' or 'cuda'."
        )


    # ========================================================
    # Artifact validation
    # ========================================================

    def _validate_artifacts(
        self,
    ) -> None:

        # ----------------------------------------------------
        # retrieval-only
        # ----------------------------------------------------

        if settings.is_retrieval_only:

            required = (
                settings
                .required_retrieval_paths()
            )

        # ----------------------------------------------------
        # full
        # ----------------------------------------------------

        elif settings.is_full:

            required = (
                settings
                .required_full_paths()
            )

        else:

            # api-only không bao giờ tới đây.
            return


        missing = [

            str(path)

            for path
            in required

            if not Path(
                path
            ).exists()
        ]


        if missing:

            raise FileNotFoundError(

                "Missing deployment artifacts "
                f"for APP_MODE={settings.app_mode}:\n"
                +
                "\n".join(
                    missing
                )
            )


        print(
            "✅ Deployment artifacts found."
        )


    # ========================================================
    # Deployment config
    # ========================================================

    def _load_config(
        self,
    ) -> None:

        with (
            settings
            .inference_config_path
            .open(
                "r",
                encoding="utf-8",
            )
        ) as f:

            self.config = json.load(
                f
            )


        with (
            settings
            .manifest_path
            .open(
                "r",
                encoding="utf-8",
            )
        ) as f:

            self.manifest = json.load(
                f
            )


        print(
            "✅ Config + manifest loaded"
        )


    # ========================================================
    # Expected corpus count
    # ========================================================

    def _expected_corpus_count(
        self,
    ) -> int:

        # Phase 10 manifest là source of truth.
        try:

            return int(
                self.manifest[
                    "corpus"
                ][
                    "count"
                ]
            )

        except Exception:

            # Fallback theo Phase 9 hiện tại.
            return 58_603


    # ========================================================
    # Corpus
    # ========================================================

    def _load_corpus(
        self,
    ) -> None:

        chunks: list[
            dict[str, Any]
        ] = []


        with (
            settings
            .corpus_path
            .open(
                "r",
                encoding="utf-8",
            )
        ) as f:

            for line_no, line in enumerate(
                f,
                1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    row = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:

                    raise RuntimeError(

                        "Invalid JSON in corpus at "
                        f"line {line_no}"

                    ) from exc


                if "chunk_id" not in row:

                    raise RuntimeError(
                        "Corpus row missing chunk_id "
                        f"at line {line_no}"
                    )


                chunks.append(
                    row
                )


        self.chunks = chunks


        self.chunk_by_id = {

            str(
                chunk[
                    "chunk_id"
                ]
            ):
                chunk

            for chunk
            in chunks
        }


        expected = (
            self._expected_corpus_count()
        )


        if len(
            self.chunks
        ) != expected:

            raise RuntimeError(

                "Corpus count mismatch: "
                f"{len(self.chunks):,} "
                f"!= {expected:,}"
            )


        if (
            len(
                self.chunk_by_id
            )
            !=
            len(
                self.chunks
            )
        ):

            raise RuntimeError(
                "Duplicate chunk_id detected "
                "in deployment corpus."
            )


        print(
            f"✅ Corpus loaded: "
            f"{len(self.chunks):,}"
        )


    # ========================================================
    # FAISS
    # ========================================================

    def _load_faiss(
        self,
    ) -> None:

        self.faiss_index = (
            faiss.read_index(
                str(
                    settings.faiss_path
                )
            )
        )


        expected = len(
            self.chunks
        )


        if (
            self.faiss_index.ntotal
            != expected
        ):

            raise RuntimeError(

                "FAISS/corpus count mismatch: "
                f"{self.faiss_index.ntotal:,} "
                f"vs {expected:,}"
            )


        print(
            f"✅ FAISS loaded: "
            f"{self.faiss_index.ntotal:,}"
        )


    # ========================================================
    # BM25S
    # ========================================================

    def _load_bm25(
        self,
    ) -> None:

        self.bm25 = (
            bm25s.BM25.load(

                str(
                    settings.bm25_path
                ),

                mmap=True,

                load_corpus=False,
            )
        )


        # ----------------------------------------------------
        # Validate BM25 manifest count
        # ----------------------------------------------------

        try:

            with (
                settings
                .bm25_manifest_path
                .open(
                    "r",
                    encoding="utf-8",
                )
            ) as f:

                bm25_manifest = (
                    json.load(
                        f
                    )
                )


            bm25_count = int(
                bm25_manifest.get(
                    "count",
                    -1,
                )
            )


            if (
                bm25_count
                !=
                len(
                    self.chunks
                )
            ):

                raise RuntimeError(

                    "BM25/corpus count mismatch: "
                    f"{bm25_count:,} "
                    f"vs "
                    f"{len(self.chunks):,}"
                )


        except FileNotFoundError:

            raise RuntimeError(
                "BM25 manifest not found: "
                f"{settings.bm25_manifest_path}"
            )


        print(
            f"✅ BM25S loaded: "
            f"{len(self.chunks):,}"
        )


    # ========================================================
    # E5 Embedder
    # ========================================================

    def _load_embedder(
        self,
    ) -> None:

        if self.config is None:

            raise RuntimeError(
                "Inference config must be "
                "loaded before embedder."
            )


        model_id = (
            self.config[
                "retrieval"
            ][
                "embedding_model_id"
            ]
        )


        device = (
            self._resolve_compute_device()
        )


        self.embedder = (
            SentenceTransformer(
                model_id,
                device=device,
            )
        )


        try:

            self.embedder.max_seq_length = 512

        except Exception:
            pass


        print(
            f"✅ Embedder loaded: "
            f"{model_id} on {device}"
        )


    # ========================================================
    # Cross-Encoder reranker
    # ========================================================

    def _load_reranker(
        self,
    ) -> None:

        if self.config is None:

            raise RuntimeError(
                "Inference config must be "
                "loaded before reranker."
            )


        model_id = (
            self.config[
                "retrieval"
            ][
                "reranker_model_id"
            ]
        )


        device = (
            self._resolve_compute_device()
        )


        self.reranker = (
            CrossEncoder(
                model_id,
                device=device,
            )
        )


        print(
            f"✅ Reranker loaded: "
            f"{model_id} on {device}"
        )


    # ========================================================
    # Qwen generation model
    # ========================================================

    def _load_generation_model(
        self,
    ) -> None:

        device = (
            self._resolve_compute_device()
        )


        self.runtime_device = device


        # ----------------------------------------------------
        # dtype
        # ----------------------------------------------------

        if device == "cuda":

            major, _ = (
                torch.cuda
                .get_device_capability(
                    0
                )
            )


            dtype = (
                torch.bfloat16
                if major >= 8
                else torch.float16
            )

        else:

            dtype = torch.float32


        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(settings.model_path),
            trust_remote_code=True,
            fix_mistral_regex=True,
        )


        if (
            self.tokenizer
            .pad_token_id
            is None
        ):

            self.tokenizer.pad_token = (
                self.tokenizer
                .eos_token
            )


        self.tokenizer.padding_side = (
            "left"
        )


        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        load_kwargs = {

            "dtype":
                dtype,

            "trust_remote_code":
                True,

            "low_cpu_mem_usage":
                True,
        }


        if device == "cuda":

            load_kwargs[
                "device_map"
            ] = {
                "": 0
            }


        self.model = (
            AutoModelForCausalLM
            .from_pretrained(

                settings.model_path,

                **load_kwargs,
            )
        )


        self.model.eval()

        self.model.config.use_cache = (
            True
        )


        if (
            self.tokenizer
            .pad_token_id
            is not None
        ):

            self.model.config.pad_token_id = (
                self.tokenizer
                .pad_token_id
            )


        print(
            "✅ Qwen merged model loaded "
            f"on {device} "
            f"with dtype={dtype}"
        )


    # ========================================================
    # Readiness
    # ========================================================

    def readiness(
        self,
    ) -> dict[str, Any]:

        # ====================================================
        # API-only
        # ====================================================

        if settings.is_api_only:

            return {

                "ready":
                    bool(
                        self.loaded
                    ),

                "corpus_loaded":
                    False,

                "faiss_loaded":
                    False,

                "bm25_loaded":
                    False,

                "embedder_loaded":
                    False,

                "reranker_loaded":
                    False,

                "model_loaded":
                    False,

                "corpus_chunks":
                    None,

                "faiss_vectors":
                    None,

                "device":
                    "api-only",
            }


        # ====================================================
        # retrieval-only
        # ====================================================

        if settings.is_retrieval_only:

            ready = all([

                bool(
                    self.loaded
                ),

                bool(
                    self.chunks
                ),

                self.faiss_index
                is not None,

                self.bm25
                is not None,

                self.embedder
                is not None,

                self.reranker
                is not None,
            ])


            return {

                "ready":
                    ready,

                "corpus_loaded":
                    bool(
                        self.chunks
                    ),

                "faiss_loaded":
                    self.faiss_index
                    is not None,

                "bm25_loaded":
                    self.bm25
                    is not None,

                "embedder_loaded":
                    self.embedder
                    is not None,

                "reranker_loaded":
                    self.reranker
                    is not None,

                "model_loaded":
                    False,

                "corpus_chunks":
                    (
                        len(
                            self.chunks
                        )
                        if self.chunks
                        else None
                    ),

                "faiss_vectors":
                    (
                        int(
                            self.faiss_index
                            .ntotal
                        )
                        if (
                            self.faiss_index
                            is not None
                        )
                        else None
                    ),

                "device":
                    (
                        self
                        ._resolve_compute_device()
                    ),
            }


        # ====================================================
        # Full mode
        # ====================================================

        ready = all([

            bool(
                self.loaded
            ),

            bool(
                self.chunks
            ),

            self.faiss_index
            is not None,

            self.bm25
            is not None,

            self.embedder
            is not None,

            self.reranker
            is not None,

            self.tokenizer
            is not None,

            self.model
            is not None,
        ])


        model_device = None


        if self.model is not None:

            try:

                model_device = str(

                    self.model
                    .get_input_embeddings()
                    .weight
                    .device
                )

            except Exception:

                model_device = (
                    self.runtime_device
                )


        return {

            "ready":
                ready,

            "corpus_loaded":
                bool(
                    self.chunks
                ),

            "faiss_loaded":
                self.faiss_index
                is not None,

            "bm25_loaded":
                self.bm25
                is not None,

            "embedder_loaded":
                self.embedder
                is not None,

            "reranker_loaded":
                self.reranker
                is not None,

            "model_loaded":
                self.model
                is not None,

            "corpus_chunks":
                (
                    len(
                        self.chunks
                    )
                    if self.chunks
                    else None
                ),

            "faiss_vectors":
                (
                    int(
                        self.faiss_index
                        .ntotal
                    )
                    if (
                        self.faiss_index
                        is not None
                    )
                    else None
                ),

            "device":
                model_device,
        }