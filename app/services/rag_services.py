import json
import time
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

        self.started_at = None

        self.config: dict[str, Any] | None = None
        self.manifest: dict[str, Any] | None = None

        self.chunks: list[dict[str, Any]] = []
        self.chunk_by_id: dict[str, dict[str, Any]] = {}

        self.faiss_index = None
        self.bm25 = None

        self.embedder = None
        self.reranker = None

        self.tokenizer = None
        self.model = None

        self.loaded = False


    # ========================================================
    # Public lifecycle
    # ========================================================

    def load(self):

        if self.loaded:
            return

        started = time.perf_counter()

        print("=" * 70)
        print("Loading Vietnamese History RAG service")
        print("=" * 70)

        self._validate_artifacts()

        self._load_config()
        self._load_corpus()

        self._load_faiss()
        self._load_bm25()

        self._load_embedder()
        self._load_reranker()

        if settings.load_model_on_startup:
            self._load_generation_model()

        self.loaded = True

        self.started_at = time.time()

        print(
            f"✅ RAG service ready in "
            f"{time.perf_counter() - started:.2f}s"
        )


    def shutdown(self):

        print("Shutting down RAG service...")

        self.model = None
        self.tokenizer = None

        self.embedder = None
        self.reranker = None

        self.faiss_index = None
        self.bm25 = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.loaded = False


    # ========================================================
    # Artifact validation
    # ========================================================

    def _validate_artifacts(self):

        required = [
            settings.model_path,
            settings.corpus_path,
            settings.faiss_path,
            settings.bm25_path,
            settings.inference_config_path,
            settings.manifest_path,
        ]

        missing = [
            str(path)
            for path in required
            if not Path(path).exists()
        ]

        if missing:

            raise FileNotFoundError(
                "Missing deployment artifacts:\n"
                + "\n".join(missing)
            )


    # ========================================================
    # Config
    # ========================================================

    def _load_config(self):

        with settings.inference_config_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            self.config = json.load(f)

        with settings.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            self.manifest = json.load(f)

        print("✅ Config + manifest loaded")


    # ========================================================
    # Corpus
    # ========================================================

    def _load_corpus(self):

        chunks = []

        with settings.corpus_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line_no, line in enumerate(f, 1):

                line = line.strip()

                if not line:
                    continue

                try:

                    row = json.loads(line)

                except json.JSONDecodeError as exc:

                    raise RuntimeError(
                        f"Invalid JSON at corpus line "
                        f"{line_no}"
                    ) from exc

                chunks.append(row)


        self.chunks = chunks

        self.chunk_by_id = {
            str(c["chunk_id"]): c
            for c in chunks
        }


        if len(self.chunks) != 58_603:

            raise RuntimeError(
                f"Corpus mismatch: "
                f"{len(self.chunks)} != 58603"
            )


        if (
            len(self.chunk_by_id)
            != len(self.chunks)
        ):

            raise RuntimeError(
                "Duplicate chunk_id detected"
            )


        print(
            f"✅ Corpus loaded: "
            f"{len(self.chunks):,}"
        )


    # ========================================================
    # FAISS
    # ========================================================

    def _load_faiss(self):

        self.faiss_index = faiss.read_index(
            str(
                settings.faiss_path
            )
        )


        if (
            self.faiss_index.ntotal
            != len(self.chunks)
        ):

            raise RuntimeError(
                "FAISS/corpus count mismatch: "
                f"{self.faiss_index.ntotal} "
                f"vs {len(self.chunks)}"
            )


        print(
            f"✅ FAISS loaded: "
            f"{self.faiss_index.ntotal:,}"
        )


    # ========================================================
    # BM25S
    # ========================================================

    def _load_bm25(self):

        self.bm25 = bm25s.BM25.load(
            str(
                settings.bm25_path
            ),
            mmap=True,
            load_corpus=False,
        )

        print("✅ BM25S loaded")


    # ========================================================
    # E5
    # ========================================================

    def _load_embedder(self):

        model_id = (
            self.config[
                "retrieval"
            ][
                "embedding_model_id"
            ]
        )


        embed_device = (
            "cuda"
            if (
                settings.device == "cuda"
                and torch.cuda.is_available()
            )
            else "cpu"
        )


        self.embedder = SentenceTransformer(
            model_id,
            device=embed_device,
        )


        try:
            self.embedder.max_seq_length = 512
        except Exception:
            pass


        print(
            f"✅ Embedder loaded: "
            f"{model_id} "
            f"on {embed_device}"
        )


    # ========================================================
    # Cross Encoder
    # ========================================================

    def _load_reranker(self):

        model_id = (
            self.config[
                "retrieval"
            ][
                "reranker_model_id"
            ]
        )


        device = (
            "cuda"
            if (
                settings.device == "cuda"
                and torch.cuda.is_available()
            )
            else "cpu"
        )


        self.reranker = CrossEncoder(
            model_id,
            device=device,
        )


        print(
            f"✅ Reranker loaded: "
            f"{model_id} "
            f"on {device}"
        )


    # ========================================================
    # Qwen
    # ========================================================

    def _load_generation_model(self):

        device = (
            "cuda"
            if (
                settings.device == "cuda"
                and torch.cuda.is_available()
            )
            else "cpu"
        )


        if device == "cuda":

            major, _ = (
                torch.cuda.get_device_capability(0)
            )

            dtype = (
                torch.bfloat16
                if major >= 8
                else torch.float16
            )

        else:

            dtype = torch.float32


        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                settings.model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        )


        if self.tokenizer.pad_token_id is None:

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )


        self.tokenizer.padding_side = "left"


        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }


        if device == "cuda":

            load_kwargs[
                "device_map"
            ] = {
                "": 0
            }


        self.model = (
            AutoModelForCausalLM.from_pretrained(
                settings.model_path,
                **load_kwargs,
            )
        )


        self.model.eval()
        self.model.config.use_cache = True


        print(
            f"✅ Qwen merged model loaded "
            f"on {device}"
        )


    # ========================================================
    # Readiness
    # ========================================================

    def readiness(self):

        model_ready = (
            self.model is not None
            if settings.load_model_on_startup
            else True
        )


        ready = all([
            bool(self.chunks),
            self.faiss_index is not None,
            self.bm25 is not None,
            self.embedder is not None,
            self.reranker is not None,
            model_ready,
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

                pass


        return {

            "ready":
                ready,

            "corpus_loaded":
                bool(self.chunks),

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
                len(self.chunks)
                if self.chunks
                else None,

            "faiss_vectors":
                int(
                    self.faiss_index.ntotal
                )
                if self.faiss_index
                is not None
                else None,

            "device":
                model_device,
        }