from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.agents.model_registry import CENTRAL_BASE_MODEL_ID, SHARED_BASE_MODEL_ID
from app.chat_modes import ChatMode, normalize_chat_mode


class Settings(BaseSettings):
    # ========================================================
    # Application
    # ========================================================

    app_name: str = "Vietnamese History RAG API"
    app_version: str = "1.0.0"
    app_env: str = "development"

    # api-only | retrieval-only | full
    app_mode: Literal["api-only", "retrieval-only", "full"] = "api-only"

    # ========================================================
    # Runtime
    # ========================================================

    artifact_root: Path = Path("./artifacts/vn_history_deployment")
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    llm_backend: Literal["transformers", "vllm"] = "transformers"
    shared_base_model_id: str = SHARED_BASE_MODEL_ID
    research_agent_model: str = SHARED_BASE_MODEL_ID
    research_agent_adapter_path: Path | None = None
    evidence_agent_model: str = SHARED_BASE_MODEL_ID
    evidence_agent_adapter_path: Path | None = None
    history_agent_adapter_path: Path | None = None
    vllm_base_url: str = "http://127.0.0.1:8001/v1"
    vllm_api_key: str | None = None
    history_model_path: Path | None = None
    central_agent_model_id: str = CENTRAL_BASE_MODEL_ID
    central_agent_adapter_path: Path | None = None
    central_agent_hf_cache_dir: Path | None = None
    central_agent_local_files_only: bool = False
    runtime_loading_strategy: Literal["lazy", "eager"] = "lazy"
    enable_hybrid_mode: bool = True
    enable_three_llm_mode: bool = True
    enable_central_mode: bool = True
    max_agent_steps: int = 6
    max_wikipedia_searches: int = 2
    max_web_searches: int = 3
    max_page_fetches: int = 5
    web_search_provider: str = "local-only"
    web_search_api_key: str | None = None
    default_inference_mode: ChatMode = ChatMode.HYBRID
    agent_max_tool_results: int = 10
    agent_observation_char_budget: int = 24_000
    agent_timeout_seconds: float = 120.0
    agent_enable_web: bool = True
    agent_enable_wikipedia: bool = True
    agent_enable_document_search: bool = True
    central_agent_max_action_rounds: int = 2
    central_agent_repair_max_generations: int = 1
    central_action_max_new_tokens: int = 256
    central_final_max_new_tokens: int = 1536
    central_repair_max_new_tokens: int = 1024
    central_repair_min_new_tokens: int = 192
    central_repair_token_margin: int = 96
    central_biography_max_sources: int = 4
    central_biography_min_exact_hits: int = 2
    central_analytical_retrieval_candidates: int = 10
    central_analytical_query_variants: int = 2
    central_analytical_max_sources: int = 4
    central_comparison_min_strong_sources: int = 1
    central_strong_evidence_min_chars: int = 100
    central_synthesis_char_budget: int = 12_000
    central_reranker_tail_gap_ratio: float = 0.75
    central_reranker_score_mode: Literal["raw", "probability"] = "raw"
    central_reranker_score_floor: float | None = None
    central_reranker_strong_score: float = 0.5
    central_agent_timeout_seconds: float = 180.0
    central_model_load_timeout_seconds: float = 300.0
    central_tool_timeout_seconds: float = 30.0
    central_agent_observation_char_budget: int = 12_000
    central_agent_max_tool_results: int = 6
    central_agent_enable_history: bool = True
    central_agent_enable_documents: bool = True
    central_agent_enable_wikipedia: bool = True
    central_agent_enable_web: bool = True

    # ========================================================
    # Conversation storage
    # ========================================================

    chat_database_path: Path = Path("./data/chat.sqlite3")

    # ========================================================
    # HTTP and CORS
    # ========================================================

    cors_origins_value: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        validation_alias="CORS_ORIGINS",
        exclude=True,
    )

    # ========================================================
    # Environment configuration
    # ========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator(
        "research_agent_adapter_path",
        "evidence_agent_adapter_path",
        "history_agent_adapter_path",
        "central_agent_adapter_path",
        "central_agent_hf_cache_dir",
        "history_model_path",
        mode="before",
    )
    @classmethod
    def empty_path_is_none(cls, value):
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("default_inference_mode", mode="before")
    @classmethod
    def normalize_default_inference_mode(cls, value):
        return normalize_chat_mode(value, default=ChatMode.HYBRID)

    # ========================================================
    # Parsed configuration
    # ========================================================

    @property
    def cors_origins(self) -> list[str]:
        origins = [
            origin.strip().rstrip("/")
            for origin in self.cors_origins_value.split(",")
            if origin.strip()
        ]

        return list(dict.fromkeys(origins))

    # ========================================================
    # Runtime mode helpers
    # ========================================================

    @property
    def is_api_only(self) -> bool:
        return self.app_mode == "api-only"

    @property
    def is_retrieval_only(self) -> bool:
        return self.app_mode == "retrieval-only"

    @property
    def is_full(self) -> bool:
        return self.app_mode == "full"

    @property
    def should_load_retrieval(self) -> bool:
        return self.app_mode in {"retrieval-only", "full"}

    @property
    def should_load_model(self) -> bool:
        return self.app_mode == "full"

    @property
    def uses_shared_backend(self) -> bool:
        return self.should_load_model and self.llm_backend in {"transformers", "vllm"}

    # ========================================================
    # Deployment artifact paths
    # ========================================================

    @property
    def model_path(self) -> Path:
        if self.history_model_path is not None:
            return self.history_model_path
        return self.artifact_root / "history_answerer" / "model"

    @property
    def corpus_path(self) -> Path:
        return self.artifact_root / "corpus" / "vn_history_rag_chunks_enriched.jsonl"

    @property
    def faiss_path(self) -> Path:
        return self.artifact_root / "retrieval" / "faiss" / "chunks.index"

    @property
    def faiss_manifest_path(self) -> Path:
        return self.artifact_root / "retrieval" / "faiss" / "manifest.json"

    @property
    def bm25_path(self) -> Path:
        return self.artifact_root / "retrieval" / "bm25s_index"

    @property
    def bm25_manifest_path(self) -> Path:
        return self.artifact_root / "retrieval" / "bm25s_index" / "phase9_manifest.json"

    @property
    def inference_config_path(self) -> Path:
        return self.artifact_root / "config" / "inference_config.json"

    @property
    def manifest_path(self) -> Path:
        return self.artifact_root / "manifest.json"

    @property
    def central_adapter_path(self) -> Path | None:
        return self.central_agent_adapter_path

    # ========================================================
    # Validation helpers
    # ========================================================

    def required_retrieval_paths(self) -> list[Path]:
        return [
            self.corpus_path,
            self.faiss_path,
            self.faiss_manifest_path,
            self.bm25_path,
            self.bm25_manifest_path,
            self.inference_config_path,
            self.manifest_path,
        ]

    def required_full_paths(self) -> list[Path]:
        paths = [*self.required_retrieval_paths()]
        if self.llm_backend == "transformers":
            if self.enable_hybrid_mode or self.enable_three_llm_mode:
                if self.research_agent_adapter_path is not None and self.enable_three_llm_mode:
                    paths.append(self.research_agent_adapter_path)
                if self.evidence_agent_adapter_path is not None and self.enable_three_llm_mode:
                    paths.append(self.evidence_agent_adapter_path)
                if self.history_agent_adapter_path is not None:
                    paths.append(self.history_agent_adapter_path)
            if self.enable_central_mode and self.central_adapter_path is not None:
                paths.append(self.central_adapter_path)
        return paths


settings = Settings()
