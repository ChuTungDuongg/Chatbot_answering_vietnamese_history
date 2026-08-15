from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ========================================================
    # Deployment artifact paths
    # ========================================================

    @property
    def model_path(self) -> Path:
        return self.artifact_root / "model" / "qwen2_5_3b_vnhistory_stage12_merged"

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
        return [*self.required_retrieval_paths(), self.model_path]


settings = Settings()