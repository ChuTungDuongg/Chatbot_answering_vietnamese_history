"""Configuration for the reproducible vanilla-model baseline."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.chat_modes import ChatMode, normalize_chat_mode


HYBRID_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
CENTRAL_MODEL_ID = "Qwen/Qwen3-8B"


class Settings(BaseSettings):
    app_name: str = "Vietnamese History LLM API"
    app_version: str = "2.0.0"
    app_env: str = "development"
    app_mode: Literal["api-only", "retrieval-only", "full"] = "api-only"
    artifact_root: Path = Path("./artifacts/vn_history_deployment")
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    hybrid_model_id: str = HYBRID_MODEL_ID
    central_model_id: str = CENTRAL_MODEL_ID
    hybrid_model_revision: str | None = None
    central_model_revision: str | None = None
    model_cache_dir: Path | None = None
    model_local_files_only: bool = False
    runtime_loading_strategy: Literal["lazy", "eager"] = "lazy"
    do_sample: bool = False
    enable_thinking: bool = False
    hybrid_max_new_tokens: int = Field(default=768, ge=1)
    central_action_max_new_tokens: int = Field(default=256, ge=1)
    central_final_max_new_tokens: int = Field(default=1536, ge=1)
    central_max_action_rounds: int = Field(default=2, ge=0, le=8)
    enable_hybrid_mode: bool = True
    enable_central_mode: bool = True
    default_inference_mode: ChatMode = ChatMode.HYBRID
    central_enable_documents: bool = True
    central_enable_wikipedia: bool = True
    central_enable_web: bool = False
    web_search_provider: str = "local-only"
    web_search_api_key: str | None = None
    chat_database_path: Path = Path("./data/chat.sqlite3")
    cors_origins_value: str = Field(default="http://localhost:5173,http://127.0.0.1:5173", validation_alias="CORS_ORIGINS", exclude=True)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("default_inference_mode", mode="before")
    @classmethod
    def normalize_default_inference_mode(cls, value):
        return normalize_chat_mode(value, default=ChatMode.HYBRID)

    @field_validator("hybrid_model_revision", "central_model_revision", "model_cache_dir", mode="before")
    @classmethod
    def empty_is_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("hybrid_model_id")
    @classmethod
    def require_hybrid_model(cls, value: str) -> str:
        if value != HYBRID_MODEL_ID:
            raise ValueError(f"Hybrid baseline requires {HYBRID_MODEL_ID}")
        return value

    @field_validator("central_model_id")
    @classmethod
    def require_central_model(cls, value: str) -> str:
        if value != CENTRAL_MODEL_ID:
            raise ValueError(f"Central baseline requires {CENTRAL_MODEL_ID}")
        return value

    @property
    def cors_origins(self) -> list[str]:
        return list(dict.fromkeys(x.strip().rstrip("/") for x in self.cors_origins_value.split(",") if x.strip()))

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
        return not self.is_api_only

    @property
    def should_load_model(self) -> bool:
        return self.is_full

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
        return self.bm25_path / "phase9_manifest.json"

    @property
    def inference_config_path(self) -> Path:
        return self.artifact_root / "config" / "inference_config.json"

    @property
    def manifest_path(self) -> Path:
        return self.artifact_root / "manifest.json"

    def required_retrieval_paths(self) -> list[Path]:
        return [self.corpus_path, self.faiss_path, self.faiss_manifest_path,
                self.bm25_path, self.bm25_manifest_path, self.inference_config_path, self.manifest_path]


settings = Settings()
