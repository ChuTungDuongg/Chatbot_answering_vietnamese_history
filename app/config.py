from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Vietnamese History RAG API"
    app_version: str = "1.0.0"
    app_env: str = "development"

    artifact_root: Path = Path(
        "./artifacts/vn_history_deployment"
    )

    device: str = "cuda"

    load_model_on_startup: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def model_path(self) -> Path:
        return (
            self.artifact_root
            / "model"
            / "qwen2_5_3b_vnhistory_stage12_merged"
        )

    @property
    def corpus_path(self) -> Path:
        return (
            self.artifact_root
            / "corpus"
            / "vn_history_rag_chunks_enriched.jsonl"
        )

    @property
    def faiss_path(self) -> Path:
        return (
            self.artifact_root
            / "retrieval"
            / "faiss"
            / "chunks.index"
        )

    @property
    def bm25_path(self) -> Path:
        return (
            self.artifact_root
            / "retrieval"
            / "bm25s_index"
        )

    @property
    def inference_config_path(self) -> Path:
        return (
            self.artifact_root
            / "config"
            / "inference_config.json"
        )

    @property
    def manifest_path(self) -> Path:
        return (
            self.artifact_root
            / "manifest.json"
        )


settings = Settings()