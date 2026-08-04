from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool

    corpus_loaded: bool
    faiss_loaded: bool
    bm25_loaded: bool

    embedder_loaded: bool
    reranker_loaded: bool

    model_loaded: bool

    corpus_chunks: int | None = None
    faiss_vectors: int | None = None

    device: str | None = None


class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        examples=[
            "Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?"
        ],
    )

    debug: bool = False


class SourceItem(BaseModel):

    chunk_id: str
    title: str | None = None


class ChatResponse(BaseModel):

    answer: str

    status: str

    sources: list[SourceItem] = []

    latency_ms: float

    rewrite_used: bool = False

    debug: dict[str, Any] | None = None