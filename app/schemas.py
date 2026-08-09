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


class RetrieveRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        examples=["Khởi nghĩa Lam Sơn diễn ra trong bối cảnh nào và kết quả ra sao?"],
    )
    final_k: int = Field(default=4, ge=1, le=10)
    debug: bool = False


class RetrievalIntent(BaseModel):
    history_anchor: float
    ood_anchor: float
    margin: float
    explicit_ood: bool


class QuestionAnalysis(BaseModel):
    question: str
    facets: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    is_multi_part: bool = False


class RetrievalContextItem(BaseModel):
    chunk_id: str
    title: str | None = None
    text: str | None = None
    final_retrieval_score: float | None = None
    reranker_score: float | None = None
    rrf_score: float | None = None
    metadata_bonus: float | None = None
    metadata_hits: list[str] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    question: str
    is_ood: bool
    ood_reason: str = ""
    intent: RetrievalIntent | None = None
    analysis: QuestionAnalysis | None = None
    query_variants: list[str] = Field(default_factory=list)
    final_context: list[RetrievalContextItem] = Field(default_factory=list)
    candidates: list[RetrievalContextItem] | None = None
    tool_trace: list[str] | None = None
    max_dense: float | None = None
    context_title_diversity: float = 0.0
    latency_ms: float


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        examples=["Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?"],
    )
    final_k: int = Field(default=8, ge=1, le=10)
    debug: bool = False


class SourceItem(BaseModel):
    chunk_id: str
    title: str | None = None


class ChatResponse(BaseModel):
    answer: str
    status: str
    sources: list[SourceItem] = Field(default_factory=list)
    latency_ms: float
    rewrite_used: bool = False
    debug: dict[str, Any] | None = None