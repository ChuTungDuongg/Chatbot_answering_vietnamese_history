from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.chat_modes import ChatMode, normalize_chat_mode


MessageRole = Literal["user", "assistant"]
InferenceMode = ChatMode
AttachmentStatus = Literal[
    "processing",
    "ready",
    "failed",
]


# ============================================================
# System
# ============================================================

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


# ============================================================
# Retrieval
# ============================================================

class RetrieveRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
    )
    final_k: int = Field(default=6, ge=1, le=10)
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
    source_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    display_index: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    comparison_target: str | None = Field(default=None, exclude_if=lambda value: value is None)
    comparison_targets: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    viewpoint_sensitive: bool = Field(default=False, exclude_if=lambda value: not value)
    url: str | None = Field(default=None, exclude_if=lambda value: value is None)
    title: str | None = None
    text: str | None = None

    source_kind: Literal[
        "history",
        "attachment",
        "wikipedia",
        "web",
    ] = "history"

    attachment_id: UUID | None = None
    page_number: int | None = None

    final_retrieval_score: float | None = None
    reranker_score: float | None = None
    rrf_score: float | None = None
    metadata_bonus: float | None = None
    metadata_hits: list[str] = Field(
        default_factory=list,
    )


class RetrieveResponse(BaseModel):
    question: str
    is_ood: bool
    ood_reason: str = ""

    intent: RetrievalIntent | None = None
    analysis: QuestionAnalysis | None = None

    query_variants: list[str] = Field(
        default_factory=list,
    )

    final_context: list[RetrievalContextItem] = Field(
        default_factory=list,
    )

    candidates: list[RetrievalContextItem] | None = None
    tool_trace: list[str] | None = None

    max_dense: float | None = None
    context_title_diversity: float = 0.0
    latency_ms: float


# ============================================================
# Sources
# ============================================================

class SourceItem(BaseModel):
    chunk_id: str
    source_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    display_index: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    comparison_target: str | None = Field(default=None, exclude_if=lambda value: value is None)
    comparison_targets: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    viewpoint_sensitive: bool = Field(default=False, exclude_if=lambda value: not value)
    title: str | None = None

    source_kind: Literal[
        "history",
        "attachment",
        "wikipedia",
        "web",
    ] = "history"

    attachment_id: UUID | None = None
    page_number: int | None = None
    url: str | None = Field(default=None, exclude_if=lambda value: value is None)


# ============================================================
# Conversations
# ============================================================

class ConversationCreate(BaseModel):
    title: str | None = Field(
        default=None,
        max_length=120,
    )


class ConversationUpdate(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=120,
    )


class ConversationSummary(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    attachment_count: int = 0


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary] = Field(
        default_factory=list,
    )


# ============================================================
# Messages
# ============================================================

class MessageItem(BaseModel):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str

    sources: list[SourceItem] = Field(
        default_factory=list,
    )
    debug_trace: dict[str, Any] | None = None

    status: str = "done"
    created_at: datetime

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_null_sources(cls, value: Any) -> Any:
        return [] if value is None else value


# ============================================================
# Attachments
# ============================================================

class AttachmentItem(BaseModel):
    id: UUID
    conversation_id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    status: AttachmentStatus
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime


class AttachmentUploadResponse(BaseModel):
    attachment: AttachmentItem


# ============================================================
# Conversation detail
# ============================================================

class ConversationDetailResponse(BaseModel):
    conversation: ConversationSummary

    messages: list[MessageItem] = Field(
        default_factory=list,
    )

    attachments: list[AttachmentItem] = Field(
        default_factory=list,
    )


# ============================================================
# Chat
# ============================================================

class ChatRequest(BaseModel):
    conversation_id: UUID

    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
    )

    final_k: int = Field(
        default=6,
        ge=1,
        le=10,
    )

    mode: InferenceMode | None = None

    debug: bool = False

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: Any) -> ChatMode | None:
        return None if value is None else normalize_chat_mode(value)


class ChatResponse(BaseModel):
    conversation_id: UUID
    message_id: UUID

    answer: str
    status: str
    mode: InferenceMode

    sources: list[SourceItem] = Field(
        default_factory=list,
    )

    latency_ms: float
    rewrite_used: bool = False
    debug: dict[str, Any] | None = None
