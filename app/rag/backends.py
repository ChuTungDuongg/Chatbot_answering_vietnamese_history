"""Request-local backend names and safe retrieval failures."""

from typing import Literal

RetrievalBackend = Literal["faiss", "qdrant"]


class DenseBackendError(RuntimeError):
    """Safe to surface over HTTP/SSE; never contains connection credentials."""


class DenseBackendUnavailable(DenseBackendError):
    pass


class QdrantSearchError(DenseBackendError):
    def __init__(self):
        super().__init__("Qdrant hiện không khả dụng. Hãy chọn FAISS hoặc thử lại.")
