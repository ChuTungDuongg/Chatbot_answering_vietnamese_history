"""Retrieval boundary shared by Hybrid and Central."""

from typing import Any, Protocol


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> dict[str, Any]: ...
