from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Protocol

from .io_utils import iter_jsonl


class RetrievalBackend(Protocol):
    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        ...


def _tokens(text: str) -> set[str]:
    value = unicodedata.normalize("NFKD", str(text).casefold())
    value = "".join(character for character in value if not unicodedata.combining(character))
    return {token for token in re.findall(r"[a-z0-9]+", value) if len(token) > 2}


class FixtureRetriever:
    """Cheap lexical adapter for tests/offline fixtures, not production parity."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = list(records)

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        ranked = sorted(
            self.records,
            key=lambda row: (
                len(query_tokens & _tokens(f"{row.get('title', '')} {row.get('text', '')}")),
                str(row.get("chunk_id") or ""),
            ),
            reverse=True,
        )
        return [dict(row, source_kind=row.get("source_kind") or "history") for row in ranked[:top_k]]


class PrecomputedRetriever:
    def __init__(self, path: str | Path):
        self.results: dict[str, list[dict[str, Any]]] = {}
        for row in iter_jsonl(path):
            query = str(row.get("query") or "").strip()
            results = row.get("results")
            if not query or not isinstance(results, list):
                raise ValueError("precomputed retrieval rows require query and results list")
            self.results[query] = results

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if query not in self.results:
            raise KeyError(f"no precomputed retrieval results for query: {query}")
        return self.results[query][:top_k]


class PrecomputedToolRetriever:
    """Offline external-tool observations keyed by exact tool name and query."""

    def __init__(self, path: str | Path):
        self.results: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in iter_jsonl(path):
            tool = str(row.get("tool") or "").strip()
            query = str(row.get("query") or "").strip()
            results = row.get("results")
            if tool not in {"search_wikipedia", "search_web"} or not query or not isinstance(results, list):
                raise ValueError(
                    "external result rows require tool=search_wikipedia|search_web, query, and results list"
                )
            self.results[(tool, query)] = results

    def search(self, tool: str, query: str, *, top_k: int) -> list[dict[str, Any]]:
        key = (tool, query)
        if key not in self.results:
            raise KeyError(f"no precomputed {tool} results for query: {query}")
        return self.results[key][:top_k]


class ProjectRetriever:
    def __init__(self, search_tool: Any, input_type: Any, service: Any):
        self._tool = search_tool
        self._input_type = input_type
        self._service = service

    @classmethod
    def load(cls, corpus_file: str | Path, *, device: str = "cpu") -> "ProjectRetriever":
        # Heavy project imports and model/index loading happen only after the user
        # explicitly selects this backend and never during package import/dry-run.
        from app.config import settings
        from app.rag.retrieval import HybridRetriever
        from app.services.rag_service import RAGService
        from app.tools.local_search import SearchHistoryInput, SearchHistoryTool

        corpus_file = Path(corpus_file).resolve()
        expected = settings.artifact_root / "corpus" / corpus_file.name
        artifact_root = corpus_file.parent.parent
        settings.app_mode = "retrieval-only"
        settings.artifact_root = artifact_root
        settings.device = device
        service = RAGService()
        service.load()
        return cls(SearchHistoryTool(HybridRetriever(service)), SearchHistoryInput, service)

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        return self._tool.run(self._input_type(query=query, top_k=top_k))

    def close(self) -> None:
        self._service.shutdown()
