from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


@dataclass
class SessionEvidenceStore:
    max_items: int = 128
    _sessions: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def add_documents(self, session_id: str, chunks: list[dict[str, Any]]) -> None:
        items = self._sessions.setdefault(session_id, {})
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            items[chunk_id] = dict(chunk)
        while len(items) > self.max_items:
            oldest = next(iter(items))
            items.pop(oldest, None)

    def add_many(self, chunks: list[dict[str, Any]], session_id: str = "default") -> None:
        self.add_documents(session_id, chunks)

    def get(self, chunk_id: str, session_id: str = "default") -> dict[str, Any] | None:
        item = self._sessions.get(session_id, {}).get(str(chunk_id))
        return dict(item) if item else None

    def all(self, session_id: str = "default") -> list[dict[str, Any]]:
        return [dict(item) for item in self._sessions.get(session_id, {}).values()]

    def search(self, query: str, top_k: int = 8, session_id: str = "default") -> list[dict[str, Any]]:
        q = _tokens(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._sessions.get(session_id, {}).values():
            text = " ".join(str(item.get(key, "")) for key in ("title", "text", "summary"))
            tokens = _tokens(text)
            score = len(q & tokens) / max(len(q), 1)
            if score > 0:
                out = dict(item)
                out["session_evidence_score"] = score
                scored.append((score, out))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def deduplicate(self, session_id: str) -> int:
        items = self._sessions.get(session_id, {})
        seen: set[str] = set()
        removed = 0
        for chunk_id, item in list(items.items()):
            fingerprint = re.sub(r"\s+", " ", str(item.get("text", "")).strip().lower())
            if fingerprint and fingerprint in seen:
                items.pop(chunk_id, None)
                removed += 1
            elif fingerprint:
                seen.add(fingerprint)
        return removed

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class RetrieveEvidenceInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)
    session_id: str = "default"


class InspectEvidenceInput(BaseModel):
    ids: list[str] = Field(default_factory=list)
    session_id: str = "default"


class RetrieveEvidenceTool:
    name = "retrieve_evidence"
    description = "Search chunks already collected in the current agent session."
    input_schema = RetrieveEvidenceInput

    def __init__(self, store: SessionEvidenceStore):
        self.store = store

    def run(self, arguments: RetrieveEvidenceInput) -> list[dict[str, Any]]:
        return self.store.search(arguments.query, arguments.top_k, arguments.session_id)


class InspectEvidenceTool:
    name = "inspect_evidence"
    description = "Return full evidence chunks by ID from the current agent session."
    input_schema = InspectEvidenceInput

    def __init__(self, store: SessionEvidenceStore):
        self.store = store

    def run(self, arguments: InspectEvidenceInput) -> list[dict[str, Any]]:
        return [
            item
            for chunk_id in arguments.ids
            if (item := self.store.get(chunk_id, arguments.session_id))
        ]
