from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


def normalized_question(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def first_user_question(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"])
    return ""


@dataclass(frozen=True)
class DedupResult:
    rows: list[dict[str, Any]]
    rejected: list[dict[str, Any]]


def deduplicate(rows: Iterable[dict[str, Any]]) -> DedupResult:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for row in rows:
        row_id = str(row.get("id") or "")
        question = normalized_question(first_user_question(row))
        reason = ""
        if not row_id:
            reason = "missing trajectory id"
        elif row_id in seen_ids:
            reason = f"duplicate trajectory id: {row_id}"
        elif question and question in seen_questions:
            reason = "duplicate normalized user question"
        if reason:
            rejected.append({"id": row_id or None, "reason": reason, "record": row})
            continue
        seen_ids.add(row_id)
        if question:
            seen_questions.add(question)
        kept.append(row)
    return DedupResult(rows=kept, rejected=rejected)
