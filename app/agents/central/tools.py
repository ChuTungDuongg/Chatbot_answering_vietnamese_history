from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from app.tools.registry import ToolRegistry


EXTERNAL_TOOLS = {
    "search_wikipedia", "fetch_wikipedia_page", "search_web", "fetch_web_page",
}


def qwen_tool_schemas(registry: ToolRegistry, allowed_names: Iterable[str]) -> list[dict[str, Any]]:
    allowed = set(allowed_names)
    schemas: list[dict[str, Any]] = []
    for item in registry.describe():
        if item["name"] not in allowed:
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": item["input_schema"],
            },
        })
    return schemas


def bounded_tool_arguments(name: str, arguments: dict[str, Any], *, max_results: int) -> dict[str, Any]:
    bounded = dict(arguments)
    if name.startswith("search_"):
        try:
            requested_top_k = int(bounded.get("top_k") or max_results)
        except (TypeError, ValueError):
            requested_top_k = max_results
        bounded["top_k"] = min(max(1, requested_top_k), max_results)
    if name in {"fetch_wikipedia_page", "fetch_web_page"}:
        try:
            requested_chars = int(bounded.get("max_chars") or 6000)
        except (TypeError, ValueError):
            requested_chars = 6000
        bounded["max_chars"] = min(max(500, requested_chars), 8000)
    return bounded


def _stable_source_id(tool_name: str, row: dict[str, Any], index: int) -> str:
    existing = str(row.get("chunk_id") or row.get("evidence_id") or "").strip()
    if existing:
        return existing
    seed = "\n".join((tool_name, str(row.get("url") or ""), str(row.get("title") or ""), str(index)))
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    prefix = "web" if "web" in tool_name else "source"
    return f"{prefix}_{digest}"


def normalize_tool_result(
    tool_name: str,
    result: Any,
    *,
    max_results: int,
    char_budget: int,
) -> tuple[str, list[dict[str, Any]]]:
    rows = result if isinstance(result, list) else ([] if result is None else [result])
    compact: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    remaining = max(0, char_budget)
    for index, raw in enumerate(rows[:max_results]):
        row = dict(raw) if isinstance(raw, dict) else {"text": str(raw)}
        source_id = _stable_source_id(tool_name, row, index)
        text = str(row.get("text") or row.get("content") or row.get("snippet") or "").strip()
        text = " ".join(text.split())
        fixed_cost = len(source_id) + len(str(row.get("title") or "")) + 120
        text_budget = max(0, min(3200, remaining - fixed_cost))
        text = text[:text_budget]
        item = {
            "source_id": source_id,
            "title": row.get("title"),
            "source_kind": row.get("source_kind") or (
                "wikipedia" if "wikipedia" in tool_name else "web" if "web" in tool_name else "history"
            ),
            "text": text,
        }
        if row.get("url"):
            item["url"] = row["url"]
        if row.get("page_number") is not None:
            item["page_number"] = row["page_number"]
        encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > remaining:
            break
        remaining -= len(encoded)
        compact.append(item)
        source = {**row, "chunk_id": source_id, "text": text, "source_kind": item["source_kind"]}
        sources.append(source)
    observation = json.dumps({"results": compact}, ensure_ascii=False, separators=(",", ":"))
    return observation, sources
