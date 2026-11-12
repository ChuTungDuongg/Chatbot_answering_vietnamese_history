from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field
from app.tools.declarations import BuiltinTool, builtin_description
from app.tools.registry import ToolExecutionPolicy
from app.tools.wikipedia_client import _request_json, _clean_text


class SearchWikipediaInput(BaseModel):
    query: str = Field(..., min_length=1)
    language: Literal["vi", "en"] = "vi"
    top_k: int = Field(default=5, ge=1, le=10)


class SearchWikipediaTool(BuiltinTool):
    execution_policy = ToolExecutionPolicy(read_only=True, parallel_safe=True)
    name = "search_wikipedia"
    description = builtin_description("search_wikipedia")
    input_schema = SearchWikipediaInput

    def run(self, arguments: SearchWikipediaInput) -> list[dict[str, Any]]:
        payload = _request_json(
            arguments.language,
            {
                "action": "query",
                "list": "search",
                "srsearch": arguments.query,
                "srlimit": arguments.top_k,
            },
        )
        rows = payload.get("query", {}).get("search", [])
        result: list[dict[str, Any]] = []
        for row in rows[: arguments.top_k]:
            page_id = int(row.get("pageid") or 0)
            if page_id <= 0:
                continue
            result.append({
                "chunk_id": f"wiki_{arguments.language}_{page_id}",
                "source_kind": "wikipedia",
                "title": row.get("title"),
                "url": f"https://{arguments.language}.wikipedia.org/?curid={page_id}",
                "text": _clean_text(row.get("snippet", "")),
                "metadata": {"page_id": page_id, "language": arguments.language},
            })
        return result
