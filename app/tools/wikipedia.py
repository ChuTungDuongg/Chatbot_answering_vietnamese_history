from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Literal

from pydantic import BaseModel, Field


WIKIPEDIA_TIMEOUT_SEC = 8
WIKIPEDIA_READ_LIMIT = 1_000_000


def _api_base(language: str) -> str:
    return f"https://{language}.wikipedia.org/w/api.php"


def _clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", without_tags).strip()


def _request_json(language: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode({
        **params,
        "format": "json",
        "formatversion": 2,
        "utf8": 1,
    })
    request = urllib.request.Request(
        f"{_api_base(language)}?{query}",
        headers={"User-Agent": "vn-history-agent/1.0"},
    )
    with urllib.request.urlopen(request, timeout=WIKIPEDIA_TIMEOUT_SEC) as response:
        return json.loads(response.read(WIKIPEDIA_READ_LIMIT).decode("utf-8"))


class SearchWikipediaInput(BaseModel):
    query: str = Field(..., min_length=1)
    language: Literal["vi", "en"] = "vi"
    top_k: int = Field(default=5, ge=1, le=10)


class FetchWikipediaPageInput(BaseModel):
    page_id_or_title: str = Field(..., min_length=1)
    language: Literal["vi", "en"] = "vi"
    max_chars: int = Field(default=8000, ge=200, le=20000)


class SearchWikipediaTool:
    name = "search_wikipedia"
    description = (
        "Search Vietnamese Wikipedia first, or English Wikipedia when language='en'. "
        "Returns stable wikipedia evidence IDs and snippets only."
    )
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


class FetchWikipediaPageTool:
    name = "fetch_wikipedia_page"
    description = (
        "Fetch one bounded plain-text Wikipedia page by page ID or exact title. "
        "Use language='vi' first and language='en' only as fallback."
    )
    input_schema = FetchWikipediaPageInput

    def run(self, arguments: FetchWikipediaPageInput) -> dict[str, Any]:
        key = str(arguments.page_id_or_title).strip()
        params: dict[str, Any] = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "redirects": 1,
        }
        if key.isdigit():
            params["pageids"] = key
        else:
            params["titles"] = key
        payload = _request_json(arguments.language, params)
        pages = payload.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        page = next((item for item in pages if int(item.get("pageid") or 0) > 0), None)
        if not page:
            raise ValueError(f"Wikipedia page not found: {key}")
        page_id = int(page["pageid"])
        text = _clean_text(page.get("extract", ""))[: arguments.max_chars]
        return {
            "chunk_id": f"wiki_{arguments.language}_{page_id}",
            "source_kind": "wikipedia",
            "title": page.get("title"),
            "url": page.get("fullurl") or f"https://{arguments.language}.wikipedia.org/?curid={page_id}",
            "text": text,
            "metadata": {"page_id": page_id, "language": arguments.language},
        }
