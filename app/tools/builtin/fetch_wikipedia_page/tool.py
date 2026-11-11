from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field
from app.tools.declarations import BuiltinTool, builtin_description
from app.tools.wikipedia_client import _request_json, _clean_text


class FetchWikipediaPageInput(BaseModel):
    page_id_or_title: str = Field(..., min_length=1)
    language: Literal["vi", "en"] = "vi"
    max_chars: int = Field(default=8000, ge=200, le=20000)


class FetchWikipediaPageTool(BuiltinTool):
    name = "fetch_wikipedia_page"
    description = builtin_description("fetch_wikipedia_page")
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
