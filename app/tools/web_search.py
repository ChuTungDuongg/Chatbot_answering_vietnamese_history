from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any, Protocol

from pydantic import BaseModel, Field


class WebSearchProvider(Protocol):
    async def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        ...


class LocalOnlyWebProvider:
    async def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        return []


class TavilyWebProvider:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Tavily provider requires WEB_SEARCH_API_KEY.")
        self.api_key = api_key

    async def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._search_sync, query, top_k)

    def _search_sync(self, query: str, top_k: int) -> list[dict[str, Any]]:
        body = json.dumps(
            {"api_key": self.api_key, "query": query, "max_results": top_k},
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "vn-history-agent/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read(1_000_000).decode("utf-8"))
        return [
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "text": row.get("content", ""),
                "score": row.get("score"),
                "source_kind": "web",
            }
            for row in payload.get("results", [])[:top_k]
        ]


def build_web_search_provider(name: str, api_key: str | None = None) -> WebSearchProvider:
    normalized = name.strip().lower()
    if normalized in {"", "none", "local", "local-only"}:
        return LocalOnlyWebProvider()
    if normalized == "tavily":
        return TavilyWebProvider(api_key or "")
    raise ValueError(f"Unsupported WEB_SEARCH_PROVIDER: {name}")


class SearchWebInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchWebTool:
    name = "search_web"
    description = "Search external web evidence when configured. Defaults to local-only no-op."
    input_schema = SearchWebInput

    def __init__(self, provider: WebSearchProvider | None = None):
        self.provider = provider or LocalOnlyWebProvider()

    async def run(self, arguments: SearchWebInput) -> list[dict[str, Any]]:
        return await self.provider.search(arguments.query, arguments.top_k)
