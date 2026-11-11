from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from app.tools.declarations import BuiltinTool, builtin_description
from app.tools.web_providers import WebSearchProvider, LocalOnlyWebProvider


class SearchWebInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchWebTool(BuiltinTool):
    name = "search_web"
    description = builtin_description("search_web")
    input_schema = SearchWebInput

    def __init__(self, provider: WebSearchProvider | None = None):
        self.provider = provider or LocalOnlyWebProvider()

    async def run(self, arguments: SearchWebInput) -> list[dict[str, Any]]:
        return await self.provider.search(arguments.query, arguments.top_k)
