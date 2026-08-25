from __future__ import annotations

import re
import urllib.request
from html.parser import HTMLParser

from pydantic import BaseModel, Field


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)
            if self._in_title and not self.title:
                self.title = text


class FetchPageInput(BaseModel):
    url: str = Field(..., min_length=8)
    max_chars: int = Field(default=8000, ge=200, le=20000)


class FetchPageTool:
    name = "fetch_web_page"
    description = "Fetch and extract visible text from a web page URL."
    input_schema = FetchPageInput

    def run(self, arguments: FetchPageInput) -> dict[str, str]:
        request = urllib.request.Request(arguments.url, headers={"User-Agent": "vn-history-agent/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read(1_000_000)
            content_type = response.headers.get_content_type().lower()
            final_url = response.geturl()
        if content_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
            raise ValueError(f"Unsupported content type: {content_type}")
        text = raw.decode("utf-8", errors="ignore")
        title = ""
        if content_type in {"text/html", "application/xhtml+xml"}:
            parser = _TextExtractor()
            parser.feed(text)
            text = " ".join(parser.parts)
            title = parser.title
        text = re.sub(r"\s+", " ", text).strip()
        return {
            "url": final_url,
            "title": title,
            "content_type": content_type,
            "text": text[: arguments.max_chars],
        }
