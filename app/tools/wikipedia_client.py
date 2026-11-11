from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any



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
