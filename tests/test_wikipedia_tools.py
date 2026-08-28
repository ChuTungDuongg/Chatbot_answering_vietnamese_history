from __future__ import annotations

import json
from types import SimpleNamespace

from app.tools.wikipedia import FetchWikipediaPageTool, SearchWikipediaTool


class FakeHTTPResponse:
    def __init__(self, payload: dict, url: str = "https://vi.wikipedia.org/w/api.php"):
        self.payload = payload
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int = -1):
        return json.dumps(self.payload).encode("utf-8")

    def geturl(self):
        return self.url


def test_search_wikipedia_returns_stable_wikipedia_sources(monkeypatch):
    def fake_urlopen(request, timeout):
        assert timeout == 8
        assert "vi.wikipedia.org" in request.full_url
        return FakeHTTPResponse({
            "query": {
                "search": [
                    {
                        "pageid": 123,
                        "title": "Trận Bạch Đằng (938)",
                        "snippet": "Chiến thắng của Ngô Quyền",
                    }
                ]
            }
        })

    monkeypatch.setattr("app.tools.wikipedia.urllib.request.urlopen", fake_urlopen)

    rows = SearchWikipediaTool().run(SimpleNamespace(query="Bạch Đằng 938", language="vi", top_k=3))

    assert rows == [
        {
            "chunk_id": "wiki_vi_123",
            "source_kind": "wikipedia",
            "title": "Trận Bạch Đằng (938)",
            "url": "https://vi.wikipedia.org/?curid=123",
            "text": "Chiến thắng của Ngô Quyền",
            "metadata": {"page_id": 123, "language": "vi"},
        }
    ]


def test_fetch_wikipedia_page_caps_content_and_preserves_provenance(monkeypatch):
    def fake_urlopen(request, timeout):
        assert timeout == 8
        assert "prop=extracts" in request.full_url
        return FakeHTTPResponse({
            "query": {
                "pages": {
                    "123": {
                        "pageid": 123,
                        "title": "Trận Bạch Đằng (938)",
                        "fullurl": "https://vi.wikipedia.org/wiki/Tr%E1%BA%ADn_B%E1%BA%A1ch_%C4%90%E1%BA%B1ng_(938)",
                        "extract": "A" * 500,
                    }
                }
            }
        })

    monkeypatch.setattr("app.tools.wikipedia.urllib.request.urlopen", fake_urlopen)

    row = FetchWikipediaPageTool().run(SimpleNamespace(page_id_or_title="123", language="vi", max_chars=120))

    assert row["chunk_id"] == "wiki_vi_123"
    assert row["source_kind"] == "wikipedia"
    assert row["title"] == "Trận Bạch Đằng (938)"
    assert row["url"].startswith("https://vi.wikipedia.org/wiki/")
    assert row["text"] == "A" * 120
    assert row["metadata"] == {"page_id": 123, "language": "vi"}
