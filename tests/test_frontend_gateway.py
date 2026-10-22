"""CPU gateway preserves request identity and bytes for SSE proxying."""

import httpx
from fastapi.testclient import TestClient

from frontend_server import server


def test_frontend_health_without_gpu(monkeypatch):
    monkeypatch.setattr(server, "UPSTREAM", "")
    with TestClient(server.gateway) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["backend_configured"] is False


def test_cpu_gateway_proxies_sse_without_rewriting_chunks(monkeypatch):
    monkeypatch.setattr(server, "UPSTREAM", "https://example-gpu.modal.run")
    seen = {}
    payload = b'event: answer_delta\ndata: {"delta":"Waterloo"}\n\n'

    class Client:
        def __init__(self, **_kwargs):
            pass

        def build_request(self, method, url, **kwargs):
            seen["method"] = method
            seen["url"] = url
            seen["headers"] = kwargs["headers"]
            return httpx.Request(method, url, headers=kwargs["headers"])

        async def send(self, request, stream=False):
            assert stream
            return httpx.Response(200, headers={"content-type": "text/event-stream",
                                                  "cache-control": "no-cache",
                                                  "x-request-id": "abc"},
                                  stream=httpx.ByteStream(payload))

        async def aclose(self):
            seen["closed"] = True

    monkeypatch.setattr(server.httpx, "AsyncClient", Client)
    with TestClient(server.gateway) as client:
        response = client.post("/api/v1/chat/stream", headers={"X-Client-ID": "tester"})
    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("text/event-stream")
    assert seen["url"] == "https://example-gpu.modal.run/api/v1/chat/stream"
    assert seen["headers"]["x-client-id"] == "tester"
    assert seen["closed"] is True
