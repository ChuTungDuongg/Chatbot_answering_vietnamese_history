"""HTTP integration tests: streaming must be observed before upstream completion."""
import asyncio
from contextlib import contextmanager
import gzip
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient

from frontend_server import server


@contextmanager
def running(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    runner = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="off"))
    thread = threading.Thread(target=runner.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not runner.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("Test HTTP server did not start")
        time.sleep(.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        runner.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


def test_stream_is_incremental_and_abort_closes_upstream(monkeypatch):
    api = FastAPI()
    closed = threading.Event()
    release = threading.Event()
    @api.post("/api/v1/chat/stream")
    async def stream():
        async def frames():
            try:
                yield b"event: answer_delta\ndata: first\n\n"
                while not release.is_set():
                    await asyncio.sleep(.01)
                yield b"event: done\ndata: {}\n\n"
            finally:
                closed.set()
        return StreamingResponse(frames(), media_type="text/event-stream")
    with running(api) as upstream:
        monkeypatch.setattr(server, "UPSTREAM", upstream)
        with running(server.gateway) as gateway:
            try:
                with httpx.Client(timeout=5) as client:
                    with client.stream("POST", gateway + "/api/v1/chat/stream") as response:
                        assert response.status_code == 200
                        assert next(response.iter_bytes()) == b"event: answer_delta\ndata: first\n\n"
                        assert not release.is_set() and not closed.is_set()
                assert closed.wait(5), "Browser abort did not close the GPU response"
            finally:
                release.set()


def test_upload_bytes_repeated_query_headers_and_error_status(monkeypatch):
    api = FastAPI()
    seen = {}
    @api.post("/api/v1/upload")
    async def upload(request: Request):
        seen.update(query=request.query_params.getlist("tag"), encoding=request.headers.get("content-encoding"))
        return Response(await request.body(), status_code=422, headers={
            "content-type": request.headers["content-type"],
            "x-request-id": request.headers["x-request-id"],
            "retry-after": "5",
            "x-seen-query": str(request.query_params.getlist("tag")),
            "x-seen-encoding": request.headers.get("content-encoding", "missing")})
    # Raw bytes must remain untouched, including compressed multipart bodies.
    payload = gzip.compress(b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\nhistory\r\n--boundary--')
    with running(api) as upstream:
        monkeypatch.setattr(server, "UPSTREAM", upstream)
        with TestClient(server.gateway) as client:
            result = client.post("/api/v1/upload?tag=a&tag=b", content=payload,
                headers={"Content-Type": "multipart/form-data; boundary=boundary",
                         "Content-Encoding": "gzip", "X-Request-ID": "test"})
        assert result.status_code == 422 and result.content == payload
        assert result.headers["retry-after"] == "5"
        assert seen == {"query": ["a", "b"], "encoding": "gzip"}


def test_cancel_before_response_body_starts_closes_resources(monkeypatch):
    async def run():
        opened = asyncio.Event()
        closed = []
        class Client:
            def __init__(self, **kwargs): pass
            def build_request(self, *args, **kwargs): return object()
            async def send(self, *args, **kwargs):
                opened.set()
                await asyncio.sleep(100)
            async def aclose(self): closed.append(True)
        monkeypatch.setattr(server.httpx, "AsyncClient", Client)
        monkeypatch.setattr(server, "UPSTREAM", "https://gpu.example")
        request = Request({"type": "http", "method": "POST", "headers": [], "query_string": b""})
        task = asyncio.create_task(server._proxy(request, "api/v1/chat/stream"))
        await opened.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [True]
    asyncio.run(run())


def test_static_routes_are_cpu_only_and_index_is_never_cached(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("UI", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(server, "DIST", tmp_path)
    monkeypatch.setattr(server, "UPSTREAM", "")
    with TestClient(server.gateway) as client:
        assert client.get("/").text == "UI"
        assert client.get("/chat/abc").text == "UI"
        assert client.get("/favicon.svg").status_code == 200
        assert client.get("/index.html").headers["cache-control"] == "no-cache"
        assert client.get("/api/unknown").status_code == 404
        assert client.get("/api/v1/ready").status_code == 503
        assert client.get("/ready").status_code == 503


def test_browser_abort_while_waiting_for_headers_closes_gpu_request(monkeypatch):
    api = FastAPI()
    disconnected = threading.Event()
    @api.post("/api/v1/wait")
    async def wait(request: Request):
        await request.body()
        deadline = time.monotonic() + 6
        while not await request.is_disconnected():
            if time.monotonic() > deadline:
                return Response(status_code=504)
            await asyncio.sleep(.01)
        disconnected.set()
        return Response(status_code=204)
    with running(api) as upstream:
        monkeypatch.setattr(server, "UPSTREAM", upstream)
        with running(server.gateway) as gateway:
            with pytest.raises(httpx.ReadTimeout):
                httpx.post(gateway + "/api/v1/wait", content=b"payload", timeout=.3)
            assert disconnected.wait(3)


def test_no_body_and_unavailable_backend_preserve_status(monkeypatch):
    api = FastAPI()
    @api.delete("/api/v1/empty")
    async def empty():
        return Response(status_code=204)
    with running(api) as upstream:
        monkeypatch.setattr(server, "UPSTREAM", upstream)
        with TestClient(server.gateway) as client:
            response = client.delete("/api/v1/empty")
            assert response.status_code == 204 and response.content == b""
    # The socket was closed: errors must be sanitized gateway errors, not HTML.
    with TestClient(server.gateway) as client:
        response = client.get("/api/v1/missing")
        assert response.status_code == 502
        assert response.json() == {"detail": "GPU backend unavailable"}
