"""Behavioral regressions for serving metrics and request resource ownership."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.observability import ServingMetrics


def test_traffic_window_does_not_truncate_at_latency_sample_capacity(monkeypatch):
    import app.observability as module
    clock = [100.5]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    metrics = ServingMetrics(max_samples=3)
    for _ in range(700):
        metrics.begin()
        metrics.finish(mode="hybrid", backend="faiss", outcome="success",
                       metrics={"e2e_ms": 10, "output_tokens": 2})
    clock[0] = 110.5
    result = metrics.snapshot()
    assert result["requests_last_5m"] == 700
    assert result["throughput"]["completed_requests_per_second"] == 70
    assert result["throughput"]["known_output_tokens_per_second"] == 140
    assert result["latency"][0]["sample_count"] == 3
    clock[0] = 401
    assert metrics.snapshot()["requests_last_5m"] == 0


def test_metrics_reject_private_values_and_bound_labels(caplog):
    metrics = ServingMetrics()
    with caplog.at_level("INFO", logger="app.telemetry"):
        metrics.begin()
        metrics.finish(mode="PRIVATE", backend="PRIVATE", outcome="PRIVATE",
                       metrics={"output_tokens": "PRIVATE", "e2e_ms": -2,
                                "generation_ms": True, "model_ttft_ms": float("nan"),
                                "answer": "PRIVATE"},
                       cache={"engine": "PRIVATE", "prompt": "PRIVATE",
                              "prefix_cache_hit_rate": "PRIVATE"})
    encoded = json.dumps(metrics.snapshot()) + caplog.text
    assert "PRIVATE" not in encoded
    item = metrics.snapshot()["latency"][0]
    assert all(item[key]["count"] == 0 for key in ("e2e_ms", "generation_ms", "model_ttft_ms"))
    assert metrics.snapshot()["totals"][0]["outcome"] == "unknown"


def test_errors_and_success_have_separate_percentiles():
    metrics = ServingMetrics()
    for outcome, latency in (("success", 100), ("error", 900)):
        metrics.begin()
        metrics.finish(mode="hybrid", backend="qdrant", outcome=outcome, metrics={"e2e_ms": latency})
    rows = {item["outcome"]: item for item in metrics.snapshot()["latency"]}
    assert rows["success"]["e2e_ms"]["p95"] == 100
    assert rows["error"]["e2e_ms"]["p95"] == 900


def test_cancelled_execution_records_elapsed_ttft_and_closes_model(monkeypatch, tmp_path):
    from app.api import routes
    from app.chat_modes import ChatMode
    from app.models.base import ModelDelta
    from app.schemas import ChatRequest
    from app.telemetry import RequestTrace
    from tests.test_baseline_runtime import _app
    import time
    metrics = ServingMetrics()
    monkeypatch.setattr(routes, "serving_metrics", metrics)
    app, store, model, _ = _app(tmp_path)
    closed = []
    async def stream(messages, *, max_new_tokens, cancel):
        try:
            yield ModelDelta("partial", time.perf_counter_ns())
            await asyncio.sleep(100)
        finally:
            closed.append(True)
    model.stream = stream
    owner = "cancel-test-owner"
    conversation = store.create_conversation(owner)
    payload = ChatRequest(conversation_id=conversation["id"], question="Bach Dang?")
    runtime = app.state.chat_mode_router.runtime_for(ChatMode.HYBRID)
    async def run():
        events = routes._execute(payload, SimpleNamespace(app=app), owner, store, runtime,
                                 ChatMode.HYBRID, RequestTrace("test", "hybrid"),
                                 watch_disconnect=False)
        async for event, _ in events:
            if event == "answer_delta":
                await events.aclose()
                break
        assert closed == [True]
    asyncio.run(run())
    result = metrics.snapshot()
    assert result["inflight"] == 0
    assert result["totals"] == [{"mode": "hybrid", "retrieval_backend": "faiss",
                                "outcome": "cancelled", "count": 1}]
    sample = result["latency"][0]
    assert sample["e2e_ms"]["p50"] > 0
    assert sample["final_answer_ttft_ms"]["count"] == 1
    assert sample["model_ttft_ms"]["count"] == 0


def test_admin_unicode_is_forbidden_and_success_is_uncacheable(monkeypatch):
    from app.main import observability_summary
    monkeypatch.setenv("OBSERVABILITY_ADMIN_TOKEN", "test-token")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(observability_summary(SimpleNamespace(headers={"Authorization": "Bearer \u0103"})))
    assert exc.value.status_code == 403
    app = FastAPI()
    app.add_api_route("/summary", observability_summary)
    with TestClient(app) as client:
        result = client.get("/summary", headers={"Authorization": "Bearer test-token"})
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    assert result.json()["scope"] == "process_local"


def test_memory_unavailable_is_explicit():
    result = ServingMetrics().snapshot()
    assert "process_memory" in result
    assert "gpu_memory_bytes" in result
    assert result["gpu_memory_bytes"] is None or result["gpu_memory_bytes"]["scope"] == "current_process_allocator"


def test_http_admission_errors_and_sse_errors_are_distinct(monkeypatch):
    from app.observability import RequestMetricsMiddleware
    from fastapi.responses import StreamingResponse
    metrics = ServingMetrics()
    app = FastAPI()
    app.add_middleware(RequestMetricsMiddleware, metrics=metrics)
    @app.post("/api/v1/chat")
    async def chat(value: int):
        raise HTTPException(status_code=503, detail="private details")
    @app.post("/api/v1/chat/stream")
    async def stream():
        return StreamingResponse(iter(['event: error\ndata: {}\n\n']), media_type="text/event-stream")
    with TestClient(app) as client:
        assert client.post("/api/v1/chat").status_code == 422
        assert client.post("/api/v1/chat?value=1").status_code == 503
        assert client.post("/api/v1/chat/stream").status_code == 200
    result = metrics.snapshot()
    assert result["http"]["inflight"] == 0
    assert result["http"]["requests_started"] == 3
    assert {(r["route"], r["status_class"], r["outcome"], r["count"])
            for r in result["http"]["totals"]} == {
        ("chat", "4xx", "complete", 1), ("chat", "5xx", "complete", 1),
        ("chat_stream", "2xx", "complete", 1)}
    assert result["requests_started"] == 0  # HTTP success does not imply inference success.
    assert "private details" not in json.dumps(result)


def test_asgi_send_failure_closes_sse_execution_immediately(monkeypatch, tmp_path):
    from app.api import routes
    from app.models.base import ModelDelta
    from app.schemas import ChatRequest
    from starlette.requests import ClientDisconnect
    from tests.test_baseline_runtime import _app
    import time
    metrics = ServingMetrics()
    monkeypatch.setattr(routes, "serving_metrics", metrics)
    app, store, model, _ = _app(tmp_path)
    closed = []
    async def stream(messages, *, max_new_tokens, cancel):
        try:
            yield ModelDelta("partial", time.perf_counter_ns())
            await asyncio.sleep(100)
        finally:
            closed.append(True)
    model.stream = stream
    owner = "asgi-cancel-test"
    conversation = store.create_conversation(owner)
    payload = ChatRequest(conversation_id=conversation["id"], question="Bach Dang?")
    async def run():
        async def disconnected(): return False
        response = await routes.chat_stream(payload, SimpleNamespace(app=app, is_disconnected=disconnected), owner, store)
        async def receive(): await asyncio.sleep(100)
        async def send(frame):
            if frame["type"] == "http.response.body" and b"answer_delta" in frame["body"]:
                raise OSError("client gone")
        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        # Keep response referenced: passing must not depend on async generator GC.
        assert response is not None and closed == [True]
        assert metrics.snapshot()["inflight"] == 0
    asyncio.run(run())


def test_http_partial_stream_disconnect_is_cancelled():
    from app.observability import RequestMetricsMiddleware
    from fastapi.responses import StreamingResponse
    metrics = ServingMetrics()
    async def run():
        sent = asyncio.Event()
        async def frames():
            yield b"first"
            await asyncio.sleep(100)
        async def application(scope, receive, send):
            await StreamingResponse(frames())(scope, receive, send)
        async def receive():
            await sent.wait()
            return {"type": "http.disconnect"}
        async def send(frame):
            if frame["type"] == "http.response.body":
                sent.set()
        await RequestMetricsMiddleware(application, metrics)(
            {"type": "http", "path": "/api/v1/chat/stream", "asgi": {"spec_version": "2.3"}},
            receive, send)
    asyncio.run(run())
    assert metrics.snapshot()["http"]["totals"] == [
        {"route": "chat_stream", "status_class": "2xx", "outcome": "cancelled", "count": 1}]


def test_structured_serving_log_is_enabled_without_external_logging_config():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c",
        "from app.telemetry import log_event; log_event('serving_request', e2e_ms=1)"],
        capture_output=True, text=True, check=True)
    row = json.loads(result.stderr.strip())
    assert row["event"] == "serving_request" and row["e2e_ms"] == 1
