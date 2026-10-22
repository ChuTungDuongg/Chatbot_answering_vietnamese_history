"""Regression checks for bounded, private, process-local telemetry."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.observability import ServingMetrics
from app.rag.prompting import LIVE_SYSTEM_PROMPT, SYSTEM_PROMPT, build_messages


def test_serving_metrics_counts_latency_and_cache_without_inventing_hit_rate():
    metrics = ServingMetrics(max_samples=3)
    metrics.begin()
    metrics.finish(mode="hybrid", backend="faiss", outcome="success",
                   metrics={"e2e_ms": 100, "final_answer_ttft_ms": 40, "output_tokens": 10},
                   cache={"engine": "transformers", "prefix_cache_hit_rate": None})
    metrics.begin()
    metrics.finish(mode="hybrid", backend="faiss", outcome="success",
                   metrics={"e2e_ms": 300, "final_answer_ttft_ms": None})
    result = metrics.snapshot()
    assert result["inflight"] == 0
    assert result["requests_last_5m"] == 2
    assert {(r["outcome"], r["count"]) for r in result["totals"]} == {("success", 2)}
    latency = result["latency"][0]
    assert latency["e2e_ms"]["p50"] == 200
    assert latency["e2e_ms"]["p95"] == 290
    assert latency["final_answer_ttft_ms"]["count"] == 1
    assert result["cache"]["hybrid"]["prefix_cache_hit_rate"] is None
    assert result["durable"] is False


def test_live_style_does_not_mutate_frozen_benchmark_prompt():
    question = "Nếu Grouchy đến Waterloo thì sao?"
    chunks = [{"chunk_id": "source-1", "text": "Grouchy dẫn 33.000 quân đến Wavre."}]
    live = build_messages(question, chunks)
    frozen = build_messages(question, chunks, response_mode="standard")
    assert live[0]["content"] == LIVE_SYSTEM_PROMPT
    assert frozen[0]["content"].startswith(SYSTEM_PROMPT)
    assert "theo thông tin được cung cấp" in live[0]["content"].lower()
    assert "[source-1]" in live[-1]["content"]


def test_monitoring_endpoint_requires_secret(monkeypatch):
    from app.main import observability_summary

    request = SimpleNamespace(headers={"Authorization": "Bearer test-token"})
    monkeypatch.delenv("OBSERVABILITY_ADMIN_TOKEN", raising=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(observability_summary(request))
    assert exc.value.status_code == 404

    monkeypatch.setenv("OBSERVABILITY_ADMIN_TOKEN", "test-token")
    assert asyncio.run(observability_summary(request)).status_code == 200
    with pytest.raises(HTTPException) as exc:
        asyncio.run(observability_summary(SimpleNamespace(headers={})))
    assert exc.value.status_code == 403
