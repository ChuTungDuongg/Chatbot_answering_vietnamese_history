"""Fast model-free checks of HTTP timing and deterministic evaluation."""
from __future__ import annotations

import json
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchmarks.latency.metrics import summarize as summarize_latency
from benchmarks.latency.runner import _merge_observed, run as run_benchmark
from benchmarks.latency.sse_client import measure_request, parse_sse
from evaluation.metrics.answer import score_answer
from evaluation.metrics.citations import score_citations
from evaluation.metrics.grounding import score_grounding
from evaluation.metrics.retrieval import score_retrieval
from evaluation.runner import run as run_evaluation


def test_sse_frames_support_multiline_data_and_monotonic_observations():
    ticks = iter((1_000_000, 2_000_000))
    events = list(parse_sse([b": keepalive\n", b"event: status\n", b"data: {\"stage\":\n",
                             b"data: \"ready\"}\n", b"\n", b"event: answer_delta\n",
                             b"data: {\"delta\":\"ok\"}\n", b"\n"],
                            clock=lambda: next(ticks)))
    assert [(event.name, event.data) for event in events] == [
        ("status", {"stage": "ready"}), ("answer_delta", {"delta": "ok"})]
    assert [event.observed_ns for event in events] == [1_000_000, 2_000_000]


class FakeSSEHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        assert self.path == "/api/v1/baseline/metadata?mode=hybrid"
        body = json.dumps({"schema_version": 1, "git_commit": "server-commit",
                           "model_ids": {"hybrid": "Qwen/Qwen3-4B-Instruct-2507", "central": "Qwen/Qwen3-8B"},
                           "server_hardware": {"gpu_model": "Fake GPU", "gpu_memory_bytes": 16_000_000_000},
                           "corpus_hash": "corpus-sha", "retrieval_index_hash": "index-sha",
                           "generation_settings": {"hybrid": {"do_sample": False, "enable_thinking": False}},
                           "retrieval_settings": {"hybrid": {"top_k": 6}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(size))
        if self.path == "/api/v1/conversations":
            body = json.dumps({"id": "00000000-0000-0000-0000-000000000001"}).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        assert payload["mode"] == "hybrid"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for frame in (
            'event: status\ndata: {"stage":"retrieval"}\n\n',
            'event: answer_delta\ndata: {"delta":"Đáp "}\n\n',
            'event: answer_delta\ndata: {"delta":"án [S1]"}\n\n',
            'event: sources\ndata: {"items":[{"chunk_id":"c1","source_id":"s1","display_index":1}],"cited_source_ids":["s1"]}\n\n',
            'event: done\ndata: {"status":"ok","request_id":"server-1","model_id":"Qwen/Qwen3-4B-Instruct-2507","metrics":{"retrieval_ms":5.0,"output_tokens":3}}\n\n',
        ):
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()


def test_external_http_client_records_only_observed_server_metrics():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        row = measure_request(base_url=f"http://127.0.0.1:{server.server_port}",
                              question={"id": "q1", "question": "Câu hỏi?"},
                              mode="hybrid", client_id="test-client-123", timeout=5,
                              phase="warm", run_index=0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert row["success"] and row["http_status"] == 200
    assert row["answer"] == "Đáp án [S1]"
    assert row["server_request_id"] == "server-1"
    assert 0 <= row["ttfb_ms"] <= row["answer_ttft_ms"] <= row["e2e_ms"]
    assert row["first_status_event_ms"] is not None
    assert row["retrieval_ms"] == 5.0 and row["output_tokens"] == 3
    assert row["model_ttft_ms"] is None and row["generation_ms"] is None
    assert len(row["inter_token_latency_ms"]) == 1


def test_ttfb_uses_http_response_start_before_first_status():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    ticks = iter(range(1_000_000, 9_000_000, 1_000_000))
    try:
        row = measure_request(base_url=f"http://127.0.0.1:{server.server_port}",
                              question={"id": "q1", "question": "Câu hỏi?"},
                              mode="hybrid", client_id="test-client-123", timeout=5,
                              phase="warm", run_index=0, clock=lambda: next(ticks))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert row["success"]
    assert row["ttfb_ms"] == 1.0
    assert row["first_status_event_ms"] == 2.0
    assert row["answer_ttft_ms"] == 3.0


def test_latency_cold_and_warm_aggregation_are_separate():
    summary = summarize_latency([
        {"phase": "cold", "success": True, "e2e_ms": 100},
        {"phase": "warm", "success": True, "e2e_ms": 10},
        {"phase": "warm", "success": False, "e2e_ms": None},
    ])
    assert summary["groups"]["cold"]["metrics"]["e2e_ms"]["mean"] == 100
    assert summary["groups"]["warm"]["metrics"]["e2e_ms"]["mean"] == 10
    assert summary["groups"]["warm"]["success_rate"] == .5
    assert summary["groups"]["warm"]["metrics"]["model_ttft_ms"]["mean"] is None


def test_declared_metadata_fills_nulls_but_conflicts_fail():
    assert _merge_observed({"gpu_model": "A", "cuda_version": None},
                           {"gpu_model": "A", "cuda_version": "12.8"},
                           "server_hardware") == {"gpu_model": "A", "cuda_version": "12.8"}
    with pytest.raises(ValueError, match="conflicts"):
        _merge_observed("server-hash", "different-hash", "corpus_hash")


def test_retrieval_metrics_and_missing_labels():
    ranked = [{"chunk_id": "c1", "source_id": "s1"},
              {"chunk_id": "c2", "source_id": "s1"},
              {"chunk_id": "c3", "source_id": "s2"}]
    score = score_retrieval(ranked, relevant_chunk_ids=["c2", "c3"],
                            relevant_source_ids=["s2"])
    assert score["chunk"]["hit_rate@1"] == 0
    assert score["chunk"]["recall@3"] == 1
    assert score["chunk"]["precision@3"] == pytest.approx(2 / 3)
    assert score["chunk"]["mrr@3"] == .5
    assert score["chunk"]["ndcg@3"] == pytest.approx((1 / 1.5849625007 + .5) / (1 + 1 / 1.5849625007))
    assert score["source"]["mrr@3"] == .5
    assert score_retrieval(ranked, relevant_chunk_ids=None,
                           relevant_source_ids=None)["chunk"]["recall@3"] is None


def test_answer_citation_and_grounding_metrics():
    answer = score_answer("a b c", "a c")
    assert answer["token_f1"] == pytest.approx(.8)
    assert answer["rouge_l_f1"] == pytest.approx(.8)
    assert score_answer("anything", None)["token_f1"] is None
    sources = [{"chunk_id": "c1", "source_id": "s1", "display_index": 1,
                "text": "Sự kiện diễn ra năm 938."}]
    citations = score_citations("Năm 938. [S1] [S99]", sources,
                                gold_source_ids=["s1"], known_source_ids={"s1"},
                                factual_paragraph_indices=[0])
    assert citations["citation_validity_rate"] == .5
    assert citations["citation_precision"] == .5
    assert citations["citation_recall"] == 1
    assert citations["source_id_existence_rate"] == .5
    assert citations["answer_citation_coverage"] == 1
    grounding = score_grounding("Sự kiện năm 938 và 1945. [S1]", "Năm nào?", sources)
    assert grounding["unverified_years"] == ["1945"]
    assert score_grounding("1945", "Năm nào?", [])["unverified_years"] is None


def test_offline_runner_writes_na_for_unlabeled_questions(tmp_path: Path):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"schema_version":1,"id":"q1","question":"Vì sao?","category":"fact"}\n', encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"question_id": "q1", "phase": "warm", "success": True,
                                       "answer": "Không rõ.", "sources": []}) + "\n", encoding="utf-8")
    output = run_evaluation(dataset, predictions, tmp_path / "report")
    per_question = json.loads((output / "per_question.jsonl").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert per_question["answer_metrics"]["token_f1"] is None
    assert per_question["retrieval"]["chunk"]["recall@1"] is None
    assert metrics["answer"]["token_f1"] == {"value": None, "observed": 0}


def test_full_external_benchmark_and_offline_report(tmp_path: Path):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"schema_version":1,"id":"q1","question":"Câu hỏi?","category":"fact"}\n', encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        benchmark_dir = run_benchmark(Namespace(
            dataset=dataset, mode="hybrid", output=tmp_path / "benchmark",
            base_url=f"http://127.0.0.1:{server.server_port}", warmup=1, runs=2,
            concurrency=1, cold_start=True, client_id="test-client-123",
            timeout=5, metadata_file=None,
        ))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    rows = [json.loads(line) for line in (benchmark_dir / "latency_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["phase"] for row in rows] == ["cold", "warmup", "warm", "warm"]
    assert all(row["success"] for row in rows)
    summary = json.loads((benchmark_dir / "latency_summary.json").read_text(encoding="utf-8"))
    assert summary["groups"]["cold"]["count"] == 1
    assert summary["groups"]["warm"]["count"] == 2
    metadata = json.loads((benchmark_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert metadata["server_hardware"]["gpu_model"] == "Fake GPU"
    assert metadata["corpus_hash"] == "corpus-sha"
    assert metadata["retrieval_index_hash"] == "index-sha"
    assert metadata["generation_settings"]["do_sample"] is False
    assert metadata["git_commit"] == "server-commit"
    report = run_evaluation(dataset, benchmark_dir / "latency_records.jsonl", tmp_path / "evaluation")
    evaluation = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    assert evaluation["prediction_count"] == 2
    assert evaluation["provenance"]["model_id"] == metadata["model_id"]
