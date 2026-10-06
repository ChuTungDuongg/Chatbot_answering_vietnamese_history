"""Frozen workload, HTTP matrix, comparison arithmetic and verified Drive transfers."""
from argparse import Namespace
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from benchmarks.latency.compare import difference, execute, load_experiment, validate_invariants
from benchmarks.latency.freeze import DEFAULT_SOURCE, ROOT, sample
from benchmarks.latency.runner import main as legacy_main
from tools import drive_cli


def identity():
    return {"schema_version": 1, "model_id": "Qwen/Qwen3-4B-Instruct-2507", "model_revision": "revision-sha",
            "model_variant": "vanilla", "adapter_attached": False, "adapter_fingerprint": None,
            "corpus_hash": "c" * 64, "retrieval_index_hash": "i" * 64,
            "git_commit": "commit", "deployment_source_sha256": "s" * 64,
            "generation_settings": {"do_sample": False, "enable_thinking": False, "max_new_tokens": 1536},
            "retrieval_settings": {"final_k": 6}, "mode": "hybrid", "population": "hybrid",
            "concurrency": 1, "request_options": {}, "dataset_sha256": "d" * 64,
            "server_hardware": {"gpu_model": "NVIDIA A100", "gpu_memory_bytes": 80_000_000_000,
                                "torch_version": "2.11.0", "transformers_version": "4.57.6", "cuda_version": "13.0"},
            "server_environment": {"enable_hybrid_mode": True, "enable_central_mode": False}}


@pytest.mark.parametrize("field", ["dataset_sha256", "model_revision", "generation_settings", "retrieval_settings", "corpus_hash", "mode", "concurrency", "population"])
def test_comparison_rejects_unfair_invariants(field):
    reference = identity(); candidate = copy.deepcopy(reference)
    candidate[field] = "different"
    with pytest.raises(ValueError, match="differs"):
        validate_invariants(reference, candidate)


def test_comparison_rejects_unknown_revision_hardware_and_coloading():
    reference = identity(); reference["model_revision"] = None
    with pytest.raises(ValueError, match="unknown"):
        validate_invariants(reference, copy.deepcopy(reference))
    reference = identity(); candidate = copy.deepcopy(reference)
    candidate["server_hardware"]["gpu_model"] = "H100"
    with pytest.raises(ValueError, match="Hardware"):
        validate_invariants(reference, candidate)
    candidate = copy.deepcopy(reference); candidate["server_environment"]["enable_central_mode"] = True
    with pytest.raises(ValueError, match="isolated"):
        validate_invariants(reference, candidate)


def test_comparison_math_labels_latency_reduction_and_throughput_increase():
    assert difference(100, 80) == {"absolute_difference": -20, "percentage": 20,
                                  "percentage_label": "latency_reduction_pct"}
    assert difference(100, 120, higher_is_better=True)["percentage"] == 20
    assert difference(0, 10)["percentage"] is None
    assert difference(None, 10)["absolute_difference"] is None


def test_frozen_100_question_selection_ids_and_hash_are_reproducible():
    data, manifest = sample(DEFAULT_SOURCE)
    again, same = sample(DEFAULT_SOURCE)
    frozen = ROOT / "evaluation/datasets/latency_v1/questions_100.jsonl"
    assert data == again == frozen.read_bytes()
    assert manifest == same
    assert manifest["count"] == 100 and len(set(manifest["selected_question_ids"])) == 100
    assert hashlib.sha256(data).hexdigest() == manifest["dataset_sha256"]
    assert len(manifest["distributions"]["category"]) == 21
    assert manifest["distributions"]["answerable"]["False"] == 1


class ExperimentHandler(BaseHTTPRequestHandler):
    backend = "transformers"
    def log_message(self, *_): pass
    def metadata(self):
        return {**identity(), "inference_backend": self.backend,
                "inference_engine_version": "4.57.6" if self.backend == "transformers" else "0.23.0",
                "inference_engine_config": {"enable_prefix_caching": False} if self.backend == "vllm" else {"generation_serialized": True}}
    def do_GET(self):
        body = json.dumps(self.metadata()).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/v1/conversations":
            body = b'{"id":"00000000-0000-0000-0000-000000000001"}'
            self.send_response(201); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        assert self.path == "/api/v1/chat/stream" and payload["mode"] == "hybrid"
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        done = {**self.metadata(), "status": "done", "request_id": "request",
                "metrics": {"input_tokens": 10, "output_tokens": 4, "generation_ms": 10,
                            "model_ttft_ms": 4, "tpot_ms": 2, "decode_tokens_per_second": 500,
                            "max_new_tokens": 1536, "finish_reason": "stop", "hit_max_new_tokens": False,
                            "truncated": False, "planning_model_ms": None}}
        for event, data in [("status", {"stage": "generation"}),
                            ("answer_delta", {"delta": "Năm 938. [c1]"}),
                            ("sources", {"items": [{"chunk_id": "c1", "text": "Năm 938."}], "cited_source_ids": ["c1"]}),
                            ("done", done)]:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data,ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()


def test_non_gpu_http_matrix_and_legacy_cli_smoke(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"id":"q1","question":"Năm nào?","category":"chronology"}\n', encoding="utf-8")
    servers = []
    try:
        for backend in ("transformers", "vllm"):
            handler = type(backend + "Handler", (ExperimentHandler,), {"backend": backend})
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            servers.append((server, thread))
        config = {"schema_version": 1, "experiment_id": "fixture", "mode": "hybrid", "population": "hybrid",
                  "dataset": str(dataset), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  "concurrency": [1, 2], "runs": 1, "warmup": 0, "request_options": {},
                  "reference_variant": "transformers_reference", "offline_evaluation": True,
                  "variants": [{"id": backend + "_reference", "base_url": f"http://127.0.0.1:{server.server_port}",
                                "expected": {"inference_backend": backend}} for backend, (server, _) in zip(("transformers", "vllm"), servers)]}
        path = tmp_path / "experiment.json"; path.write_text(json.dumps(config), encoding="utf-8")
        output = execute(path, tmp_path / "comparison")
        report = json.loads((output / "comparison.json").read_text())
        assert [item["concurrency"] for item in report["groups"]] == [1, 2]
        assert all(item["comparable"] for group in report["groups"] for item in group["candidates"])
        assert (output / "transformers_reference/c1/quality/metrics.json").is_file()
        record = json.loads((output / "vllm_reference/c2/latency_records.jsonl").read_text())
        assert record["inference_backend"] == "vllm" and all(record["guardrails"].values())
        metadata = json.loads((output / "vllm_reference/c2/run_metadata.json").read_text())
        assert metadata["variant_id"] == "vllm_reference" and metadata["concurrency"] == 2
        assert metadata["inference_engine_config"] == {"enable_prefix_caching": False}
        assert legacy_main(["--mode", "hybrid", "--base-url", config["variants"][0]["base_url"],
                            "--dataset", str(dataset), "--output", str(tmp_path / "legacy"), "--runs", "1", "--warmup", "0"]) == 0
        config["dataset_sha256"] = "wrong"; path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="SHA256"):
            load_experiment(path)
    finally:
        for server, thread in servers:
            server.shutdown(); server.server_close(); thread.join()


def test_drive_benchmark_assets_results_and_hash_validation(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; drive = tmp_path / "drive"
    monkeypatch.setattr(drive_cli, "ROOT", repo)
    for local in drive_cli.BENCHMARK_ASSETS:
        path = repo / local; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
    assert drive_cli.main(["init", "--drive-root", str(drive)]) == 0
    for command in ("push-benchmark-datasets", "verify-benchmark-datasets", "pull-benchmark-datasets"):
        assert drive_cli.main([command, "--drive-root", str(drive)]) == 0
    remote_file = drive / next(iter(drive_cli.BENCHMARK_ASSETS.values()))
    remote_file.write_text("different")
    assert drive_cli.main(["verify-benchmark-datasets", "--drive-root", str(drive)]) == 2
    results = tmp_path / "results"; results.mkdir()
    (results / "run_metadata.json").write_text("{}")
    (results / "latency_records.jsonl").write_text("{}\n")
    common = ["--drive-root", str(drive), "--local-path", str(results), "--name", "experiment"]
    assert drive_cli.main(["push-benchmark-results", *common]) == 0
    assert drive_cli.main(["verify-benchmark-results", *common]) == 0
    downloaded = tmp_path / "downloaded"
    assert drive_cli.main(["pull-benchmark-results", "--drive-root", str(drive), "--local-path", str(downloaded), "--name", "experiment"]) == 0
    assert (downloaded / "latency_records.jsonl").read_bytes() == (results / "latency_records.jsonl").read_bytes()
    (results / ".env.production").write_text("PRIVATE=secret")
    assert drive_cli.main(["push-benchmark-results", *common]) == 2
    assert not (drive / "benchmarks/results/experiment/.env.production").exists()
    assert drive_cli.main(["push-benchmark-results", "--drive-root", str(drive), "--local-path", str(results), "--name", "../outside"]) == 2
