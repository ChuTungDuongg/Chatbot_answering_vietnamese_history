"""Manual deployment smoke through the existing production HTTP/SSE endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request
from uuid import uuid4

from benchmarks.latency.report import write_json
from benchmarks.latency.runner import _server_metadata
from benchmarks.latency.sse_client import measure_request


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-backend", choices=["transformers", "vllm"], required=True)
    parser.add_argument("--mode", choices=["hybrid", "central"], default="hybrid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--question", default="Trình bày nguyên nhân, diễn biến chính và ý nghĩa của chiến thắng Bạch Đằng năm 938.")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    client = f"inference-smoke-{uuid4().hex}"
    root = args.base_url.rstrip("/")
    for name in ("health", "ready"):
        with urllib.request.urlopen(f"{root}/{name}", timeout=args.timeout) as response:
            value = json.load(response)
        write_json(args.output / f"{name}.json", value)
        if name == "health" and value.get("status") != "ok":
            raise RuntimeError("Health failed")
        if name == "ready" and (not value.get("ready") or not value.get(f"{args.mode}_loaded")):
            raise RuntimeError("Configured model is not ready")
        print(f"PASS /{name}", flush=True)
    metadata = _server_metadata(root, client, args.timeout, args.mode)
    write_json(args.output / "metadata.json", metadata)
    if metadata.get("inference_backend") != args.expected_backend:
        raise RuntimeError("Wrong deployed inference backend")
    print(f"PASS metadata backend={args.expected_backend}; GPU={metadata['server_hardware']['gpu_model']}", flush=True)
    record = measure_request(base_url=root, question={"id": "http-inference-smoke", "question": args.question},
                             mode=args.mode, client_id=client, timeout=args.timeout,
                             phase="warm", run_index=0, strict_contract=True,
                             expected_model_id=metadata["model_id"])
    write_json(args.output / "stream.json", record)
    if (not record["success"] or record["inference_backend"] != args.expected_backend
            or record["answer_delta_count"] < 2 or not any(record["inter_token_latency_ms"])):
        raise RuntimeError(f"Incremental SSE smoke failed: {record['error']}")
    print(f"PASS SSE: {record['answer_delta_count']} deltas; {record['output_tokens']} tokens; finish={record.get('finish_reason')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
