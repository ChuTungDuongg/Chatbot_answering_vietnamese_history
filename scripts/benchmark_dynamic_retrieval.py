"""Small paired latency sanity check against a warm, shared app runtime."""

import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx


def percentiles(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"p50": None, "p95": None}
    def percentile(p):
        position = (len(values) - 1) * p
        index = int(position)
        return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (position - index)
    return {"p50": percentile(.5), "p95": percentile(.95)}


def run(client, *, queries=None, repeats=3, warmup=1, mode="hybrid"):
    queries = queries or ["Ngô Quyền giành chiến thắng trên sông Bạch Đằng vào năm nào?"]
    ready = client.get("/ready"); ready.raise_for_status(); ready = ready.json()
    if set(ready.get("retrieval", {}).get("available_backends", [])) != {"faiss", "qdrant"}:
        raise RuntimeError("Both validated FAISS/Qdrant backends must be available for a paired benchmark")
    headers = {"X-Client-ID": f"dynamic-sanity-{uuid4()}"}
    samples = []
    for iteration in range(warmup + repeats):
        # Alternate order so one lane does not always get the warmer cache.
        backends = ["faiss", "qdrant"] if iteration % 2 == 0 else ["qdrant", "faiss"]
        for question in queries:
            for backend in backends:
                conversation = client.post("/api/v1/conversations", headers=headers, json={})
                conversation.raise_for_status(); conversation_id = conversation.json()["id"]
                started = time.perf_counter_ns()
                try:
                    with client.stream("POST", "/api/v1/chat/stream", headers=headers, json={
                        "conversation_id": conversation_id, "question": question, "mode": mode,
                        "retrieval_backend": backend, "final_k": 3, "debug": True}) as response:
                        response.raise_for_status()
                        event = None; done = None; stages = []; answer = ""
                        for line in response.iter_lines():
                            if line.startswith("event: "): event = line[7:]
                            elif line.startswith("data: "):
                                data = json.loads(line[6:])
                                if event == "error": raise RuntimeError(data.get("message", "Request failed"))
                                if event == "status": stages.append(data)
                                if event == "answer_delta": answer += data.get("delta", "")
                                if event == "done": done = data
                        if not done or done.get("status") != "done" or done.get("retrieval_backend") != backend or not answer:
                            raise RuntimeError("Incomplete or incorrect backend response")
                    if iteration >= warmup:
                        samples.append({"backend": backend, "mode": mode, "query": question,
                            "request_id": done["request_id"], "metrics": done["metrics"],
                            "client_e2e_ms": (time.perf_counter_ns() - started) / 1e6,
                            "status_events": len(stages), "stages": stages,
                            "model_id": done.get("model_id"), "model_revision": done.get("model_revision"),
                            "generation_settings": done.get("generation_settings"),
                            "retrieval_settings": done.get("retrieval_settings")})
                finally:
                    cleanup = client.delete(f"/api/v1/conversations/{conversation_id}", headers=headers)
                    cleanup.raise_for_status()
    summary = {}
    for backend in ("faiss", "qdrant"):
        rows = [row for row in samples if row["backend"] == backend]
        summary[backend] = {"samples": len(rows), **{name: percentiles([
            row["metrics"].get(metric) for row in rows]) for name, metric in (
                ("dense_ms", "dense_search_ms"), ("retrieval_ms", "retrieval_ms"),
                ("final_answer_ttft_ms", "final_answer_ttft_ms"), ("e2e_ms", "e2e_ms"))}}
    return {"purpose": "latency sanity; not a frozen research benchmark", "units": "ms", "mode": mode,
            "warmup_rounds": warmup, "repeats": repeats, "corpus_identity": ready.get("corpus_identity"),
            "summary": summary, "samples": samples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="HTTP URL of the app with both validated backends")
    parser.add_argument("--queries", type=Path, help="UTF-8 JSONL containing question fields (default: one short history question)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mode", choices=["hybrid", "central"], default="hybrid")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, help="New report path, preferably under gitignored reports/")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0: parser.error("repeats >= 1 and warmup >= 0 required")
    if args.output and args.output.exists(): parser.error("Output already exists; choose a new report path")
    queries = ([json.loads(line)["question"] for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
               if args.queries else None)
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        result = run(client, queries=queries, repeats=args.repeats, warmup=args.warmup, mode=args.mode)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
