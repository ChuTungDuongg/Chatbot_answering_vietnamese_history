"""Isolated real-model telemetry checks and warm paired V1 retrieval benchmark."""

import json
import os
from pathlib import Path
import tempfile

from modal_app import app, image, artifacts, hf_cache, runtime_secrets


image = image.add_local_python_source("modal_app", "scripts")


@app.function(image=image, gpu="A100", cpu=4, memory=32768, timeout=1800, startup_timeout=900,
              volumes={"/artifacts": artifacts, "/hf-cache": hf_cache}, secrets=runtime_secrets)
def runtime_telemetry_smoke(variant: str = "vanilla", repeats: int = 10) -> str:
    if variant not in {"vanilla", "sft"} or repeats < 0:
        raise ValueError("variant must be vanilla/sft and repeats >= 0")
    os.environ["CHAT_DATABASE_PATH"] = str(Path(tempfile.mkdtemp()) / "telemetry-smoke.sqlite3")
    os.environ["MODEL_VARIANT"] = variant
    from fastapi.testclient import TestClient
    from app.main import app as api
    from app.config import settings
    from scripts.benchmark_dynamic_retrieval import percentiles

    world_query = "tóm tắt lịch sử thế giới từ thế kỷ X tới hiện nay"
    local_query = "Ngô Quyền giành chiến thắng trên sông Bạch Đằng vào năm nào?"
    report = {"variant": variant, "benchmark": {"status": "skipped", "reason": "repeats=0"}, "requests": []}
    with TestClient(api) as client:
        ready = client.get("/ready").json()
        if str(settings.corpus_path) != "/artifacts/corpus_v1/chunks.jsonl":
            raise RuntimeError("Smoke must use V1 corpus")
        resources = tuple(id(getattr(api.state.rag_service, name)) for name in
                          ("chunks", "bm25", "embedder", "reranker", "faiss_index"))
        report["corpus_identity"] = ready["corpus_identity"]
        available = ready.get("retrieval", {}).get("available_backends", [])
        if repeats and set(available) != {"faiss", "qdrant"}:
            report["benchmark"] = {"status": "skipped", "reason": "Both validated FAISS/Qdrant lanes not available"}
        elif repeats:
            samples = []
            for question in (world_query, local_query):
                for iteration in range(repeats + 2):
                    for backend in (("faiss", "qdrant") if iteration % 2 == 0 else ("qdrant", "faiss")):
                        response = client.post("/api/v1/retrieve", json={
                            "question": question, "retrieval_backend": backend, "final_k": 3})
                        response.raise_for_status()
                        row = response.json()
                        if row["retrieval_backend"] != backend:
                            raise RuntimeError("Wrong request-local backend")
                        if iteration >= 2:
                            samples.append({"query": question, "backend": backend,
                                            "dense_ms": row["timings_ms"].get("dense_search"),
                                            "retrieval_ms": row["latency_ms"]})
            summary = []
            for question in (world_query, local_query):
                for backend in ("faiss", "qdrant"):
                    rows = [r for r in samples if r["query"] == question and r["backend"] == backend]
                    summary.append({"query": question, "backend": backend, "samples": len(rows),
                                    "dense_ms": percentiles([r["dense_ms"] for r in rows]),
                                    "retrieval_ms": percentiles([r["retrieval_ms"] for r in rows])})
            report["benchmark"] = {"status": "done", "units": "ms", "warmup_rounds": 2,
                                   "repeats": repeats, "summary": summary, "samples": samples}
        cases = [("hybrid", local_query)]
        if variant == "vanilla":
            cases += [("central", local_query), ("central", world_query)]
        for mode, question in cases:
            backend = "qdrant" if "qdrant" in available else "faiss"
            owner = "isolated-runtime-telemetry-smoke"
            headers = {"X-Client-ID": owner}
            conversation = client.post("/api/v1/conversations", headers=headers, json={}).json()
            response = client.post("/api/v1/chat/stream", headers=headers, json={
                "conversation_id": conversation["id"], "question": question,
                "mode": mode, "retrieval_backend": backend, "final_k": 3, "debug": True})
            response.raise_for_status()
            events = []
            for frame in response.text.split("\n\n"):
                lines = frame.splitlines()
                if len(lines) >= 2:
                    events.append((lines[0][7:], json.loads(lines[1][6:])))
            if not events or events[0][0] != "status" or events[-1][1].get("status") != "done":
                raise RuntimeError("SSE request did not complete")
            debug = next(data for name, data in events if name == "debug_trace")
            done = events[-1][1]
            answer = "".join(data["delta"] for name, data in events if name == "answer_delta")
            saved = api.state.chat_store.list_messages(owner, conversation["id"])[-1]
            if not answer or saved["content"] != answer or saved["debug_trace"] != debug:
                raise RuntimeError("Streamed/saved answer or trace mismatch")
            if done["metrics"]["e2e_ms"] is None or debug["performance"] != done["metrics"]:
                raise RuntimeError("Completed trace metrics mismatch")
            expected_budget = settings.hybrid_max_new_tokens if mode == "hybrid" else settings.central_final_max_new_tokens
            if debug["generation"]["settings"]["max_new_tokens"] != expected_budget:
                raise RuntimeError("Incorrect actual generation budget")
            if done["metrics"]["finish_reason"] not in {"stop", "length"}:
                raise RuntimeError("Real Qwen termination was not identified")
            if mode == "central":
                if not debug["planning"]["rounds"] or any(r["model_ms"] is None for r in debug["planning"]["rounds"]):
                    raise RuntimeError("Real planner timing missing")
                if question == world_query and not any(r["name"] != "search_history" for r in debug["tool_trace"]):
                    raise RuntimeError("World history was routed exclusively to Vietnamese corpus")
            report["requests"].append({"question": question, "mode": mode, "backend": backend,
                                        "answer": answer, "trace": debug, "generation_settings": done["generation_settings"]})
        if resources != tuple(id(getattr(api.state.rag_service, name)) for name in
                              ("chunks", "bm25", "embedder", "reranker", "faiss_index")):
            raise RuntimeError("Shared retrieval resources changed")
        report["resources_reused"] = True
        report["status"] = "passed"
    return json.dumps(report, ensure_ascii=False, indent=2)
