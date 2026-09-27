"""Run one headless retrieval pass per question with durable, resumable JSONL output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.rag.artifact_identity import fingerprint
from app.rag.retrieval import HybridRetriever
from app.services.rag_service import RAGService
from evaluation.metrics.retrieval import score_retrieval
from evaluation.schema import load_questions


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Unsupported retrieval diagnostic type: {type(value).__name__}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source(row: dict[str, Any], rank: int) -> dict[str, Any]:
    hits = row.get("retrieval_hits") or []
    def first_rank(prefix: str) -> int | None:
        values = [int(match.group(1)) for hit in hits
                  if (match := re.fullmatch(rf"{prefix}:q\d+@(\d+)", str(hit)))]
        return min(values) if values else None
    metadata = row.get("metadata") or {}
    source_id = row.get("source_id") or (metadata.get("source_sha1") if isinstance(metadata, dict) else None)
    if not source_id and row.get("hf_dataset") is not None and row.get("raw_record_index") is not None:
        source_id = f"{row['hf_dataset']}:{row['raw_record_index']}"
    return {"chunk_id": str(row["chunk_id"]), "source_id": source_id,
            "document_id": row.get("document_id"), "title": row.get("title"),
            "text": row.get("text"), "url": row.get("url"),
            "final_rank": rank, "dense_rank": first_rank("dense"),
            "bm25_rank": first_rank("bm25"), "rrf_rank": row.get("rrf_rank"),
            "reranker_rank": row.get("reranker_rank"),
            "best_dense_score": row.get("best_dense_score"),
            "best_bm25_score": row.get("best_bm25_score"),
            "rrf_score": row.get("rrf_score"),
            "reranker_score": row.get("reranker_score"),
            "final_retrieval_score": row.get("final_retrieval_score"),
            "metadata_bonus": row.get("metadata_bonus"),
            "retrieval_hits": hits, "retrieval_query_roles": row.get("retrieval_query_roles")}


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _rss_bytes() -> int | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        if os.name != "nt":
            return None
        import ctypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("page_faults", ctypes.c_ulong),
                        ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                        ("quota_peak_paged", ctypes.c_size_t), ("quota_paged", ctypes.c_size_t),
                        ("quota_peak_nonpaged", ctypes.c_size_t), ("quota_nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if memory_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.working_set
        return None


def metadata(dataset: Path, output: Path, service: RAGService, retriever: HybridRetriever,
             final_k: int) -> dict[str, Any]:
    backend = settings.retrieval_dense_backend
    dense_path = settings.faiss_manifest_path if backend == "faiss" else settings.qdrant_manifest_path
    return {"schema_version": 1, "run_id": output.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": file_sha(dataset), "dataset_path": str(dataset.resolve()),
            "dense_backend": backend, "qdrant_hnsw_ef": settings.qdrant_hnsw_ef if backend == "qdrant" else None,
            "corpus_count": len(service.chunks), "corpus_sha256": service.corpus_sha256,
            "corpus_hash": service.corpus_sha256,
            "ordered_chunk_id_sha256": service.ordered_chunk_id_sha256,
            "embedding_model_id": service.config["retrieval"]["embedding_model_id"],
            "embedding_model_resolved_revision": service.embedding_revision,
            "reranker_model_id": service.config["retrieval"]["reranker_model_id"],
            "reranker_model_revision": service.config["retrieval"].get("reranker_model_revision"),
            "dense_manifest_fingerprint": fingerprint(dense_path),
            "bm25_manifest_fingerprint": fingerprint(settings.bm25_manifest_path),
            "retrieval_index_hash": hashlib.sha256((fingerprint(dense_path) +
                fingerprint(settings.bm25_manifest_path)).encode("ascii")).hexdigest(),
            "index_manifest_fingerprint": fingerprint(settings.index_manifest_path) if settings.retrieval_root else None,
            "retrieval_settings": retriever.retrieval_config, "final_k": final_k,
            "git_commit": _git_commit(),
            "hardware": {"system": platform.system(), "machine": platform.machine(),
                         "processor": platform.processor(), "device": settings.device,
                         "process_rss_after_load_bytes": _rss_bytes()},
            "startup_timings_ms": service.startup_timings_ms}


COMPARE_FIELDS = ("dataset_sha256", "dense_backend", "qdrant_hnsw_ef", "corpus_count",
                  "corpus_sha256", "ordered_chunk_id_sha256", "embedding_model_id",
                  "embedding_model_resolved_revision", "reranker_model_id", "reranker_model_revision",
                  "dense_manifest_fingerprint", "bm25_manifest_fingerprint",
                  "index_manifest_fingerprint", "retrieval_settings", "final_k", "git_commit")


def _completed(path: Path) -> set[str]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        with path.open("r+b") as handle:
            handle.truncate(boundary)
        raw = raw[:boundary]
    seen = set()
    for number, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or question_id in seen:
            raise RuntimeError(f"Invalid or duplicate prediction at line {number}")
        seen.add(question_id)
    return seen


def _summary(predictions: Path, questions: list) -> dict[str, Any]:
    by_id = {item.id: item for item in questions}
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line]
    names = [f"{metric}@{k}" for k in (1, 3, 5, 10)
             for metric in ("hit_rate", "recall", "precision")]
    names.extend(("mrr@10", "ndcg@10"))
    metrics: dict[str, dict[str, list[float]]] = {
        level: {name: [] for name in names} for level in ("chunk", "source")}
    successes = 0
    for row in rows:
        if not row["success"]:
            continue
        successes += 1
        question = by_id[row["question_id"]]
        scores = score_retrieval(row["sources"], relevant_chunk_ids=question.relevant_chunk_ids,
                                 relevant_source_ids=question.relevant_source_ids)
        for level in ("chunk", "source"):
            for name in names:
                value = scores[level][name]
                if value is not None:
                    metrics[level][name].append(value)
    return {"question_count": len(questions), "completed_count": len(rows),
            "success_count": successes, "failure_count": len(rows) - successes,
            "retrieval_metrics": {level: {name: {"mean": sum(values) / len(values) if values else None,
                                                 "labeled_count": len(values)}
                                         for name, values in names.items()} for level, names in metrics.items()},
            "missing_labels": "N/A; no gold IDs were fabricated"}


def run(dataset: Path, output: Path, *, resume: bool = False, final_k: int = 10,
        service: RAGService | None = None, retriever: HybridRetriever | None = None) -> Path:
    if settings.app_mode != "retrieval-only":
        raise RuntimeError("Set APP_MODE=retrieval-only for headless retrieval evaluation")
    dataset = dataset.resolve()
    questions = load_questions(dataset)
    if not questions:
        raise ValueError("Dataset has no questions")
    output = output.resolve()
    predictions = output / "predictions.jsonl"
    metadata_path = output / "run_metadata.json"
    if resume:
        if not predictions.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Cannot resume without predictions.jsonl and run_metadata.json")
        if (output / "summary.json").is_file():
            raise RuntimeError("Run is already complete")
    else:
        output.mkdir(parents=True, exist_ok=False)
    own_service = service is None
    service = service or RAGService()
    try:
        service.load()
        retriever = retriever or HybridRetriever(service)
        current = metadata(dataset, output, service, retriever, final_k)
        if resume:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            for field in COMPARE_FIELDS:
                if previous.get(field) != current.get(field):
                    raise RuntimeError(f"Resume metadata mismatch: {field}")
            done = _completed(predictions)
            unknown = done - {item.id for item in questions}
            if unknown:
                raise RuntimeError("Predictions contain IDs absent from dataset")
        else:
            _write_json_atomic(metadata_path, current)
            predictions.touch(exist_ok=False)
            done = set()
        with predictions.open("a", encoding="utf-8") as out, (output / "progress.log").open("a", encoding="utf-8") as progress:
            for index, question in enumerate(questions, 1):
                if question.id in done:
                    continue
                started = time.perf_counter()
                try:
                    result = retriever.retrieve(question.question, final_k=final_k)
                    sources = [_source(row, rank) for rank, row in enumerate(result["final_context"], 1)]
                    row = {"question_id": question.id, "success": True, "error": None,
                           "phase": "warm", "mode": "retrieval-only", "sources": sources,
                           "retrieved": sources, "retrieval_diagnostics": {
                               key: value for key, value in result.items()
                               if key not in ("final_context", "candidates20", "target_retrieval_results")},
                           "candidate_diagnostics": [_source(item, rank) for rank, item in enumerate(result.get("candidates20", []), 1)]}
                except Exception as exc:
                    row = {"question_id": question.id, "success": False,
                           "error": f"retrieval failed ({type(exc).__name__})", "phase": "warm",
                           "mode": "retrieval-only", "sources": [], "retrieved": []}
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                     default=_json_default) + "\n")
                out.flush()
                os.fsync(out.fileno())
                message = f"[eval] {index}/{len(questions)} id={question.id} backend={settings.retrieval_dense_backend} latency_ms={row['latency_ms']} success={row['success']}"
                print(message, flush=True)
                progress.write(message + "\n")
                progress.flush()
        summary = _summary(predictions, questions)
        _write_json_atomic(output / "summary.json", summary)
        print(f"[eval] {len(questions)}/{len(questions)} complete", flush=True)
        return output
    finally:
        if own_service:
            service.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New reports/retrieval/<run-id> directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--final-k", type=int, default=10)
    args = parser.parse_args(argv)
    if args.final_k < 1:
        parser.error("--final-k must be positive")
    run(args.dataset, args.output, resume=args.resume, final_k=args.final_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
