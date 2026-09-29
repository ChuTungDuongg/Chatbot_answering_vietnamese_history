"""Resumable fixed-TEST six-system Qwen3 answer-quality benchmark.

One question is generated at a time. Retrieval results are cached per backend;
no-RAG systems never load retrieval artifacts. GPU and Qdrant are never touched
by --dry-run. No backend fallback is permitted.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import HYBRID_MODEL_ID, settings
from app.rag.prompting import build_messages, build_no_rag_messages
from evaluation.gpu_memory import CudaMemory, append_sample
from evaluation.metrics.grounding import score_grounding
from evaluation.metrics.retrieval import score_retrieval
from evaluation.report import summarize
from evaluation.run_retrieval import _json_default, _source, file_sha
from evaluation.runner import evaluate_one
from evaluation.schema import Question, load_questions

SYSTEM_NAMES = ("vanilla_no_rag", "vanilla_faiss", "vanilla_qdrant",
                "sft_no_rag", "sft_faiss", "sft_qdrant")
DEFAULT_TEST = Path("evaluation/datasets/v1_silver/splits/v1_seed42/test.jsonl")
FROZEN_CANONICAL_SHA = "97acf491e7409e54ed27f38f1e28a16b9e85107c9c2ed296bffaf48814d539d8"
FROZEN_TEST_SHA = "f0a32500ef83e42f7d8748e02e41752c61fc4491afdb248dbe65c3198ec0b2a5"


class ProvenanceMismatch(RuntimeError):
    """A changed model or artifact cannot be counted as a question failure."""


@dataclass(frozen=True)
class System:
    name: str
    model_type: str
    dense_backend: str | None

    @property
    def rag(self) -> bool:
        return self.dense_backend is not None


SYSTEMS = {name: System(name, "sft" if name.startswith("sft_") else "vanilla",
                         "faiss" if name.endswith("_faiss") else "qdrant" if name.endswith("_qdrant") else None)
           for name in SYSTEM_NAMES}


def select_systems(raw: str) -> list[System]:
    names = SYSTEM_NAMES if raw == "all" else tuple(dict.fromkeys(raw.split(",")))
    if not names or any(name not in SYSTEMS for name in names):
        raise ValueError(f"--systems must be all or comma-separated names from {SYSTEM_NAMES}")
    return [SYSTEMS[name] for name in names]


def validate_test(path: Path) -> tuple[list[Question], dict]:
    if path.name != "test.jsonl":
        raise ValueError("Primary six-way benchmark requires frozen test.jsonl")
    manifest_path = path.parent / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen split manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("seed") != 42 or
        manifest.get("canonical_dataset_sha256") != FROZEN_CANONICAL_SHA or
        manifest.get("split_sha256", {}).get("test") != file_sha(path) or
        file_sha(path) != FROZEN_TEST_SHA):
        raise ValueError("Frozen TEST split identity mismatch")
    questions = load_questions(path)
    if [item.id for item in questions] != manifest.get("ids", {}).get("test"):
        raise ValueError("TEST row order/IDs differ from frozen split manifest")
    return questions, manifest


def file_tree_sha(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory missing: {path}")
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(file.relative_to(path).as_posix().encode())
            digest.update(file_sha(file).encode())
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    # A killed process can leave one unfinished final line. Only that line is discarded.
    with path.open("r+b") as stream:
        data = stream.read()
        if data and not data.endswith(b"\n"):
            end = data.rfind(b"\n") + 1
            stream.truncate(end)
            data = data[:end]
    rows = {}
    for line in data.splitlines():
        value = json.loads(line)
        key = value["question_id"]
        if key in rows:
            raise ValueError(f"Duplicate persisted question ID: {key}")
        rows[key] = value
    return rows


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def score_prediction(question: Question, prediction: dict, system: System) -> dict:
    record = evaluate_one(question, prediction)
    record["system"] = system.name
    record["difficulty"] = question.difficulty
    record["answerable"] = question.answerable
    record["in_domain"] = question.in_domain
    record["latency_ms"] = prediction.get("latency_ms")
    # Answerability and required-fact checks apply to every generated answer.
    behavior = score_grounding(prediction.get("answer") or "", question.question, prediction.get("sources") or [],
                               required_facts=question.required_facts,
                               answerable=question.answerable, in_domain=question.in_domain)
    for key in ("required_fact_phrase_recall", "insufficient_answer_behavior", "ood_refusal_behavior", "unnecessary_refusal"):
        record["answer_metrics"][key] = behavior[key] if prediction.get("success") else None
    record["answer_metrics"]["false_premise_correction_behavior"] = (
        float(bool(re.search(r"\b(?:không|sai|thực tế|nhầm|khác với)\b", prediction.get("answer") or "", re.I)))
        if prediction.get("success") and question.question_type == "false_premise" else None)
    if system.rag:
        return record
    # Evidence and citation checks are inapplicable when no retrieval occurred.
    record["retrieval"] = score_retrieval([], relevant_chunk_ids=None, relevant_source_ids=None)
    record["grounding"] = {key: None for key in record["grounding"]}
    record["citations"] = {key: None for key in record["citations"]}
    return record


def aggregate(questions: list[Question], predictions: dict[str, dict], system: System) -> dict:
    rows = [score_prediction(question, predictions[question.id], system)
            for question in questions if question.id in predictions]
    overall = summarize(rows)
    breakdown = {}
    for field in ("difficulty", "answerable", "category"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for question in questions:
            if question.id in predictions:
                groups[str(getattr(question, field))].append(score_prediction(question, predictions[question.id], system))
        breakdown[field] = {name: summarize(group) for name, group in groups.items()}
    latencies = [float(row["latency_ms"]) for row in predictions.values()
                 if row.get("success") and isinstance(row.get("latency_ms"), (int, float))]
    return {"system": system.name, "model_type": system.model_type, "rag": system.rag,
            "dense_backend": system.dense_backend, "question_count": len(rows),
            "failure_count": sum(not row["success"] for row in rows),
            "latency_ms_mean": sum(latencies) / len(latencies) if latencies else None,
            "overall": overall, "breakdown": breakdown,
            "warning": "SILVER labels are automatically annotated; lexical answer metrics do not prove historical correctness."}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--systems", default="all")
    parser.add_argument("--response-mode", choices=("concise", "standard", "detailed"), default="standard")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--sft-model-path", type=Path,
                        help="Optional merged local model for SFT systems instead of --adapter-path")
    parser.add_argument("--model-id", default=HYBRID_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--corpus-path", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("artifacts/corpus_v1/retrieval"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--start-after")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--retrieval-batch-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _identity(args, systems: list[System], split: dict) -> dict:
    if args.model_id != HYBRID_MODEL_ID:
        raise ValueError(f"Six-way baseline requires {HYBRID_MODEL_ID}")
    if any(system.model_type == "sft" for system in systems):
        if (args.adapter_path is None) == (args.sft_model_path is None):
            raise ValueError("SFT systems require exactly one of --adapter-path or --sft-model-path")
        if args.adapter_path and not (args.adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError("SFT adapter_config.json missing")
        if args.sft_model_path and not (args.sft_model_path / "config.json").is_file():
            raise FileNotFoundError("Merged SFT config.json missing")
    if any(system.dense_backend == "qdrant" for system in systems):
        manifest = args.retrieval_root / "qdrant" / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError("Qdrant V1 index is not finalized; retrieval/qdrant/manifest.json is missing.")
    for backend in {system.dense_backend for system in systems if system.rag}:
        path = args.retrieval_root / backend / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Dense backend manifest missing: {path}")
    if min(args.generation_batch_size, args.retrieval_batch_size, args.concurrency) != 1 or max(args.generation_batch_size, args.retrieval_batch_size, args.concurrency) != 1:
        raise ValueError("Primary quality evaluation currently requires batch sizes and concurrency of 1")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.max_new_tokens < 1 or args.final_k < 1:
        raise ValueError("Invalid decoding or final-k setting")
    if args.resume and args.overwrite:
        raise ValueError("Choose either --resume or --overwrite")
    dense = {backend: file_sha(args.retrieval_root / backend / "manifest.json")
             for backend in {system.dense_backend for system in systems if system.rag}}
    bm25_path = args.retrieval_root / "bm25s_index" / "phase9_manifest.json"
    runtime_path = args.corpus_path.parent / "runtime" / "manifest.json"
    runtime_manifest = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else None
    if dense and runtime_manifest is None:
        raise FileNotFoundError(f"V1 runtime manifest missing: {runtime_path}")
    if dense and not bm25_path.is_file():
        raise FileNotFoundError(f"BM25 manifest missing: {bm25_path}")
    return {"schema_version": 1, "test_sha256": file_sha(args.test_file),
            "canonical_sha256": split["canonical_dataset_sha256"], "split_seed": split["seed"],
            "systems": [system.name for system in systems], "response_mode": args.response_mode,
            "model_id": args.model_id, "model_revision": args.model_revision,
            "adapter_fingerprint": file_tree_sha(args.adapter_path) if args.adapter_path else None,
            "merged_model_fingerprint": file_tree_sha(args.sft_model_path) if args.sft_model_path else None,
            "corpus_path": str(args.corpus_path.resolve()),
            "corpus_sha256": (runtime_manifest or {}).get("corpus", {}).get("corpus_sha256"),
            "ordered_chunk_id_sha256": (runtime_manifest or {}).get("corpus", {}).get("ordered_chunk_id_sha256"),
            "corpus_count": (runtime_manifest or {}).get("corpus", {}).get("count"),
            "embedding_model_id": (runtime_manifest or {}).get("embedding_model_id"),
            "embedding_revision": (runtime_manifest or {}).get("embedding_model_resolved_revision"),
            "reranker_model_id": (runtime_manifest or {}).get("reranker_model_id"),
            "reranker_revision": (runtime_manifest or {}).get("reranker_model_revision"),
            "retrieval_settings": (runtime_manifest or {}).get("retrieval_settings"),
            "runtime_manifest_sha256": file_sha(runtime_path) if runtime_manifest else None,
            "qdrant_collection": settings.qdrant_collection if "qdrant" in dense else None,
            "qdrant_hnsw_ef": settings.qdrant_hnsw_ef if "qdrant" in dense else None,
            "dense_manifest_sha256": dense,
            "bm25_manifest_sha256": file_sha(bm25_path) if bm25_path.is_file() else None,
            "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens, "final_k": args.final_k,
            "start_after": args.start_after, "max_questions": args.max_questions,
            "generation_batch_size": args.generation_batch_size,
            "retrieval_batch_size": args.retrieval_batch_size, "concurrency": args.concurrency}


async def _generate(model, messages: list[dict], args) -> tuple[str, Any]:
    return await model.generate(messages, max_new_tokens=args.max_new_tokens)


def _load_cache(path: Path, fingerprint: dict, *, resume: bool) -> dict[str, dict]:
    meta = path.with_suffix(".manifest.json")
    if meta.exists():
        if json.loads(meta.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError(f"Retrieval cache identity mismatch: {meta}")
        return _read_jsonl(path)
    if path.exists():
        raise ValueError(f"Retrieval cache has no manifest: {path}")
    _atomic_json(meta, fingerprint)
    return {}


async def run(args: argparse.Namespace) -> dict:
    questions, split = validate_test(args.test_file)
    systems = select_systems(args.systems)
    if args.start_after:
        ids = [question.id for question in questions]
        if args.start_after not in ids:
            raise ValueError(f"--start-after ID absent from TEST: {args.start_after}")
        questions = questions[ids.index(args.start_after) + 1:]
    if args.max_questions is not None:
        if args.max_questions < 1:
            raise ValueError("--max-questions must be positive")
        questions = questions[:args.max_questions]
    identity = _identity(args, systems, split)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "questions": len(questions), "identity": identity}, indent=2))
        return {"identity": identity}
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.overwrite:
            (args.output_dir / "gpu_memory.jsonl").unlink(missing_ok=True)
            for system in systems:
                (args.output_dir / system.name / "predictions.jsonl").unlink(missing_ok=True)
                (args.output_dir / system.name / "progress.log").unlink(missing_ok=True)
            (args.output_dir / "retrieval_cache_faiss.jsonl").unlink(missing_ok=True)
            (args.output_dir / "retrieval_cache_qdrant.jsonl").unlink(missing_ok=True)
            (args.output_dir / "retrieval_cache_faiss.manifest.json").unlink(missing_ok=True)
            (args.output_dir / "retrieval_cache_qdrant.manifest.json").unlink(missing_ok=True)
        elif not args.resume:
            raise FileExistsError(f"Existing run; use --resume or --overwrite: {args.output_dir}")
        elif old.get("identity") != identity:
            raise ValueError("Resume identity/config mismatch")
    elif args.resume:
        raise FileNotFoundError(f"No run manifest for --resume: {manifest_path}")
    if not manifest_path.exists() or args.overwrite:
        try:
            import torch
            hardware = {"platform": platform.platform(), "gpus": [torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count())]}
        except ImportError:
            hardware = {"platform": platform.platform(), "gpus": []}
        _atomic_json(manifest_path, {"identity": identity, "created_at": datetime.now(timezone.utc).isoformat(),
                                     "git_commit": _git_commit(), "hardware": hardware,
                                     "status": "running"})
    elif args.resume:
        # Code provenance is additive metadata, never part of experiment identity.
        saved_run = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_commit = _git_commit()
        if current_commit and current_commit != saved_run.get("git_commit"):
            history = saved_run.setdefault("resume_git_commits", [])
            if current_commit not in history:
                history.append(current_commit)
                _atomic_json(manifest_path, saved_run)
    random.seed(args.seed)
    memory = CudaMemory(args.device)
    memory_path = args.output_dir / "gpu_memory.jsonl"
    settings.corpus_path_override = args.corpus_path
    settings.retrieval_root = args.retrieval_root
    settings.app_mode = "retrieval-only"
    summaries = {}
    for system in systems:
        system_start_memory = memory.sample()
        output = args.output_dir / system.name
        predictions_path = output / "predictions.jsonl"
        predictions = _read_jsonl(predictions_path)
        unknown = set(predictions) - {question.id for question in questions}
        if unknown:
            raise ValueError(f"Persisted predictions outside selected TEST subset: {sorted(unknown)[:3]}")
        service = retriever = model = None
        cache = {}
        cache_path = args.output_dir / f"retrieval_cache_{system.dense_backend}.jsonl"
        if system.rag:
            settings.retrieval_dense_backend = system.dense_backend
            cache_identity = {"test_sha256": identity["test_sha256"],
                              "backend": system.dense_backend,
                              "dense_manifest_sha256": identity["dense_manifest_sha256"][system.dense_backend],
                              "bm25_manifest_sha256": identity["bm25_manifest_sha256"],
                              "corpus_sha256": identity["corpus_sha256"], "final_k": args.final_k,
                              "runtime_manifest_sha256": identity["runtime_manifest_sha256"],
                              "qdrant_collection": identity["qdrant_collection"],
                              "qdrant_hnsw_ef": identity["qdrant_hnsw_ef"]}
            cache = _load_cache(cache_path, cache_identity, resume=args.resume)
        try:
            if any(question.id not in predictions for question in questions):
                from app.models.qwen import QwenRuntime
                merged = args.sft_model_path if system.model_type == "sft" else None
                model = QwenRuntime(model_id=str(merged) if merged else args.model_id,
                                    revision=None if merged else args.model_revision,
                                    device=args.device, dtype=args.dtype,
                                    adapter_path=args.adapter_path if system.model_type == "sft" else None,
                                    do_sample=args.temperature > 0, temperature=args.temperature,
                                    top_p=args.top_p, enable_thinking=False,
                                    allow_local_model=bool(merged), local_files_only=bool(merged))
                if system.rag and any(question.id not in predictions and question.id not in cache for question in questions):
                    from app.rag.retrieval import HybridRetriever
                    from app.services.rag_service import RAGService
                    service = RAGService()
                    service.load()  # Validates full corpus/index identity and remote Qdrant count.
                    retriever = HybridRetriever(service)
            started = time.monotonic()
            for index, question in enumerate(questions, 1):
                if question.id in predictions:
                    continue
                memory.reset_peaks()
                before_generation = after_generation = None
                began = time.perf_counter()
                row: dict[str, Any] = {"question_id": question.id, "system": system.name,
                                       "response_mode": args.response_mode, "success": False,
                                       "error": None, "sources": [], "answer": None}
                try:
                    retrieval_ms = 0.0
                    if system.rag:
                        cached = cache.get(question.id)
                        if cached is None:
                            if retriever is None:
                                from app.rag.retrieval import HybridRetriever
                                from app.services.rag_service import RAGService
                                service = RAGService(); service.load(); retriever = HybridRetriever(service)
                            retrieval_start = time.perf_counter()
                            result = retriever.retrieve(question.question, args.final_k)
                            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
                            sources = [_source(item, rank) for rank, item in enumerate(result.get("final_context") or [], 1)]
                            cached = {"question_id": question.id, "sources": sources,
                                      "retrieval_ms": retrieval_ms,
                                      "diagnostics": {"query_variants": result.get("query_variants"),
                                                      "is_ood": result.get("is_ood")}}
                            _append(cache_path, cached)
                            cache[question.id] = cached
                        sources = cached["sources"]
                        row["sources"] = sources
                        row["retrieval_ms"] = cached["retrieval_ms"]
                        row["retrieval_reused"] = question.id in cache and retrieval_ms == 0
                        messages = build_messages(question.question, sources, response_mode=args.response_mode)
                    else:
                        messages = build_no_rag_messages(question.question, response_mode=args.response_mode)
                    before_generation = memory.sample()
                    generation_start = time.perf_counter()
                    try:
                        answer, done = await _generate(model, messages, args)
                    finally:
                        after_generation = memory.sample()
                    if done.model_revision:
                        saved_run = json.loads(manifest_path.read_text(encoding="utf-8"))
                        previous_revision = saved_run.get("resolved_model_revision")
                        if previous_revision and previous_revision != done.model_revision:
                            raise ProvenanceMismatch("Resolved base-model revision changed during benchmark")
                        if not previous_revision:
                            saved_run["resolved_model_revision"] = done.model_revision
                            _atomic_json(manifest_path, saved_run)
                    row.update({"answer": answer, "success": bool(answer.strip()),
                                "generation_ms": (time.perf_counter() - generation_start) * 1000,
                                "input_tokens": done.input_tokens, "output_tokens": done.output_tokens,
                                "model_id": done.model_id, "model_revision": done.model_revision})
                    if not row["success"]:
                        row["error"] = "Empty model answer"
                except ProvenanceMismatch:
                    raise
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                # Keep only scalar/string prediction data across questions. The
                # Qwen stream has joined its worker before this point.
                messages = answer = done = result = None
                row["latency_ms"] = (time.perf_counter() - began) * 1000
                _append(predictions_path, row)
                predictions[question.id] = row
                # Periodic collection bounds cached/fragmented CUDA blocks while
                # avoiding allocator synchronization on every question.
                cleaned = index % 10 == 0
                if cleaned:
                    gc.collect()
                    memory.empty_cache()
                after_cleanup = memory.sample()
                append_sample(memory_path, {"event": "question", "system": system.name,
                    "question_index": index, "question_total": len(questions),
                    "question_id": question.id, "before_generation": before_generation,
                    "after_generation": after_generation, "after_cleanup": after_cleanup,
                    "cache_cleanup": cleaned, "success": row["success"]})
                _atomic_json(output / "progress.json", {"completed": len(predictions), "total": len(questions),
                                                         "last_question_id": question.id})
                with (output / "progress.log").open("a", encoding="utf-8") as log:
                    log.write(f"{datetime.now(timezone.utc).isoformat()} {index}/{len(questions)} {question.id} "
                              f"success={row['success']} latency_ms={row['latency_ms']:.1f}\n")
                elapsed = time.monotonic() - started
                remaining = (len(questions) - index) * elapsed / max(index, 1)
                successes = sum(value.get("success") is True for value in predictions.values())
                latencies = [float(value["latency_ms"]) for value in predictions.values()
                             if isinstance(value.get("latency_ms"), (int, float))]
                average_ms = sum(latencies) / len(latencies) if latencies else 0.0
                print(f"[{system.name}] {index}/{len(questions)} id={question.id} "
                      f"retrieval_ms={row.get('retrieval_ms', 0):.1f} generation_ms={row.get('generation_ms', 0):.1f} "
                      f"success={row['success']} successes={successes} failures={len(predictions)-successes} "
                      f"avg_latency_ms={average_ms:.1f} ETA_s={remaining:.0f}", flush=True)
                if index % 10 == 0 and after_cleanup is not None:
                    print(f"[GPU] system={system.name} q={index}/{len(questions)} "
                          f"allocated={after_cleanup['allocated_mib']:.0f}MiB "
                          f"reserved={after_cleanup['reserved_mib']:.0f}MiB "
                          f"peak_allocated={after_cleanup['peak_allocated_mib']:.0f}MiB "
                          f"peak_reserved={after_cleanup['peak_reserved_mib']:.0f}MiB", flush=True)
            summary = aggregate(questions, predictions, system)
            summaries[system.name] = summary
            _atomic_json(output / "metrics.json", summary)
        finally:
            before_teardown = memory.sample()
            try:
                if service is not None:
                    service.shutdown()
            finally:
                del model, retriever, service
                gc.collect()
                memory.empty_cache()
                after_teardown = memory.sample()
                append_sample(memory_path, {"event": "system_teardown", "system": system.name,
                    "system_start": system_start_memory, "before_teardown": before_teardown,
                    "after_teardown": after_teardown})
                if system_start_memory and after_teardown:
                    residual = after_teardown["allocated_mib"] - system_start_memory["allocated_mib"]
                    peak = before_teardown["peak_allocated_mib"] if before_teardown else 0
                    if residual > max(256, 0.1 * peak):
                        print(f"[GPU] WARNING system={system.name} teardown retained "
                              f"{residual:.0f}MiB allocated above system start; inspect gpu_memory.jsonl",
                              flush=True)
    result = {"identity": identity, "systems": summaries,
              "completed_at": datetime.now(timezone.utc).isoformat()}
    _atomic_json(args.output_dir / "summary.json", result)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("system", "model_type", "rag", "dense_backend", "question_count",
                                                     "failure_count", "latency_ms_mean", "exact_match", "token_f1", "rouge_l_f1"))
        writer.writeheader()
        for summary in summaries.values():
            answer = summary["overall"]["answer"]
            writer.writerow({key: summary.get(key) for key in writer.fieldnames if key not in ("exact_match", "token_f1", "rouge_l_f1")} |
                            {key: answer.get(key, {}).get("value") for key in ("exact_match", "token_f1", "rouge_l_f1")})
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved["status"] = ("complete_with_failures" if any(item["failure_count"] for item in summaries.values())
                       else "complete")
    saved["completed_at"] = result["completed_at"]
    _atomic_json(manifest_path, saved)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        asyncio.run(run(build_parser().parse_args(argv)))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"[six-way] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
