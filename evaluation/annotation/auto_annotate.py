"""Build a resumable SILVER benchmark; dry-run is the default and makes no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.generate_candidates import candidate_pools, scaled_targets
from evaluation.annotation.silver_logic import (JsonChatProvider, PROMPT_VERSION, adjudicate_deep,
                                                adjudicate_retrieval, candidate_evidence,
                                                check_generation, semantic_call)
from evaluation.annotation.silver_selection import (FAMOUS, balanced_subset, duplicate_reason,
                                                    next_category, next_difficulty, select_seed, usable)
from evaluation.annotation.policy import fold
from evaluation.annotation.silver_store import SilverStore, digest, exclusive_run
from evaluation.annotation.workspace import now


DEFAULT_CONFIG = Path("configs/silver_benchmark_v1.json")
DEFAULT_WORKSPACE = Path("evaluation/annotation/workspace_silver_v1")


def call_estimate(target: int, deep_target: int, attempts: int, config: dict[str, Any]) -> dict[str, int]:
    batches = (config["max_evidence_chunks"] + config["semantic_batch_size"] - 1) // config["semantic_batch_size"]
    return {"questions": target, "candidate_attempt_limit": attempts,
            "generation_calls_nominal": target, "relevance_calls_nominal": target * batches,
            "verification_calls_nominal": target, "contradiction_calls_upper": target,
            "deep_calls_nominal": deep_target * 2,
            "total_nominal_upper": target * (3 + batches) + deep_target * 2,
            "total_attempt_upper": attempts * (3 + batches) + deep_target * 2}


def manifest_identity(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def provenance(args: argparse.Namespace, config: dict[str, Any], lookup: CorpusLookup,
               service: Any, qdrant_enabled: bool) -> dict[str, Any]:
    root = args.retrieval_root
    manifest = service.manifest or {}
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                             stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = None
    return {"corpus_sha256": lookup.corpus_sha256, "corpus_count": lookup.count,
            "ordered_chunk_id_sha256": service.ordered_chunk_id_sha256,
            "faiss_manifest_sha256": manifest_identity(root / "faiss" / "manifest.json"),
            "bm25_manifest_sha256": manifest_identity(root / "bm25s_index" / "phase9_manifest.json"),
            "qdrant_manifest_sha256": manifest_identity(root / "qdrant" / "manifest.json") if qdrant_enabled else None,
            "embedding_model_id": service.config["retrieval"]["embedding_model_id"],
            "embedding_revision": service.embedding_revision,
            "reranker_model_id": service.config["retrieval"]["reranker_model_id"],
            "reranker_revision": service.config["retrieval"].get("reranker_model_revision"),
            "retrieval_settings": manifest.get("retrieval_settings"),
            "adjudicator_provider": args.provider, "adjudicator_model": args.model,
            "adjudicator_revision": args.model_revision, "temperature": 0,
            "annotation_policy_version": PROMPT_VERSION,
            "generator_config_sha256": digest(config), "target": args.target,
            "deep_target": args.deep_target, "max_attempts": args.max_attempts,
            "dense_mode": "both" if qdrant_enabled else "faiss-only", "git_commit": git_commit}


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any | None]:
    # The production retrieval loader validates corpus, FAISS/BM25 manifests and counts.
    os.environ.update({"APP_MODE": "retrieval-only", "CORPUS_PATH": str(args.corpus),
                       "RETRIEVAL_ROOT": str(args.retrieval_root), "RETRIEVAL_DENSE_BACKEND": "faiss"})
    import app.services.rag_service as service_module
    from app.config import Settings
    from app.rag.retrieval import HybridRetriever

    service_module.settings = Settings()
    service = service_module.RAGService()
    service.load()
    qdrant = None
    finalized = (args.retrieval_root / "qdrant" / "manifest.json").is_file()
    credentials_ready = all((service_module.settings.qdrant_url,
                             service_module.settings.qdrant_api_key,
                             service_module.settings.qdrant_collection))
    if args.dense_mode == "both" and not (finalized and credentials_ready):
        service.shutdown()
        raise RuntimeError("Dual retrieval requires finalized Qdrant manifest and QDRANT_URL/QDRANT_API_KEY/QDRANT_COLLECTION")
    if args.dense_mode != "faiss-only" and finalized and credentials_ready:
        from app.rag.artifact_identity import fingerprint, validate_v1_manifests
        expected = {"count": len(service.chunks), "corpus_sha256": service.corpus_sha256,
                    "ordered_chunk_id_sha256": service.ordered_chunk_id_sha256}
        validate_v1_manifests(args.retrieval_root, expected, include_qdrant=True)
        if service.manifest["manifest_fingerprints"].get("qdrant") != fingerprint(service_module.settings.qdrant_manifest_path):
            service.shutdown()
            raise RuntimeError("Qdrant manifest differs from runtime metadata; regenerate runtime metadata")
        original = service.dense_retriever
        service._load_qdrant()  # Reuse production exact-count/vector/HNSW validation.
        qdrant = service.dense_retriever
        service.dense_retriever = original
    return service, HybridRetriever(service), qdrant


def retrieve_evidence(question: str, retriever: Any, service: Any, qdrant: Any | None,
                      origin_id: str | None, config: dict[str, Any],
                      paired_id: str | None = None) -> dict[str, Any]:
    result = retriever.retrieve(question, include_stage_diagnostics=True)
    raw = result.get("annotation_stages") or {}
    stages: dict[str, list[dict[str, Any]]] = {}
    for name, output in (("dense", "faiss"), ("bm25", "bm25"),
                         ("rrf", "rrf"), ("reranked", "reranked")):
        stages[output] = [{"chunk_id": service.chunks[item["row_id"]]["chunk_id"],
                           "rank": item["rank"], "score": item.get("score"),
                           "query": item.get("query"), "row_id": item["row_id"]}
                          for item in raw.get(name, [])]
    if paired_id:
        stages["origin"] = [{"chunk_id": origin_id, "rank": 0},
                            {"chunk_id": paired_id, "rank": 0}]
    if qdrant is not None:
        from app.rag.retrieval import query_for_embedding
        embedding = service.embedder.encode([query_for_embedding(question)], convert_to_numpy=True,
                                            normalize_embeddings=True).astype("float32")[0]
        stages["qdrant"] = [{"chunk_id": hit.chunk_id, "rank": hit.rank, "score": hit.score,
                              "row_id": hit.row_id}
                             for hit in qdrant.search(embedding, retriever.dense_fetch_k)]
    mixed = candidate_evidence(stages, origin_id, limit=config["max_evidence_chunks"])
    return {"stages": stages, "inspection": mixed,
            "retrieval_gate": result.get("domain_gate_result"),
            "retrieval_query_variants": result.get("query_variants")}


def process_retrieval(row: dict[str, Any], store: SilverStore, provider: Any,
                      retriever: Any, service: Any, qdrant: Any | None,
                      lookup: CorpusLookup, config: dict[str, Any], retrieval_lock: threading.Lock) -> None:
    identifier = row["candidate_id"]
    if row["evidence"] is None:
        with retrieval_lock:
            evidence = retrieve_evidence(row["generated"]["question"], retriever, service, qdrant,
                                         row["seed"].get("chunk_id"), config,
                                         (row["seed"].get("paired") or {}).get("chunk_id"))
        store.save(identifier, evidence=evidence)
    else:
        evidence = row["evidence"]
    labels = adjudicate_retrieval(store, provider, identifier, row["generated"],
                                  evidence["inspection"], lookup, config,
                                  full_stages=evidence["stages"])
    store.save(identifier, labels=labels, annotation_status=labels["annotation_status"],
               confidence=labels["confidence"]["level"])


def process_deep(row: dict[str, Any], store: SilverStore, provider: Any,
                 lookup: CorpusLookup, config: dict[str, Any]) -> None:
    deep = adjudicate_deep(store, provider, row, lookup, config)
    store.save(row["candidate_id"], deep=deep,
               deep_status="auto_reviewed" if deep["annotation_status"] == "auto_reviewed"
               else "needs_human_review")


def ensure_lookup(corpus: Path, directory: Path) -> CorpusLookup:
    path = directory / "corpus_lookup.sqlite3"
    if path.is_file():
        return CorpusLookup(corpus, path)
    runtime = corpus.parent / "runtime" / "manifest.json"
    identity = json.loads(runtime.read_text(encoding="utf-8"))["corpus"] if runtime.is_file() else {}
    return CorpusLookup.build(corpus, path, expected_sha=identity.get("corpus_sha256"),
                              expected_count=identity.get("count"))


def run(args: argparse.Namespace, config: dict[str, Any], provider: Any,
        service: Any, retriever: Any, qdrant: Any | None, lookup: CorpusLookup,
        store: SilverStore, pools: dict[str, list[dict[str, Any]]]) -> None:
    retrieval_lock = threading.Lock()
    processed = 0
    if args.phase == "deep" and len(usable(store.rows())) < args.target:
        raise RuntimeError("Deep phase requires the retrieval SILVER target first")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        while args.phase != "deep" and len(usable(store.rows())) < args.target:
            rows = store.rows()
            if args.max_questions and processed >= args.max_questions:
                break
            capacity = min(args.concurrency, args.target - len(usable(rows)),
                           args.max_questions - processed if args.max_questions else args.concurrency)
            pending = [row for row in rows if row["annotation_status"] == "draft" and
                       row["seq"] > args.start_after]
            batch = pending[:capacity]
            while len(batch) < capacity and len(rows) < args.max_attempts and (
                    not args.max_questions or processed + len(batch) < args.max_questions):
                category = next_category(rows, config, args.target)
                difficulty = next_difficulty(rows, config, args.target, category)
                seed = select_seed(category, pools, rows, config)
                if seed is None and category != "out_of_domain":
                    raise RuntimeError(f"No diverse corpus seed remains for {category}; adjust limits explicitly")
                paired = None
                if category in {"comparison", "multi_hop"}:
                    used_pair_sources = {r["seed"].get("source_id") for r in rows} | {
                        (r["seed"].get("paired") or {}).get("source_id") for r in rows}
                    paired = next((item for item in pools.get("_general", [])
                                   if item["source_id"] != seed["source_id"] and
                                   item["source_id"] not in used_pair_sources and
                                   item["title"] != seed["title"] and
                                   item["chunk_id"] not in {r["seed"].get("chunk_id") for r in rows}), None)
                    if paired is None:
                        raise RuntimeError("Two independent corpus sources are required for multi-source questions")
                spec = {**(seed or {}), "category": category, "difficulty": difficulty,
                        "paired": paired}
                row = store.insert_seed(len(rows) + 1, spec)
                rows.append(row)
                if row["seq"] <= args.start_after:
                    continue
                batch.append(row)
            if not batch:
                break
            ready = []
            for row in batch:
                if row["generated"] is None:
                    spec = row["seed"]
                    category, difficulty = spec["category"], spec["difficulty"]
                    payload = {"category": category, "difficulty": difficulty,
                               "origin": {k: v for k, v in spec.items() if k not in {"category", "difficulty"}},
                               "edge_case": category if category in
                               {"ambiguous", "insufficient_evidence", "false_premise", "out_of_domain"} else None,
                               "policy_version": PROMPT_VERSION}
                    generated = semantic_call(store, provider, row["candidate_id"], "generate", payload,
                                              check_generation)
                    generated = {**generated, "category": category,
                                 "difficulty": difficulty, "origin_chunk_id": spec.get("chunk_id"),
                                 "origin_source_id": spec.get("source_id"),
                                 "generation_metadata": {"provider": getattr(provider, "kind", "test"),
                                                         "model_id": getattr(provider, "model", None),
                                                         "revision": getattr(provider, "revision", None),
                                                         "temperature": 0, "prompt_version": PROMPT_VERSION}}
                    problem = duplicate_reason(generated["question"],
                                               [other for other in store.rows() if other["candidate_id"] != row["candidate_id"]])
                    famous_usable = sum(bool(FAMOUS.search(fold(other["generated"]["question"])))
                                        for other in usable(store.rows()))
                    if not problem and FAMOUS.search(fold(generated["question"])) and famous_usable >= config["max_famous_topics"]:
                        problem = "famous_topic_limit"
                    if (not problem and category != "out_of_domain" and
                            generated["question"].strip().casefold() == str(spec.get("title") or "").casefold()):
                        problem = "trivial_title_question"
                    if problem:
                        store.save(row["candidate_id"], generated=generated,
                                   annotation_status="rejected", rejection_reason=problem)
                        processed += 1
                        print(f"[silver] rejected={row['candidate_id']} reason={problem}", flush=True)
                        continue
                    store.save(row["candidate_id"], generated=generated)
                ready.append(store.get(row["candidate_id"]))
            futures = [executor.submit(process_retrieval, row, store, provider, retriever,
                                       service, qdrant, lookup, config, retrieval_lock) for row in ready]
            for row, future in zip(ready, futures):
                future.result()
                processed += 1
                current = store.get(row["candidate_id"])
                print(f"[silver] {len(usable(store.rows()))}/{args.target} id={row['candidate_id']} "
                      f"status={current['annotation_status']} confidence={current['confidence']}", flush=True)
    print(f"[silver] raw={store.count()} rejected={sum(row['annotation_status']=='rejected' for row in store.rows())} "
          f"usable={len(usable(store.rows()))} target={args.target}", flush=True)
    if args.phase == "retrieval" or len(usable(store.rows())) < args.target:
        return
    selected = balanced_subset(usable(store.rows()), len(usable(store.rows())), config["seed"])
    deep_processed = 0
    for row in selected:
        if sum(item["deep_status"] == "auto_reviewed" for item in store.rows()) >= args.deep_target:
            break
        current = store.get(row["candidate_id"])
        if current["deep_status"] == "not_selected":
            store.save(row["candidate_id"], deep_status="selected")
            current = store.get(row["candidate_id"])
        if current["deep_status"] != "selected":
            continue
        if args.max_questions and deep_processed >= args.max_questions:
            break
        process_deep(current, store, provider, lookup, config)
        deep_processed += 1
        print(f"[silver-deep] id={row['candidate_id']} status={store.get(row['candidate_id'])['deep_status']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("artifacts/corpus_v1/retrieval"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--deep-target", type=int, default=150)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--phase", choices=("retrieval", "deep", "all"), default="all")
    parser.add_argument("--dense-mode", choices=("auto", "faiss-only", "both"), default="auto")
    parser.add_argument("--provider", choices=("local", "external"), default="local")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model-revision")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-after", type=int, default=0)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.max_attempts = args.max_attempts or config.get("max_attempts", args.target * 3)
    if (args.target < 1 or args.deep_target < 0 or args.deep_target > args.target or
            args.max_attempts < args.target or args.concurrency < 1 or
            args.start_after < 0 or (args.max_questions is not None and args.max_questions < 1)):
        parser.error("Invalid SILVER target, attempt, resume or concurrency setting")
    if args.start_after and not args.resume:
        parser.error("--start-after requires --resume")
    estimate = call_estimate(args.target, args.deep_target, args.max_attempts, config)
    print("[silver] model-call estimate (upper planning estimate; not a price or probability): " +
          json.dumps(estimate, ensure_ascii=False), flush=True)
    if not args.execute:
        print("[silver] dry run complete; no corpus scan, model load, connection or paid call")
        return 0
    if args.provider == "external" and not args.execute_paid:
        parser.error("External provider requires --execute-paid after reviewing the call estimate")
    from dotenv import load_dotenv
    load_dotenv(override=False)
    base_url = os.getenv("SILVER_LLM_BASE_URL", "http://127.0.0.1:8001/v1")
    provider = JsonChatProvider(kind=args.provider, model=args.model,
                                revision=args.model_revision, base_url=base_url)
    args.corpus, args.retrieval_root = args.corpus.resolve(), args.retrieval_root.resolve()
    with exclusive_run(args.workspace):
        store = SilverStore(args.workspace)
        service = lookup = None
        try:
            if store.count() and not args.resume:
                raise RuntimeError("SILVER workspace already contains candidates; use --resume")
            if args.start_after > store.count():
                raise RuntimeError("--start-after exceeds existing candidate sequence")
            lookup = ensure_lookup(args.corpus, args.workspace)
            service, retriever, qdrant = load_runtime(args)
            identity = provenance(args, config, lookup, service, qdrant is not None)
            # Immutable run identity excludes mutable timestamp and git commit from resume checks.
            commit = identity.pop("git_commit")
            store.set_metadata("run_identity", identity)
            store.set_metadata("git_commit", store.metadata("git_commit") or commit)
            store.set_metadata("started_at", store.metadata("started_at") or now())
            targets = scaled_targets(config, args.target)
            pools = candidate_pools(args.corpus, targets, config["seed"]) if args.phase != "deep" else {}
            run(args, config, provider, service, retriever, qdrant, lookup, store, pools)
        finally:
            if service is not None:
                service.shutdown()
            if lookup is not None:
                lookup.close()
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
