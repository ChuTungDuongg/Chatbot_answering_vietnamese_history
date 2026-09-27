"""Dense-only FAISS/Qdrant exact/HNSW comparison; no BM25, RRF or reranker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from scripts.retrieval.build_index import check_existing, scan_corpus
from scripts.retrieval.qdrant_index import VECTOR_NAME, make_client


KS = (10, 20, 50)


def compare_vector(vector, faiss_index, client, collection: str, *, hnsw_ef: int | None) -> dict:
    from qdrant_client import models

    k = min(max(KS), int(faiss_index.ntotal))
    started = time.perf_counter()
    faiss_scores, faiss_ids = faiss_index.search(np.asarray(vector, dtype="float32").reshape(1, -1), k)
    faiss_ms = (time.perf_counter() - started) * 1000
    faiss_ranked = [(int(row), float(score)) for row, score in zip(faiss_ids[0], faiss_scores[0]) if int(row) >= 0]

    def qdrant_search(exact: bool):
        started = time.perf_counter()
        result = client.query_points(collection_name=collection,
                                     query=np.asarray(vector, dtype="float32").tolist(),
                                     using=VECTOR_NAME, limit=k, with_payload=False,
                                     with_vectors=False,
                                     search_params=models.SearchParams(exact=exact,
                                                                        hnsw_ef=None if exact else hnsw_ef))
        return [(int(hit.id), float(hit.score)) for hit in result.points], (time.perf_counter() - started) * 1000

    exact, exact_ms = qdrant_search(True)
    ann, ann_ms = qdrant_search(False)
    faiss_set = {row for row, _ in faiss_ranked}
    exact_scores = dict(exact)
    faiss_scores_by_id = dict(faiss_ranked)
    common = faiss_set & set(exact_scores)
    metrics = {}
    for cutoff in KS:
        reference = {row for row, _ in faiss_ranked[:cutoff]}
        exact_ids = {row for row, _ in exact[:cutoff]}
        ann_ids = {row for row, _ in ann[:cutoff]}
        denominator = len(reference)
        metrics[f"faiss_qdrant_exact_overlap@{cutoff}"] = len(reference & exact_ids) / denominator if denominator else None
        metrics[f"ann_recall@{cutoff}"] = len(reference & ann_ids) / denominator if denominator else None
        metrics[f"qdrant_exact_hnsw_overlap@{cutoff}"] = (
            len(exact_ids & ann_ids) / len(exact_ids) if exact_ids else None)
    return {**metrics, "max_exact_score_delta_on_common_ids":
            max((abs(faiss_scores_by_id[row] - exact_scores[row]) for row in common), default=None),
            "latency_ms": {"faiss": faiss_ms, "qdrant_exact": exact_ms, "qdrant_hnsw": ann_ms},
            "hnsw_ef": hnsw_ef}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No dense queries to compare")
    metric_names = [key for key in rows[0] if key.startswith((
        "ann_recall@", "faiss_qdrant_exact_overlap@", "qdrant_exact_hnsw_overlap@"))]
    result = {name: statistics.mean(row[name] for row in rows if row[name] is not None)
              for name in metric_names}
    for backend in ("faiss", "qdrant_exact", "qdrant_hnsw"):
        values = np.array([row["latency_ms"][backend] for row in rows])
        result[f"{backend}_latency_p50_ms"] = float(np.percentile(values, 50))
        result[f"{backend}_latency_p95_ms"] = float(np.percentile(values, 95))
    result["query_count"] = len(rows)
    result["max_exact_score_delta_on_common_ids"] = max(
        (row["max_exact_score_delta_on_common_ids"] for row in rows
         if row["max_exact_score_delta_on_common_ids"] is not None), default=None)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path,
                        help="JSONL with a question field; use a fixed evaluation set")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--hnsw-ef", type=int, required=True)
    args = parser.parse_args(argv)
    if args.hnsw_ef < 1:
        raise ValueError("--hnsw-ef must be positive")
    out = args.output_dir.resolve()
    scan = scan_corpus(args.corpus.resolve())
    check_existing(out, scan)
    faiss_manifest = json.loads((out / "faiss" / "manifest.json").read_text(encoding="utf-8"))
    qdrant_manifest = json.loads((out / "qdrant" / "manifest.json").read_text(encoding="utf-8"))
    revision = faiss_manifest.get("embedding_model_resolved_revision")
    if not revision:
        raise RuntimeError("Dense manifests lack a pinned embedding revision")
    import faiss
    from sentence_transformers import SentenceTransformer
    from app.rag.retrieval import query_for_embedding

    index = faiss.read_index(str(out / "faiss" / "chunks.index"))
    if index.ntotal != faiss_manifest["count"]:
        raise RuntimeError("FAISS count differs from manifest")
    client = make_client(args.qdrant_url, os.getenv("QDRANT_API_KEY"))
    collection = qdrant_manifest["collection_name"]
    if int(client.count(collection, exact=True).count) != index.ntotal:
        raise RuntimeError("Qdrant count differs from FAISS")
    model = SentenceTransformer(faiss_manifest["embedding_model_id"], revision=revision)
    if model.get_sentence_embedding_dimension() != faiss_manifest["embedding_dimension"]:
        raise RuntimeError("Query E5 dimension differs from dense manifests")
    rows = []
    for line in args.queries.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question = json.loads(line)["question"]
        vector = model.encode([query_for_embedding(question)], normalize_embeddings=True,
                              convert_to_numpy=True)[0].astype("float32")
        rows.append(compare_vector(vector, index, client, collection, hnsw_ef=args.hnsw_ef))
    print(json.dumps({"level": "dense_backend", "corpus_sha256": faiss_manifest["corpus_sha256"],
                      "ordered_chunk_id_sha256": faiss_manifest["ordered_chunk_id_sha256"],
                      "embedding_model_revision": revision,
                      "faiss_index_file_bytes": (out / "faiss" / "chunks.index").stat().st_size,
                      "faiss_build_seconds": faiss_manifest.get("build_duration_seconds"),
                      "qdrant_build_seconds": qdrant_manifest.get("build_duration_seconds"),
                      "qdrant_storage_bytes": None,
                      "qdrant_storage_note": "Measure persistent server or Cloud storage externally; the collection API does not expose an equivalent file size.",
                      "metrics": summarize(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
