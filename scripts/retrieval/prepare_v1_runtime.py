"""Validate existing Corpus V1 indexes and write only small runtime metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.rag.artifact_identity import fingerprint, validate_v1_manifests
from scripts.retrieval.build_index import scan_corpus


RERANKER_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RETRIEVAL_CONFIG = {
    "dense_fetch_k": 80, "bm25_fetch_k": 80, "rrf_k": 60,
    "rrf_top_k": 20, "final_context_k": 6, "rerank_batch_size": 32,
    "max_query_variants": 3, "query_expansion_weight": 0.82,
    "max_chunks_per_title": 2, "enable_context_diversity": True,
    "metadata_max_bonus": 0.18, "intent_facet_bonus": 0.025,
}


def prepare(corpus: Path, retrieval: Path, output: Path, *,
            reranker_id: str = RERANKER_ID,
            reranker_revision: str = RERANKER_REVISION) -> dict:
    corpus, retrieval = corpus.resolve(), retrieval.resolve()
    scan = scan_corpus(corpus)
    identity = {"count": scan["chunk_count"], "bytes": scan["corpus_bytes"],
                "corpus_sha256": scan["corpus_sha256"],
                "ordered_chunk_id_sha256": scan["ordered_chunk_id_sha256"]}
    manifests = validate_v1_manifests(retrieval, identity)
    if not (retrieval / "faiss" / "chunks.index").is_file():
        raise FileNotFoundError("FAISS index is missing")
    params = json.loads((retrieval / "bm25s_index" / "params.index.json").read_text(encoding="utf-8"))
    if int(params["num_docs"]) != identity["count"]:
        raise RuntimeError("BM25 num_docs differs from corpus")
    backends = ["faiss"]
    qdrant_path = retrieval / "qdrant" / "manifest.json"
    if qdrant_path.is_file():
        manifests = validate_v1_manifests(retrieval, identity, include_qdrant=True)
        backends.append("qdrant")
    faiss = manifests["faiss"]
    fingerprints = {name: fingerprint(retrieval / relative) for name, relative in (
        ("faiss", "faiss/manifest.json"),
        ("bm25", "bm25s_index/phase9_manifest.json"),
        ("index", "index_manifest.json"))}
    if qdrant_path.is_file():
        fingerprints["qdrant"] = fingerprint(qdrant_path)
    runtime = {"schema_version": 1, "corpus": identity,
               "embedding_model_id": faiss["embedding_model_id"],
               "embedding_model_resolved_revision": faiss["embedding_model_resolved_revision"],
               "embedding_dimension": faiss["embedding_dimension"],
               "reranker_model_id": reranker_id, "reranker_model_revision": reranker_revision,
               "retrieval_settings": RETRIEVAL_CONFIG,
               "available_dense_backends": backends,
               "manifest_fingerprints": fingerprints}
    config = {"retrieval": {"embedding_model_id": faiss["embedding_model_id"],
                            "reranker_model_id": reranker_id,
                            "reranker_model_revision": reranker_revision,
                            **RETRIEVAL_CONFIG}}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("manifest.json", runtime), ("inference_config.json", config)):
        target = output / name
        temporary = output / (name + ".partial")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    return runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("artifacts/corpus_v1/retrieval"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/corpus_v1/runtime"))
    parser.add_argument("--reranker-model-id", default=RERANKER_ID)
    parser.add_argument("--reranker-revision", default=RERANKER_REVISION)
    args = parser.parse_args(argv)
    result = prepare(args.corpus, args.retrieval_root, args.output,
                     reranker_id=args.reranker_model_id,
                     reranker_revision=args.reranker_revision)
    print(json.dumps({"corpus_count": result["corpus"]["count"],
                      "available_dense_backends": result["available_dense_backends"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
