"""Validate corpus and completed FAISS/Qdrant/BM25S index identities."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.retrieval.build_index import assert_dense_manifests_match, check_existing, scan_corpus


def validate(corpus: Path, out: Path, *, components: tuple[str, ...],
             qdrant_url: str | None = None, qdrant_collection: str | None = None,
             qdrant_client=None, faiss_module=None) -> dict:
    scan = scan_corpus(corpus)
    check_existing(out, scan)
    assert_dense_manifests_match(out)
    checked = {}
    if "faiss" in components:
        if faiss_module is None:
            import faiss as faiss_module
        path = out / "faiss" / "chunks.index"
        if not path.is_file():
            raise FileNotFoundError(f"FAISS index missing: {path}")
        checked["faiss"] = int(faiss_module.read_index(str(path)).ntotal)
    if "bm25" in components:
        path = out / "bm25s_index" / "params.index.json"
        if not path.is_file():
            raise FileNotFoundError(f"BM25S params missing: {path}")
        checked["bm25"] = int(json.loads(path.read_text(encoding="utf-8"))["num_docs"])
    if "qdrant" in components:
        from scripts.retrieval.qdrant_index import make_client

        path = out / "qdrant" / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Qdrant manifest missing: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        collection = qdrant_collection or manifest["collection_name"]
        if collection != manifest["collection_name"]:
            raise RuntimeError("Requested Qdrant collection differs from manifest")
        client = qdrant_client or make_client(qdrant_url, os.getenv("QDRANT_API_KEY"))
        checked["qdrant"] = int(client.count(collection, exact=True).count)
        info = client.get_collection(collection)
        if str(info.status).casefold().split(".")[-1] != "green":
            raise RuntimeError(f"Qdrant collection is not ready for evaluation: {info.status}")
        vector = info.config.params.vectors.get("dense_e5")
        if (vector is None or int(vector.size) != manifest["embedding_dimension"] or
                str(vector.distance).casefold().split(".")[-1] != "cosine" or
                info.config.quantization_config is not None):
            raise RuntimeError("Qdrant vector schema differs from manifest")
        saved_hnsw = manifest.get("hnsw_configuration") or {}
        if any(saved_hnsw.get(key) != getattr(info.config.hnsw_config, key)
               for key in ("m", "ef_construct", "full_scan_threshold")):
            raise RuntimeError("Qdrant HNSW configuration differs from manifest")
    for component, actual in checked.items():
        if actual != scan["chunk_count"]:
            raise RuntimeError(f"{component} count mismatch: {actual} != {scan['chunk_count']}")
    return {"corpus_chunk_count": scan["chunk_count"],
            "corpus_sha256": scan["corpus_sha256"],
            "ordered_chunk_id_sha256": scan["ordered_chunk_id_sha256"],
            "component_counts": checked, "validated": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--components", choices=("local", "all-backends"), default="all-backends")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION"))
    args = parser.parse_args(argv)
    components = ("faiss", "bm25") if args.components == "local" else ("faiss", "qdrant", "bm25")
    print(json.dumps(validate(args.corpus.resolve(), args.output_dir.resolve(), components=components,
                              qdrant_url=args.qdrant_url,
                              qdrant_collection=args.qdrant_collection), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
