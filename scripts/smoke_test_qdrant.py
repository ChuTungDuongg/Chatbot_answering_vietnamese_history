"""Read-only V1 Qdrant smoke test against saved corpus/FAISS identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
from urllib.parse import urlparse


def check_qdrant(settings, *, compare_faiss: bool = True) -> dict:
    from qdrant_client import QdrantClient, models
    from scripts.retrieval.qdrant_index import point_payload

    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise RuntimeError("Missing QDRANT_URL/QDRANT_API_KEY; set local environment or Modal Secret")
    manifest = json.loads(settings.qdrant_manifest_path.read_text(encoding="utf-8"))
    if settings.qdrant_collection != manifest["collection_name"]:
        raise RuntimeError("Configured collection differs from V1 manifest")
    parsed = urlparse(settings.qdrant_url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        raise RuntimeError("QDRANT_URL requires an HTTP(S) hostname")
    try:
        socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 6333))
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key.get_secret_value(), timeout=30)
        info = client.get_collection(settings.qdrant_collection)
        count = int(client.count(settings.qdrant_collection, exact=True).count)
        vector_count = int(client.count(settings.qdrant_collection, exact=True,
            count_filter=models.Filter(must=[models.HasVectorCondition(
                has_vector=manifest["vector_name"])] )).count)
    except Exception as exc:
        # Exception text may contain URL credentials or HTTP headers; report only class.
        raise RuntimeError(f"Qdrant network/authentication/collection check failed ({type(exc).__name__})") from None
    try:
        vector = info.config.params.vectors[manifest["vector_name"]]
        assert int(vector.size) == manifest["embedding_dimension"] == 768
        assert str(vector.distance).casefold().split(".")[-1] == "cosine"
        assert info.config.quantization_config is None
        assert count == manifest["point_count"] == 624288
        assert vector_count == count
        assert str(info.status).casefold().split(".")[-1] == "green"
    except (AssertionError, KeyError, TypeError):
        raise RuntimeError("Qdrant collection schema/count/status incompatible with restored V1") from None
    ids = [0, 1, 12345, 100000, 312144, 500000, count - 1]
    rows = {}
    with settings.corpus_path.open(encoding="utf-8") as stream:
        row_id = 0
        for line in stream:
            if not line.strip():
                continue
            if row_id in ids:
                rows[row_id] = json.loads(line)
            row_id += 1
    if row_id != count:
        raise RuntimeError("Local V1 row count differs from remote count")
    index = None
    if compare_faiss:
        import faiss
        index = faiss.read_index(str(settings.faiss_path))
        if index.ntotal != count or index.d != 768:
            raise RuntimeError("Restored FAISS schema differs")
    try:
        points = client.retrieve(settings.qdrant_collection, ids=ids, with_payload=True, with_vectors=True)
        if {int(p.id) for p in points} != set(ids):
            raise RuntimeError("Remote V1 sample point IDs missing")
        samples = []
        import numpy as np
        for point in points:
            expected = point_payload(rows[int(point.id)])
            if any((point.payload or {}).get(k) != v for k, v in expected.items()) or not set(expected).issubset(point.payload or {}):
                raise RuntimeError(f"V1 payload mismatch at point {point.id}")
            values = np.asarray(point.vector[manifest["vector_name"]], dtype="float32")
            if len(values) != 768 or not np.isfinite(values).all():
                raise RuntimeError("Invalid remote vector")
            difference = None
            if index is not None:
                difference = float(np.max(np.abs(values - index.reconstruct(int(point.id)))))
                if difference > 1e-6:
                    raise RuntimeError(f"Remote vector differs from preserved FAISS at point {point.id}")
            samples.append({"point_id": int(point.id), "payload": "PASS", "max_abs_diff_vs_faiss": difference})
        results = client.query_points(settings.qdrant_collection, query=points[0].vector[manifest["vector_name"]],
            using=manifest["vector_name"], limit=3, with_payload=True,
            search_params=models.SearchParams(exact=True)).points
        if not results or any(not (p.payload or {}).get("chunk_id") for p in results):
            raise RuntimeError("Basic vector search returned no valid evidence")
        return {"hostname": host, "collection": settings.qdrant_collection, "count": count,
                "vector_count": vector_count, "indexed_vectors_count": info.indexed_vectors_count,
                "dimension": int(vector.size), "distance": "COSINE", "vector_name": manifest["vector_name"],
                "connectivity": "PASS", "authentication": "PASS", "v1_compatibility": "PASS (7 payload/vector samples)",
                "samples": samples, "search_hits": [{"id": int(p.id), "chunk_id": p.payload["chunk_id"], "score": p.score} for p in results]}
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Qdrant sample/search check failed ({type(exc).__name__})") from None
    finally:
        client.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-faiss-comparison", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/restore/qdrant_smoke.json"))
    args = parser.parse_args(argv)
    from app.config import settings, REPO_ROOT
    result = check_qdrant(settings, compare_faiss=not args.skip_faiss_comparison)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
