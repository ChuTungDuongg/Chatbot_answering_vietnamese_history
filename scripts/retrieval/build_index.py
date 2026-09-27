"""Build ordered E5/FAISS, E5/Qdrant, and shared BM25S indexes.

Dense input is batched. BM25S still materializes token IDs and sparse arrays;
its independent phase avoids holding the dense model and vectors at that time.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterator


MODEL_ID = "intfloat/multilingual-e5-base"
INDEX_VERSION = "corpus_v1_index_v2"
BUILDER_VERSION = "3"
PASSAGE_TEMPLATE = "passage: {title}\n{text}"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
SHA40 = re.compile(r"^[0-9a-f]{40}$")
PROTECTED = {"vn_history_deployment", "vn_history_modal", "old_corpus"}
REQUIRED_DENSE_MATCH_FIELDS = (
    "corpus_sha256", "ordered_chunk_id_sha256", "count",
    "embedding_model_id", "embedding_model_resolved_revision",
    "embedding_dimension", "normalize_embeddings", "passage_prefix",
    "query_prefix", "passage_template",
)
# Older valid FAISS manifests predate corpus_bytes; compare it when both save it.
OPTIONAL_DENSE_MATCH_FIELDS = ("corpus_bytes",)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def ordered_chunk_id_sha256(rows) -> str:
    """Keep the original ordered-ID hash contract for iterable rows."""
    digest = hashlib.sha256()
    for row in rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Every indexed row requires chunk_id")
        digest.update((chunk_id + "\n").encode("utf-8"))
    return digest.hexdigest()


class CorpusPass:
    def __init__(self, path: Path, *, unique: bool = False):
        self.path = path
        self.digest = hashlib.sha256()
        self.ids_digest = hashlib.sha256()
        self.count = self.size = self.total_chars = self.max_chars = self.near_empty = 0
        self.min_chars: int | None = None
        self.schema_versions: Counter[str] = Counter()
        # Exact duplicate detection costs O(unique IDs), never O(corpus text).
        self.seen: set[str] | None = set() if unique else None

    def rows(self) -> Iterator[dict[str, Any]]:
        with self.path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                self.digest.update(raw)
                self.size += len(raw)
                try:
                    row = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(f"Malformed JSONL at {self.path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {self.path}:{line_number}")
                chunk_id = row.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError(f"Missing chunk_id at {self.path}:{line_number}")
                if self.seen is not None:
                    if chunk_id in self.seen:
                        raise ValueError(f"Duplicate chunk_id {chunk_id!r} at {self.path}:{line_number}")
                    self.seen.add(chunk_id)
                content = row.get("text")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"Empty or invalid text at {self.path}:{line_number}")
                if not isinstance(row.get("title", ""), str):
                    raise ValueError(f"Invalid title at {self.path}:{line_number}")
                schema = row.get("schema_version")
                if schema not in (None, 1, 2):
                    raise ValueError(f"Unsupported schema at {self.path}:{line_number}")
                self.schema_versions[str(schema) if schema is not None else "unspecified"] += 1
                self.ids_digest.update((chunk_id + "\n").encode("utf-8"))
                self.count += 1
                chars = len(content)
                self.total_chars += chars
                self.max_chars = max(chars, self.max_chars)
                self.min_chars = chars if self.min_chars is None else min(chars, self.min_chars)
                self.near_empty += int(len(content.strip()) < 40)
                yield row

    def report(self) -> dict[str, Any]:
        if not self.count:
            raise ValueError(f"Cannot index an empty corpus: {self.path}")
        return {
            "chunk_count": self.count, "corpus_bytes": self.size,
            "corpus_sha256": self.digest.hexdigest(),
            "ordered_chunk_id_sha256": self.ids_digest.hexdigest(),
            "text_chars": {"min": self.min_chars, "max": self.max_chars,
                           "mean": round(self.total_chars / self.count, 2)},
            "near_empty_chunks_under_40_chars": self.near_empty,
            "schema_versions_observed": dict(sorted(self.schema_versions.items())),
            "missing_ids": 0, "duplicate_ids": 0, "malformed_rows": 0,
            "missing_or_empty_text": 0,
        }


def scan_corpus(path: Path) -> dict[str, Any]:
    tracker = CorpusPass(path, unique=True)
    for _ in tracker.rows():
        pass
    result = tracker.report()
    if result["corpus_bytes"] != path.stat().st_size:
        raise RuntimeError("Corpus size changed during preflight")
    return result


def batches(rows: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def assert_same_corpus(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = ("chunk_count", "corpus_bytes", "corpus_sha256", "ordered_chunk_id_sha256")
    if any(actual[key] != expected[key] for key in keys):
        raise RuntimeError("Corpus changed between preflight and index pass")


def safe_output(path: Path) -> Path:
    result = path.expanduser().resolve()
    if any(part.casefold() in PROTECTED for part in result.parts):
        raise ValueError("Refusing to write indexes inside frozen V0 or archived artifacts")
    return result


def output_paths(out: Path) -> dict[str, str]:
    return {
        "faiss_index": str(out / "faiss" / "chunks.index"),
        "faiss_manifest": str(out / "faiss" / "manifest.json"),
        "bm25_directory": str(out / "bm25s_index"),
        "bm25_manifest": str(out / "bm25s_index" / "phase9_manifest.json"),
        "qdrant_manifest": str(out / "qdrant" / "manifest.json"),
        "index_manifest": str(out / "index_manifest.json"),
    }


def selected_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    import torch
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable")
    return "cuda" if available else "cpu"


def preflight_report(corpus: Path, out: Path, scan: dict[str, Any], *,
                     dimension: int, batch_size: int, device: str,
                     requested_revision: str) -> dict[str, Any]:
    vector_bytes = scan["chunk_count"] * dimension * 4
    return {
        "schema_version": 1, "mode": "preflight", "corpus_path": str(corpus), **scan,
        "embedding_model_id": MODEL_ID,
        "embedding_model_requested_revision": requested_revision,
        "embedding_model_resolved_revision": None,
        "model_resolution_note": "Local preflight does not contact the model hub; FAISS build resolves and loads an immutable revision.",
        "embedding_dimension_estimate": dimension,
        "estimated_faiss_vector_bytes": vector_bytes,
        "estimated_faiss_vector_gib": round(vector_bytes / 2**30, 3),
        "minimum_ram_consideration": "FAISS vectors alone need the stated bytes in RAM; allow more for model, batch, library, OS, and BM25S separately.",
        "device": device, "embedding_batch_size": batch_size,
        "output_dir_exists": out.exists(),
        "components_already_present": {
            "faiss": (out / "faiss").exists(),
            "qdrant": (out / "qdrant").exists(),
            "bm25": (out / "bm25s_index").exists(),
        },
        "duplicate_detection_memory": "Exact preflight duplicate checking retains unique IDs in a Python set; text and rows are streamed.",
        "expected_outputs": output_paths(out),
    }


def resolve_model_revision(model_id: str, requested: str) -> str:
    from huggingface_hub import HfApi
    resolved = requested if SHA40.fullmatch(requested) else HfApi().model_info(model_id, revision=requested).sha
    if not isinstance(resolved, str) or not SHA40.fullmatch(resolved):
        raise RuntimeError("Embedding model revision did not resolve to an immutable SHA")
    return resolved


def builder_git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[2], check=False)
    value = result.stdout.strip()
    return value if result.returncode == 0 and SHA40.fullmatch(value) else None


def common_manifest(corpus: Path, scan: dict[str, Any]) -> dict[str, Any]:
    source_manifest = corpus.parent / "manifest.json"
    fingerprint = None
    if source_manifest.exists():
        fingerprint = json.loads(source_manifest.read_text(encoding="utf-8")).get("config_fingerprint")
    return {
        "schema_version": 1, "index_version": INDEX_VERSION,
        "builder_version": BUILDER_VERSION,
        "corpus_path": str(corpus), "corpus_sha256": scan["corpus_sha256"],
        "corpus_bytes": scan["corpus_bytes"],
        "corpus_chunk_count": scan["chunk_count"],
        "ordered_chunk_id_sha256": scan["ordered_chunk_id_sha256"],
        "source_corpus_config_fingerprint": fingerprint,
        "builder_git_sha": builder_git_sha(),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def check_existing(out: Path, scan: dict[str, Any]) -> None:
    for partial in (out / "faiss.partial", out / "qdrant.partial", out / "bm25s_index.partial",
                    out / "index_manifest.json.partial"):
        if partial.exists():
            raise RuntimeError(f"Incomplete index output exists: {partial}; inspect it before retrying")
    for directory, manifest_name, required_name in (
        (out / "faiss", "manifest.json", "chunks.index"),
        (out / "bm25s_index", "phase9_manifest.json", "params.index.json"),
    ):
        if directory.exists() and not (directory / manifest_name).is_file():
            raise RuntimeError(f"Incomplete index component exists: {directory}")
        if directory.exists() and not (directory / required_name).is_file():
            raise RuntimeError(f"Incomplete index component exists: {directory}")
    if (out / "qdrant").exists() and not (out / "qdrant" / "manifest.json").is_file():
        raise RuntimeError(f"Incomplete index component exists: {out / 'qdrant'}")
    for path in (out / "faiss" / "manifest.json", out / "qdrant" / "manifest.json",
                 out / "bm25s_index" / "phase9_manifest.json"):
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("count") != scan["chunk_count"] or
                    saved.get("corpus_sha256") != scan["corpus_sha256"] or
                    saved.get("ordered_chunk_id_sha256") != scan["ordered_chunk_id_sha256"]):
                raise RuntimeError(f"Existing index component differs from corpus: {path}")
    assert_dense_manifests_match(out)


def assert_dense_manifests_match(out: Path) -> None:
    paths = [out / "faiss" / "manifest.json", out / "qdrant" / "manifest.json"]
    if not all(path.exists() for path in paths):
        return
    faiss, qdrant = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for key in REQUIRED_DENSE_MATCH_FIELDS:
        if (key not in faiss or key not in qdrant or
                faiss[key] is None or qdrant[key] is None or
                faiss[key] != qdrant[key]):
            raise RuntimeError(f"FAISS/Qdrant manifest mismatch: {key}")
    for key in OPTIONAL_DENSE_MATCH_FIELDS:
        if key in faiss and key in qdrant and faiss[key] != qdrant[key]:
            raise RuntimeError(f"FAISS/Qdrant manifest mismatch: {key}")
    if faiss.get("shared_embedding_stream") and qdrant.get("shared_embedding_stream"):
        if faiss.get("embedding_stream_sha256") != qdrant.get("embedding_stream_sha256"):
            raise RuntimeError("FAISS/Qdrant shared embedding stream hash mismatch")


def add_dense_batches(corpus: Path, scan: dict[str, Any], model, *,
                      batch_size: int, dimension: int, index=None, qdrant_sink=None) -> str:
    import numpy as np
    tracker = CorpusPass(corpus)
    embedding_digest = hashlib.sha256()
    started = time.monotonic()
    for number, batch in enumerate(batches(tracker.rows(), batch_size), 1):
        first = tracker.count - len(batch)
        if number == 1 or number % 100 == 0 or tracker.count == scan["chunk_count"]:
            log(f"[dense] batch={number} rows={first}-{tracker.count - 1}")
        passages = [PASSAGE_TEMPLATE.format(title=row.get("title", ""), text=row["text"]) for row in batch]
        vectors = np.ascontiguousarray(model.encode(passages, normalize_embeddings=True,
                                                    batch_size=batch_size, convert_to_numpy=True,
                                                    show_progress_bar=False), dtype="float32")
        if vectors.shape != (len(batch), dimension):
            raise RuntimeError(f"Embedding shape mismatch: {vectors.shape}")
        embedding_digest.update(vectors.tobytes(order="C"))
        if index is not None:
            index.add(vectors)
        if qdrant_sink is not None:
            qdrant_sink.add(batch, vectors, first)
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = tracker.count / elapsed
        eta = (scan["chunk_count"] - tracker.count) / rate
        if number == 1 or number % 100 == 0 or tracker.count == scan["chunk_count"]:
            log(f"[dense] encoded={len(batch)} total={tracker.count}/{scan['chunk_count']} rate={rate:.2f} chunks/s elapsed={elapsed:.1f}s eta={eta:.1f}s")
    assert_same_corpus(tracker.report(), scan)
    if index is not None and index.ntotal != scan["chunk_count"]:
        raise RuntimeError("FAISS vector count differs from corpus count")
    if qdrant_sink is not None and qdrant_sink.count != scan["chunk_count"]:
        raise RuntimeError("Qdrant upsert count differs from corpus count")
    return embedding_digest.hexdigest()


def add_faiss_batches(corpus: Path, scan: dict[str, Any], index, model, *,
                      batch_size: int, dimension: int) -> str:
    return add_dense_batches(corpus, scan, model, batch_size=batch_size,
                             dimension=dimension, index=index)


def build_faiss(corpus: Path, out: Path, scan: dict[str, Any], *,
                requested_revision: str, resolved_revision: str, device: str,
                batch_size: int, model_factory=None, faiss_module=None) -> dict[str, Any]:
    if (out / "faiss").exists():
        raise FileExistsError(f"FAISS output already exists: {out / 'faiss'}")
    started = time.monotonic()
    if faiss_module is None:
        import faiss as faiss_module
    if model_factory is None:
        from sentence_transformers import SentenceTransformer
        model_factory = lambda: SentenceTransformer(MODEL_ID, revision=resolved_revision, device=device)
    log(f"[faiss] loading model={MODEL_ID} revision={resolved_revision} device={device}")
    model = model_factory()
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension < 1:
        raise RuntimeError("Embedding model reported no valid dimension")
    index = faiss_module.IndexFlatIP(dimension)
    embedding_hash = add_faiss_batches(corpus, scan, index, model,
                                       batch_size=batch_size, dimension=dimension)
    manifest = commit_faiss(corpus, out, scan, index, faiss_module,
                            requested_revision=requested_revision,
                            resolved_revision=resolved_revision, device=device,
                            batch_size=batch_size, dimension=dimension,
                            build_duration_seconds=time.monotonic() - started,
                            embedding_stream_sha256=embedding_hash,
                            shared_embedding_stream=False)
    del index, model
    return manifest


def commit_faiss(corpus: Path, out: Path, scan: dict[str, Any], index, faiss_module, *,
                 requested_revision: str, resolved_revision: str, device: str,
                 batch_size: int, dimension: int,
                 build_duration_seconds: float, embedding_stream_sha256: str,
                 shared_embedding_stream: bool) -> dict[str, Any]:
    manifest = {
        **common_manifest(corpus, scan), "count": scan["chunk_count"],
        "embedding_model_id": MODEL_ID,
        "embedding_model_requested_revision": requested_revision,
        "embedding_model_resolved_revision": resolved_revision,
        "embedding_dimension": dimension, "normalize_embeddings": True,
        "passage_prefix": PASSAGE_PREFIX, "query_prefix": QUERY_PREFIX,
        "passage_template": PASSAGE_TEMPLATE, "embedding_batch_size": batch_size,
        "device": device, "faiss_index_type": "IndexFlatIP",
        "build_duration_seconds": round(build_duration_seconds, 3),
        "embedding_stream_sha256": embedding_stream_sha256,
        "shared_embedding_stream": shared_embedding_stream,
    }
    stage = out / "faiss.partial"
    stage.mkdir()
    faiss_module.write_index(index, str(stage / "chunks.index"))
    (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(stage, out / "faiss")
    return manifest


def build_qdrant(corpus: Path, out: Path, scan: dict[str, Any], *,
                 requested_revision: str, resolved_revision: str, device: str,
                 batch_size: int, url: str | None, collection: str,
                 model_factory=None, client=None) -> dict[str, Any]:
    from scripts.retrieval.qdrant_index import QdrantSink, make_client
    started = time.monotonic()

    if model_factory is None:
        from sentence_transformers import SentenceTransformer
        model_factory = lambda: SentenceTransformer(MODEL_ID, revision=resolved_revision, device=device)
    if client is None:
        client = make_client(url, os.getenv("QDRANT_API_KEY"))
    log(f"[qdrant] loading model={MODEL_ID} revision={resolved_revision} device={device}")
    model = model_factory()
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension < 1:
        raise RuntimeError("Embedding model reported no valid dimension")
    sink = QdrantSink(client, collection, dimension, out)
    embedding_hash = add_dense_batches(corpus, scan, model, batch_size=batch_size,
                                       dimension=dimension, qdrant_sink=sink)
    manifest = sink.finish(common_manifest(corpus, scan), model_revision=resolved_revision,
                           requested_revision=requested_revision, batch_size=batch_size,
                           device=device, build_duration_seconds=time.monotonic() - started,
                           embedding_stream_sha256=embedding_hash,
                           shared_embedding_stream=False)
    del model
    return manifest


def build_dense_pair(corpus: Path, out: Path, scan: dict[str, Any], *,
                     requested_revision: str, resolved_revision: str, device: str,
                     batch_size: int, url: str | None, collection: str,
                     model_factory=None, faiss_module=None, client=None) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts.retrieval.qdrant_index import QdrantSink, make_client
    started = time.monotonic()

    if faiss_module is None:
        import faiss as faiss_module
    if model_factory is None:
        from sentence_transformers import SentenceTransformer
        model_factory = lambda: SentenceTransformer(MODEL_ID, revision=resolved_revision, device=device)
    if client is None:
        client = make_client(url, os.getenv("QDRANT_API_KEY"))
    log(f"[dense] loading shared model={MODEL_ID} revision={resolved_revision} device={device}")
    model = model_factory()
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension < 1:
        raise RuntimeError("Embedding model reported no valid dimension")
    index = faiss_module.IndexFlatIP(dimension)
    sink = QdrantSink(client, collection, dimension, out)
    embedding_hash = add_dense_batches(corpus, scan, model, batch_size=batch_size,
                                       dimension=dimension, index=index,
                                       qdrant_sink=sink)
    faiss_manifest = commit_faiss(corpus, out, scan, index, faiss_module,
                                  requested_revision=requested_revision,
                                  resolved_revision=resolved_revision, device=device,
                                  batch_size=batch_size, dimension=dimension,
                                  build_duration_seconds=time.monotonic() - started,
                                  embedding_stream_sha256=embedding_hash,
                                  shared_embedding_stream=True)
    qdrant_manifest = sink.finish(common_manifest(corpus, scan),
                                  model_revision=resolved_revision,
                                  requested_revision=requested_revision,
                                  batch_size=batch_size, device=device,
                                  build_duration_seconds=time.monotonic() - started,
                                  embedding_stream_sha256=embedding_hash,
                                  shared_embedding_stream=True)
    del model, index
    return faiss_manifest, qdrant_manifest


def build_bm25(corpus: Path, out: Path, scan: dict[str, Any], *, batch_size: int,
               bm25_module=None) -> dict[str, Any]:
    started = time.monotonic()
    if (out / "bm25s_index").exists():
        raise FileExistsError(f"BM25 output already exists: {out / 'bm25s_index'}")
    if bm25_module is None:
        import bm25s as bm25_module
    from app.rag.retrieval import match_norm
    log("[bm25] phase=tokenizing; BM25S retains token IDs and sparse arrays in RAM")
    tracker = CorpusPass(corpus)
    vocab: dict[str, int] = {}
    token_ids: list[list[int]] = []
    for number, batch in enumerate(batches(tracker.rows(), batch_size), 1):
        normalized = [match_norm(f"{row.get('title', '')} {row.get('title', '')} {row['text']}") for row in batch]
        tokens = bm25_module.tokenize(normalized, stopwords=None, stemmer=None,
                                      return_ids=False, show_progress=False)
        for document in tokens:
            token_ids.append([vocab.setdefault(token, len(vocab)) for token in document])
        if number == 1 or number % 100 == 0 or tracker.count == scan["chunk_count"]:
            log(f"[bm25] tokenized={tracker.count}/{scan['chunk_count']} vocab={len(vocab)}")
    assert_same_corpus(tracker.report(), scan)
    if len(token_ids) != scan["chunk_count"]:
        raise RuntimeError("BM25 token document count differs from corpus count")
    log("[bm25] phase=indexing")
    bm25 = bm25_module.BM25()
    bm25.index((token_ids, vocab), show_progress=False)
    del token_ids, vocab
    manifest = {
        **common_manifest(corpus, scan), "count": scan["chunk_count"],
        "embedding_model_id": MODEL_ID,
        "bm25_configuration": {
            "method": bm25.method, "k1": bm25.k1, "b": bm25.b, "delta": bm25.delta,
            "normalization": "app.rag.retrieval.match_norm",
            "tokenizer": "bm25s.tokenize", "stopwords": None, "stemmer": None,
            "title_repetitions": 2,
            "note": "title duplicated x2 + full text, normalized for Vietnamese lexical retrieval",
        },
        "build_duration_seconds": round(time.monotonic() - started, 3),
    }
    stage = out / "bm25s_index.partial"
    stage.mkdir()
    log("[bm25] phase=saving")
    bm25.save(str(stage), corpus=None, show_progress=False)
    (stage / "phase9_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(stage, out / "bm25s_index")
    del bm25
    return manifest


def write_index_manifest(corpus: Path, out: Path, scan: dict[str, Any]) -> dict[str, Any]:
    assert_dense_manifests_match(out)
    components = {}
    for name, path in (("faiss", out / "faiss" / "manifest.json"),
                       ("qdrant", out / "qdrant" / "manifest.json"),
                       ("bm25", out / "bm25s_index" / "phase9_manifest.json")):
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("count") != scan["chunk_count"] or
                    saved.get("corpus_sha256") != scan["corpus_sha256"] or
                    saved.get("ordered_chunk_id_sha256") != scan["ordered_chunk_id_sha256"]):
                raise RuntimeError(f"Index component differs from corpus: {path}")
            components[name] = saved
    dense = components.get("faiss", {})
    if not dense:
        dense = components.get("qdrant", {})
    sparse = components.get("bm25", {})
    manifest = {
        **common_manifest(corpus, scan), "components_present": sorted(components),
        "component_status": {name: name in components for name in ("faiss", "qdrant", "bm25")},
        **{key: dense.get(key) for key in (
            "embedding_model_requested_revision", "embedding_model_resolved_revision",
            "embedding_dimension", "normalize_embeddings", "passage_prefix",
            "query_prefix", "passage_template",
            "embedding_batch_size", "device", "faiss_index_type",
            "embedding_stream_sha256", "shared_embedding_stream")},
        "embedding_model_id": dense.get("embedding_model_id", MODEL_ID),
        "qdrant_collection": components.get("qdrant", {}).get("collection_name"),
        "qdrant_hnsw_configuration": components.get("qdrant", {}).get("hnsw_configuration"),
        "bm25_configuration": sparse.get("bm25_configuration"),
    }
    temp = out / "index_manifest.json.partial"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, out / "index_manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight or build ordered E5/FAISS, E5/Qdrant and BM25S indexes.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=MODEL_ID)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--embedding-dimension", type=int, default=768,
                        help="Preflight estimate only; build uses actual model dimension.")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--component", choices=("all", "all-backends", "dense-all", "faiss", "qdrant", "bm25"), default="all",
                        help="all retains the FAISS+BM25S baseline; all-backends adds Qdrant")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", "vn_history_v1_e5"))
    parser.add_argument("--preflight", action="store_true",
                        help="Read and validate corpus without writing outputs or loading a model.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.embedding_model != MODEL_ID:
        raise ValueError(f"Baseline indexing requires {MODEL_ID}")
    if args.embedding_batch_size < 1 or args.embedding_dimension < 1:
        raise ValueError("Embedding batch size and dimension must be positive")
    corpus = args.corpus.expanduser().resolve()
    out = safe_output(args.output_dir)
    device = selected_device(args.device)
    log(f"[index] scanning corpus={corpus}")
    scan = scan_corpus(corpus)
    report = preflight_report(corpus, out, scan, dimension=args.embedding_dimension,
                              batch_size=args.embedding_batch_size, device=device,
                              requested_revision=args.model_revision)
    report["qdrant_collection"] = args.qdrant_collection
    report["qdrant_configured"] = bool(args.qdrant_url)
    log(f"[index] chunks={scan['chunk_count']} corpus_sha256={scan['corpus_sha256']}")
    log(f"[index] embedding_dim_estimate={args.embedding_dimension} estimated_flat_index={report['estimated_faiss_vector_gib']} GiB")
    if args.preflight:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    check_existing(out, scan)
    requested_components = ({"faiss", "qdrant", "bm25"} if args.component == "all-backends" else
                            {"faiss", "bm25"} if args.component == "all" else
                            {"faiss", "qdrant"} if args.component == "dense-all" else
                            {args.component})
    for component, path in (("faiss", out / "faiss"), ("qdrant", out / "qdrant"),
                            ("bm25", out / "bm25s_index")):
        if component in requested_components and path.exists():
            raise FileExistsError(f"Selected index component already exists: {path}")
    out.mkdir(parents=True, exist_ok=True)
    if {"faiss", "qdrant"} <= requested_components:
        revision = resolve_model_revision(MODEL_ID, args.model_revision)
        build_dense_pair(corpus, out, scan, requested_revision=args.model_revision,
                         resolved_revision=revision, device=device,
                         batch_size=args.embedding_batch_size,
                         url=args.qdrant_url, collection=args.qdrant_collection)
        gc.collect()
    elif "faiss" in requested_components:
        revision = resolve_model_revision(MODEL_ID, args.model_revision)
        build_faiss(corpus, out, scan, requested_revision=args.model_revision,
                    resolved_revision=revision, device=device,
                    batch_size=args.embedding_batch_size)
        gc.collect()
    elif "qdrant" in requested_components:
        revision = resolve_model_revision(MODEL_ID, args.model_revision)
        build_qdrant(corpus, out, scan, requested_revision=args.model_revision,
                     resolved_revision=revision, device=device,
                     batch_size=args.embedding_batch_size,
                     url=args.qdrant_url, collection=args.qdrant_collection)
        gc.collect()
    if "bm25" in requested_components:
        build_bm25(corpus, out, scan, batch_size=args.embedding_batch_size)
    manifest = write_index_manifest(corpus, out, scan)
    log(f"[index] completed components={','.join(manifest['components_present'])} output={out}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
