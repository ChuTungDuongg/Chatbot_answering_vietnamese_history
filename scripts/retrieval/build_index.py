"""Build the existing FAISS and BM25S retrieval indexes on explicit request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.corpus.jsonl import read_jsonl


def ordered_chunk_id_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            raise ValueError("Every indexed row requires chunk_id")
        digest.update((str(chunk_id) + "\n").encode("utf-8"))
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 9 build FAISS and BM25S indexes for the enriched corpus.")
    parser.add_argument("--corpus", default="artifacts/corpus/vn_history_rag_chunks_enriched.jsonl")
    parser.add_argument("--output-dir", default="artifacts/retrieval")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import bm25s
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from app.rag.retrieval import match_norm

    out = Path(args.output_dir).resolve()
    if any(part.casefold() in ("vn_history_deployment", "vn_history_modal") for part in out.parts):
        raise ValueError("Refusing to write indexes inside frozen V0 artifacts")
    rows = read_jsonl(args.corpus)
    if not rows:
        raise ValueError("Cannot index an empty corpus")
    ids_sha256 = ordered_chunk_id_sha256(rows)
    texts = [f"passage: {row.get('title', '')}\n{row.get('text', '')}" for row in rows]
    faiss_dir = out / "faiss"
    bm25_dir = out / "bm25s_index"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype="float32"))
    faiss.write_index(index, str(faiss_dir / "chunks.index"))
    (faiss_dir / "manifest.json").write_text(json.dumps({"count": len(rows), "embedding_model_id": args.embedding_model,
                                                         "ordered_chunk_id_sha256": ids_sha256}, indent=2), encoding="utf-8")

    corpus_tokens = bm25s.tokenize([
        match_norm(f"{row.get('title', '')} {row.get('title', '')} {row.get('text', '')}")
        for row in rows], stopwords=None, stemmer=None)
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(bm25_dir), corpus=None)
    (bm25_dir / "phase9_manifest.json").write_text(json.dumps({"count": len(rows),
                                                                 "ordered_chunk_id_sha256": ids_sha256,
                                                                 "note": "title duplicated x2 + full text, normalized for Vietnamese lexical retrieval"}, indent=2), encoding="utf-8")
    print(f"Built FAISS/BM25 indexes for {len(rows)} chunks in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
