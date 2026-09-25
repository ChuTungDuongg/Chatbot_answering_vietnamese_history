# Hybrid retrieval baseline

The `hybrid` mode retrieves history evidence from the preserved corpus, builds a cited prompt, and streams a grounded answer from vanilla `Qwen/Qwen3-4B-Instruct-2507`.

```text
question
  → deterministic query and domain handling
  → multilingual E5 / FAISS dense search + BM25S sparse search
  → weighted reciprocal-rank fusion
  → cross-encoder reranking
  → metadata signals, deduplication, and context diversity
  → prompt with selected chunks
  → Qwen3-4B answer stream with citations
```

`retriever.py` defines the interface used by both inference modes. `retrieval.py` contains the existing FAISS + BM25S implementation. `hybrid_runtime.py` joins retrieval, prompt construction, and model generation. The model runtime lives in `app/models/` and does not load or query indexes itself.

Corpus and index files are read from the configured artifact root. No corpus cleanup or index rebuild happens at startup. The manifest in [`../../docs/corpus_preservation_manifest.json`](../../docs/corpus_preservation_manifest.json) records the historical artifact bytes; `python -m scripts.corpus.audit_corpus` checks preservation without writing to them.

This retrieval boundary allows a later Qdrant implementation to be measured against the current baseline. Qdrant is not part of the baseline.
