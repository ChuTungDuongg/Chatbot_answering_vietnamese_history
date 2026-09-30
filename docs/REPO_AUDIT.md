# Repository audit after V2 cleanup

| Classification | Finding | Action |
| --- | --- | --- |
| REMOVE NOW | Four tracked role SFT datasets, the 1,000-message export, old role reports and smoke output, orphan trajectory fixture | Removed; see [cleanup](LEGACY_DATA_CLEANUP.md). |
| KEEP | `app/` two runtime modes, `scripts/retrieval/build_index.py`, evaluation fixture/schema, current Modal deployment entrypoint | Still used by the V2 baseline or its tools. |
| LEGACY BUT PROTECTED | `training/Dataset/`, `training/InvestigatingDataset.zip`, ignored historical source packs, protected `artifacts/training/**/*.jsonl`, V0 corpus and indexes | Preserve bytes until separate data provenance review. |
| FUTURE CLEANUP | V0 inference config contains old `role_models` and vLLM fields, although runtime uses `app/config.py` model IDs; `scripts/corpus/build_corpus.py` and `enrich_corpus.py` reflect old pack construction | Keep frozen V0 metadata and manual legacy utilities; document instead of mutating V0. |
| FUTURE CLEANUP | Multiple README files and confusing `Dataset/`, `training/Dataset/`, `evaluation/datasets/` names | Keep local guides, clarify meanings in root README. |
| FUTURE CLEANUP | Current retrieval domain anchors and query patterns emphasize Vietnamese history | Test world-history recall with reviewed questions before considering V1 for serving; do not tune V0 thresholds in this cleanup. |
| FUTURE EXPERIMENT | vLLM serving and Qdrant retrieval | Evaluate V1 against V0 with human labels first; experiment separately later. |

The active app contains no Research Agent, Evidence Agent, History Answerer, role switching, or hidden model fallback. Old strings in the preservation snapshot are historical inventory paths. The active chat modes are `hybrid` and `central`; Hybrid additionally selects vanilla or SFT PEFT at startup through `MODEL_VARIANT`.

The retriever uses E5 query/passage prefixes, FAISS dense search, normalized BM25S, weighted RRF, cross-encoder reranking, metadata bonuses, deduplication, and context diversity. The index row order is a correctness boundary: the new builder records an ordered chunk-ID SHA-256 in both sidecars, and runtime validates it when present. Old V0 sidecars contain only a corpus signature and count, so V0 row alignment cannot be proven from those sidecars alone. V0 bytes and ranking weights remain unchanged.

Central's `search_history` tool receives the same `HybridRetriever` object instantiated for Hybrid. Planning has a configured bounded number of rounds and up to four calls per round; tool and parse failures terminate or proceed within those bounds. Only final generation goes to SSE `answer_delta`. Hybrid makes one generation call after retrieval. Qwen loads explicit model IDs and optional revisions; Hybrid SFT attaches its configured PEFT adapter at startup and never falls back to vanilla. `TextIteratorStreamer` emits genuine generated text from a worker thread.

## Corpus V1 source hardening

`configs/corpus_v1/uvw_2026.json` selects the `undertheseanlp/UVW-2026` default config and all source splits. The builder resolves a requested Hugging Face revision to a commit SHA and loads by that SHA. Schema 2 records split and immutable source identity on documents and chunks. Per-split shards provide manifest-last durable checkpoints and global exact deduplication is reconstructed on resume. Pilots are explicitly partial; full builds check observed counts against source metadata. See [Corpus V1 Colab workflow](CORPUS_V1_COLAB.md) for commands and capacity limits. No full V1 corpus or V1 retrieval index is present in this repository.
