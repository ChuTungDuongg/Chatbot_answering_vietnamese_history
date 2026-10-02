# Retrieval artifacts

The current runtime defaults to `ARTIFACT_ROOT=artifacts/corpus_v1`, resolved from the repository root. Modal reads `/artifacts/corpus_v1/chunks.jsonl` from `vn-history-artifacts`. See [the current restoration report](../RESTORE_MODAL_QDRANT_REPORT.md) and [quick start](../MODAL_QUICKSTART.md). Git does not contain large deployment files; startup never builds or rewrites them.

V1 contains `chunks.jsonl`, `documents.jsonl`, `retrieval/{faiss,bm25s_index,qdrant}`, and `runtime/{inference_config.json,manifest.json}`. The 2026-09-30 release has finalized FAISS/BM25S indexes for 624,288 chunks. Qdrant metadata refers to the separate remote collection `vn_history_v1_e5`.

V0 remains available locally at `artifacts/vn_history_deployment` through an explicit `ARTIFACT_ROOT`. The following tree describes that preserved legacy bundle:

```text
artifacts/vn_history_deployment/
├── corpus/vn_history_rag_chunks_enriched.jsonl
├── retrieval/
│   ├── faiss/chunks.index
│   ├── faiss/manifest.json
│   └── bm25s_index/
│       └── phase9_manifest.json
├── config/inference_config.json
└── manifest.json
```

The FAISS and BM25S directories include additional index data files in the preserved artifact copy. At load time, the application checks required paths, corpus row count and unique chunk IDs, FAISS vector count, and BM25S manifest count. The existing retrieval configuration supplies the embedding and reranker model IDs.

| `APP_MODE` | Required artifacts |
|---|---|
| `api-only` | None. |
| `retrieval-only` | Corpus, retrieval indexes, and retrieval configuration. |
| `full` | The same retrieval files plus the Qwen model files/cache and, for Hybrid SFT, a PEFT adapter. |

Hybrid uses `Qwen/Qwen3-4B-Instruct-2507` with `MODEL_VARIANT=vanilla` or `MODEL_VARIANT=sft`; SFT loads the base tokenizer and attaches the adapter configured by `MODEL_ADAPTER_PATH`. Central uses vanilla `Qwen/Qwen3-8B`. Qwen weights may live in an external Hugging Face cache. The current PEFT adapter's `adapter_config.json` is tracked under `artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter`; its `adapter_model.safetensors` must be supplied locally or in the mounted Modal artifact Volume and is ignored by Git. Older role adapter directories in preserved artifact copies are historical.

[`../docs/corpus_preservation_manifest.json`](../docs/corpus_preservation_manifest.json) records the original file inventory and hashes, including data held at legacy paths. Run `python -m scripts.corpus.audit_corpus` to verify unchanged bytes or pass `--corpus` and `--output` for a descriptive read-only audit. See [`../docs/CORPUS_PRESERVATION.md`](../docs/CORPUS_PRESERVATION.md).

New corpus or index outputs belong in separate paths and require a separately recorded experiment. Do not replace the baseline artifact tree as a side effect of application startup, testing, or auditing.

## UVW-2026 Corpus V1 full candidate

The following build statistics are historical provenance. The restored release now includes completed retrieval indexes and runtime metadata. The supplied full-build report records 1.118M source Wikipedia
records scanned, 145,648 retained documents, and 624,288 RAG chunks. Chunking
used a hard maximum of 384 tokens and 48 configured overlap tokens; chunks
averaged about 289 tokens (median 315). The build took about 58 minutes.
The final chunks JSONL is about 1.33 GB decimal, and the documents JSONL
about 0.93 GB decimal. Exact values follow.

A **document** is a retained Wikipedia article. A **chunk** is a retrieval unit
made from a document:

```text
1,118,224 source rows → historical filtering → 145,648 documents
                      → chunking → 624,288 retrieval chunks
```

The source is `undertheseanlp/UVW-2026` at immutable dataset revision
`a0a79294e4568137e25828bb3f2a4cde8546e1fb`. Its source partitions
are train (894,579), validation (111,822), and test (111,823), for
**1,118,224 source rows**. All three partitions are included in this RAG
corpus. Those labels describe source provenance; they are **not** this
project's model-training or evaluation splits. Corpus V1 is intended for
retrieval-augmented generation, not model training.

The build used history filter `broad_history_category_v3` and
`intfloat/multilingual-e5-base` tokenization. Chunking was deterministic,
preferring section, paragraph, and sentence boundaries with a token-bounded
fallback; its maximum was **384 tokens**, with **48 tokens** of configured
overlap.

| Full-build source statistic | Exact value |
| --- | ---: |
| Source count | 1,118,224 |
| Train / validation / test source rows | 894,579 / 111,822 / 111,823 |
| Source rows per second | 319.82 |
| Source characters: count / min / max | 1,118,224 / 89 / 910,455 |
| Source characters: mean / median / p90 / p95 | 1,189.90 / 197 / 2,611 / 4,799 |
| Missing `main_category` | 33,829 |
| Missing `wikidata_id` | 6,662 |

| Full-build document statistic | Exact value |
| --- | ---: |
| Retained document count | 145,648 |
| Documents per second | 41.66 |
| Document characters: count / min / max | 145,648 / 100 / 252,015 |
| Document characters: mean / median / p90 / p95 | 4,347.19 / 2,025 / 9,737 / 16,104 |
| Duplicate document count | 7 |

| Full-build chunk statistic | Exact value |
| --- | ---: |
| Chunk count | 624,288 |
| Chunks per second | 178.55 |
| Chunk characters: count / min / max | 624,288 / 1 / 2,271 |
| Chunk characters: mean / median / p90 / p95 | 1,030.39 / 1,096 / 1,401 / 1,452 |
| Chunk tokens: count / min / max | 624,288 / 1 / 384 |
| Chunk tokens: mean / median / p90 / p95 | 289.32 / 315 / 378 / 382 |
| Duplicate chunk count | 1,362 |
| Chunks with year or period metadata | 511,784 |

| Full-build storage and duration | Exact value |
| --- | ---: |
| Final `chunks.jsonl` bytes | 1,331,391,190 |
| Final `documents.jsonl` bytes | 926,690,133 |
| Shard bytes | 2,923,891,072 |
| Approximate generated bytes | 5,181,972,395 |
| Elapsed seconds | 3,496.47 |

`documents.jsonl` preserves the full retained articles and source provenance.
`chunks.jsonl` is the primary input for retrieval. V1 is now the runtime default;
V0 is preserved for explicit reproduction and comparison. These build statistics
do not establish that V1 is better.

The intended local layout is:

```text
artifacts/
├── corpus_v1/
│   ├── documents.jsonl
│   ├── chunks.jsonl
│   └── [small provenance files, if copied from the build]
└── old_corpus/
    └── [local archive of previous artifacts]
```

The historical inspection originally found only two JSONL files; the release restore
also supplies existing retrieval and runtime manifests. `artifacts/corpus_v1/` is generated local data and
must not be committed. `artifacts/old_corpus/` is the user's local archive,
not a new runtime source of truth, and must not be committed.

The historical preservation manifest still targets original V0 paths; its broader
inventory includes files deliberately excluded from runtime backups. Runtime V0 is
restored locally, and runtime V1 is the default. The [V1 indexing guide](../docs/CORPUS_V1_INDEXING.md)
describes earlier build preparation; use the restoration report for current serving.
