# Retrieval artifacts

The baseline reads an existing historical corpus and its retrieval indexes from `ARTIFACT_ROOT` (default `artifacts/vn_history_deployment`). Git does not contain the large deployment files. Restoring them is an explicit setup step; application startup never builds or rewrites them.

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
| `full` | The same retrieval files plus the configured vanilla Qwen model files/cache. |

Hybrid uses `Qwen/Qwen3-4B-Instruct-2507`; Central uses `Qwen/Qwen3-8B`. Qwen weights may live in an external Hugging Face cache rather than this directory. Adapter directories found in an older artifact copy are historical files and are not loaded by either baseline mode.

[`../docs/corpus_preservation_manifest.json`](../docs/corpus_preservation_manifest.json) records the original file inventory and hashes, including data held at legacy paths. Run `python -m scripts.corpus.audit_corpus` to verify unchanged bytes or pass `--corpus` and `--output` for a descriptive read-only audit. See [`../docs/CORPUS_PRESERVATION.md`](../docs/CORPUS_PRESERVATION.md).

New corpus or index outputs belong in separate paths and require a separately recorded experiment. Do not replace the baseline artifact tree as a side effect of application startup, testing, or auditing.
