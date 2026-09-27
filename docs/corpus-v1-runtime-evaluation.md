# Corpus V1 retrieval runtime and evaluation

Corpus V1 lives at `artifacts/corpus_v1`. The application can read its 1.33 GB
`chunks.jsonl` in place. Keep all corpus, index, runtime output, reports, and
credentials out of Git.

## Prepare small runtime metadata

Run after the corpus, FAISS, BM25S, and index manifests have been placed locally:

```powershell
python -m scripts.retrieval.prepare_v1_runtime
```

This streams the existing corpus to verify its count, byte hash, and ordered
chunk IDs. It compares those values with the FAISS, BM25S, and index manifests,
then writes only `artifacts/corpus_v1/runtime/manifest.json` and
`inference_config.json`. It never builds an index or copies the corpus. A
`qdrant.partial` directory is ignored and left untouched. The Qdrant backend is
listed only when `retrieval/qdrant/manifest.json` exists and validates. The
runtime uses the pinned E5 revision recorded by the index and pins
`BAAI/bge-reranker-v2-m3` to a revision in the generated configuration.

Set these environment variables for the V1 application or evaluator:

```powershell
$env:APP_MODE='retrieval-only'
$env:CORPUS_PATH='./artifacts/corpus_v1/chunks.jsonl'
$env:RETRIEVAL_ROOT='./artifacts/corpus_v1/retrieval'
$env:RETRIEVAL_DENSE_BACKEND='faiss'
```

With these two path overrides, the application finds the generated files under
`artifacts/corpus_v1/runtime`. `INFERENCE_CONFIG_PATH` and
`RUNTIME_MANIFEST_PATH` can override those defaults independently. The V0
bundle layout remains the default when the path overrides are unset.
The V1 runtime verifies corpus count, byte hash, ordered chunk ID hash, and
index manifest fingerprints at startup. FAISS checks `ntotal`; BM25S checks
`num_docs`. The application holds parsed chunks in memory for retrieval. A
future compact offset sidecar could reduce startup RAM; this task does not
change corpus or metadata storage.

For Qdrant, set `RETRIEVAL_DENSE_BACKEND=qdrant` and supply `QDRANT_URL`,
`QDRANT_API_KEY`, and `QDRANT_COLLECTION` through local environment variables.
Use `QDRANT_HNSW_EF` when testing a specific search setting. The finalized
manifest must be present. Startup checks remote exact point count, collection,
vector dimension and distance, quantization, and HNSW configuration. A failed
Qdrant check stops startup; it never switches to FAISS. Never put a real key in
a command, document, log, test, or committed file.

## Headless retrieval quality

Prepare a labeled JSONL dataset using `evaluation/datasets/question.schema.json`.
Missing gold chunk or source IDs remain unscored. Run one pass per question:

```powershell
python -m evaluation.run_retrieval --dataset evaluation/datasets/fixtures/questions.jsonl --output reports/retrieval/v1-faiss
```

After the Qdrant collection is finalized, regenerate the small runtime metadata,
switch only `RETRIEVAL_DENSE_BACKEND` to `qdrant`, set the three Qdrant variables
locally, and run:

```powershell
python -m evaluation.run_retrieval --dataset evaluation/datasets/fixtures/questions.jsonl --output reports/retrieval/v1-qdrant
```

For an interrupted run, repeat its command with `--resume`. The runner checks
the dataset SHA, backend, corpus hashes, E5 and reranker revisions, manifest
fingerprints, retrieval settings, top K, and Git commit. It skips completed IDs.
It repairs only an incomplete trailing JSONL line, then appends and flushes each
result to disk. Existing runs are never overwritten; a complete run cannot be
resumed. Runs use concurrency 1 and keep `predictions.jsonl`, `run_metadata.json`,
`summary.json`, and `progress.log` under the output directory.

Each prediction contains `question_id`, `success`, `error`, ranked `sources`,
retrieval diagnostics, and latency. Sources include chunk, source and document
IDs, title, text, stage ranks, dense and BM25 scores, RRF score, reranker score,
and final retrieval score. The summary reports available chunk and source
HitRate, Recall, and Precision at 1/3/5/10, plus MRR and nDCG at 10. Labels
that are absent stay N/A.

The saved predictions can also be scored with the existing deterministic runner:

```powershell
python -m evaluation.runner --dataset evaluation/datasets/fixtures/questions.jsonl --predictions reports/retrieval/v1-faiss/predictions.jsonl --output reports/retrieval/v1-faiss-scored --phase all
```

Use the React UI for manual inspection of about 10–20 representative questions
and citations. Use the headless runner for 300–500 question retrieval quality
comparison. The separate `benchmarks.latency.runner` remains the external
HTTP/SSE path for full application and Qwen runs: use one run per labeled
question for answer quality, and repetitions only for latency variance. Keep
retrieval quality, answer quality, latency, and concurrency/load as distinct
experiments.
