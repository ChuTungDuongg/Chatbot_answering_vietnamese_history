# Corpus and retrieval utilities

Application startup reads existing corpus and index artifacts. It does not rebuild, normalize, or delete historical data. The source snapshot is recorded in [`../docs/corpus_preservation_manifest.json`](../docs/corpus_preservation_manifest.json).

## Read-only checks

Verify the recorded filenames, sizes, and SHA-256 hashes without writing to the data tree:

```powershell
python -m scripts.corpus.audit_corpus
```

Write a descriptive corpus quality report to a separate output path:

```powershell
python -m scripts.corpus.audit_corpus --corpus artifacts/vn_history_deployment/corpus/vn_history_rag_chunks_enriched.jsonl --output reports/corpus/audit.json
```

The report summarizes chunk IDs, duplicate text, empty chunks, metadata coverage, and length distributions. It does not clean or rewrite the corpus. See [`../docs/CORPUS_PRESERVATION.md`](../docs/CORPUS_PRESERVATION.md).

## Explicit data-producing commands

`scripts/corpus/build_corpus.py` and `scripts/corpus/enrich_corpus.py` create new corpus outputs. `scripts/retrieval/build_index.py` preflights a chosen corpus and can build FAISS and BM25S separately or together. Use explicit input and output paths; these commands are not part of baseline startup or audit.

```powershell
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --preflight
```

The baseline uses the preserved files and retrieval behavior already recorded in the manifest. A proposed corpus or index change is a new experiment and must use its own artifact hash, dataset, and benchmark run.

For the new Wikipedia Corpus V1, use `python -m scripts.corpus_v1.cli` and the [Colab workflow](../docs/CORPUS_V1_COLAB.md). It streams documents through checkpoints and never imports the FastAPI runtime. Its output must be a separate `corpus_v1/` root.
See [V1 indexing readiness](../docs/CORPUS_V1_INDEXING.md) for resource behavior, component builds, and the runtime-loading gap.
