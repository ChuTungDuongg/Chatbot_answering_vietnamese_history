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

`scripts/corpus/build_corpus.py` and `scripts/corpus/enrich_corpus.py` create new corpus outputs. `scripts/retrieval/build_index.py` creates FAISS and BM25S indexes from a chosen corpus. Use separate output paths when investigating a future rebuild; these commands are not part of the baseline startup or audit.

```powershell
python -m scripts.retrieval.build_index --corpus path/to/curated_corpus.jsonl --output-dir path/to/new_index_directory
```

The baseline uses the preserved files and retrieval behavior already recorded in the manifest. A proposed corpus or index change is a new experiment and must use its own artifact hash, dataset, and benchmark run.
