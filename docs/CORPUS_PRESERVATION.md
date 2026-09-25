# Historical corpus preservation

The refactor took a read-only snapshot **before** changing corpus or retrieval
utilities. The snapshot is [`corpus_preservation_manifest.json`](corpus_preservation_manifest.json).
It records relative filenames, byte sizes, SHA-256 hashes, Git status, and
physical line counts for text files. The manifest includes ignored local files,
so a fresh Git clone alone will not contain every listed artifact.

| Scope | Files | Bytes |
| --- | ---: | ---: |
| Full local inventory | 264 | 3,210,595,277 |
| Protected corpus, data, and indexes | 184 | 1,491,752,505 |

The historical locations remain in place: `Dataset/`, `training/Dataset/`,
`datasets/`, and the corpus and retrieval subdirectories of both
`artifacts/vn_history_deployment/` and `artifacts/vn_history_modal/`.
`training/Dataset/` is a legacy **data** path, even though its parent is named
`training`. The two deployed enriched corpus files are byte-identical and each
has 58,603 JSONL lines. Each deployment's FAISS and BM25S sidecar declares
58,603 indexed records. The separate `training/Dataset/merged_jsonl/all_chunk_id.jsonl`
has 520 lines; it is not a replacement for the deployed corpus.

The manifest also inventories model files, mutable `data/chat.sqlite3`,
evaluation fixtures, and local outputs to make cleanup reviewable. Those entries are marked
`inventory_only`; the default audit checks the `protected` data and indexes.

Run the read-only preservation check from the repository root:

```bash
python -m scripts.corpus.audit_corpus
```

It prints a JSON report and exits with status 1 if a protected file is missing,
changed, or unreadable. `--all-files` checks every inventoried file, including
mutable and obsolete model files. `--strict-new` additionally fails when a new
file appears under an inventoried root. The audit does not write to any data
path or rebuild an index.

To inspect corpus quality, run the same command in descriptive mode:

```bash
python -m scripts.corpus.audit_corpus \
  --corpus artifacts/vn_history_deployment/corpus/vn_history_rag_chunks_enriched.jsonl \
  --output reports/corpus/audit.json
```

The command produced a 2,124-byte local report at `reports/corpus/audit.json`;
`reports/` is Git-ignored, so regenerate it after cloning.
It records chunk and unique ID counts, duplicate ID and exact text rates,
near-empty and missing fields, malformed rows, source distribution, year/date
coverage, and character and whitespace-token length summaries. A near-empty
chunk has fewer than 40 non-whitespace characters by default; change that
criterion with `--near-empty-chars`. Whitespace tokens are only a length
estimate, not Qwen tokenizer tokens. The report counts a source ID as inferred
when `metadata.source_sha1` or `hf_dataset` plus `raw_record_index` exists.
For this snapshot, all 58,603 rows parse as JSON objects, and chunk IDs and
non-empty text are unique. The generated report's SHA-256, size, and line
count match the preservation manifest.

The old corpus construction, enrichment, and index-building algorithms are
available independently of the training package as
`scripts.corpus.build_corpus`, `scripts.corpus.enrich_corpus`, and
`scripts.retrieval.build_index`. They are manual offline tools. No corpus or
index was regenerated during this refactor; any future experiment should use
explicit new output paths and compare against this snapshot.
