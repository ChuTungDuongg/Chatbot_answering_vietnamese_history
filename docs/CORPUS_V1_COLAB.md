# Corpus V1: UVW-2026 Colab and Drive workflow

The selected source is [undertheseanlp/UVW-2026](https://huggingface.co/datasets/undertheseanlp/UVW-2026). Its card identifies the license as `cc-by-sa-4.0` and describes the source as Vietnamese Wikipedia with Wikidata metadata. Keep the dataset ID, license, card URL, and original article URLs with any later corpus use or attribution. The card supplies the precise terms; this page does not interpret them. At inspection time on 2026-09-27, `main` resolved to `a0a79294e4568137e25828bb3f2a4cde8546e1fb`. The CLI resolves the requested ref again for each new run and loads by the resulting immutable SHA. To reproduce this inspected revision, pass `--dataset-revision a0a79294e4568137e25828bb3f2a4cde8546e1fb` to inspect and both build commands.

The `default` config has `train` (894,579), `validation` (111,822), and `test` (111,823) source rows according to card metadata. These are source partitions, so the preset selects all three for retrieval. It maps `id`, `title`, `content`, `num_chars`, `num_sentences`, `quality_score`, `wikidata_id`, and `main_category`. `id` is a URL-safe article title; the pipeline derives `https://vi.wikipedia.org/wiki/<encoded id>` when no URL field exists. Real sampled `content` contains plain text and some residual infobox markup. No verified section heading field exists, so `section` stays empty unless a heading syntax is actually present in the text.

## Prepare and inspect

Run in the repository root. The Colab notebook created later should invoke these commands rather than duplicate pipeline logic.

```bash
pip install -r requirements-corpus.txt
python -m scripts.colab.bootstrap --mount-drive --drive-root /content/drive/MyDrive/vn_history_llm --scratch-dir /content/corpus_v1_scratch
python -m scripts.corpus_v1.cli inspect-source --preset uvw-2026 --examples 3
```

The bootstrap reports Colab status, Python and package versions, root, cache and scratch locations, free disk, and available RAM when observable. Corpus construction needs no GPU. `inspect-source` samples only a few streaming rows from each split, truncates article previews, shows field mapping and expected counts, and fails if the ref cannot be pinned. Inspect the resolved SHA and card metadata before building.

## Progress logging

Build progress is enabled by default and written as flushed lines to **stderr**; the final JSON remains on stdout. The builder logs startup settings, each split and shard, validated resume checkpoints, overall progress, finalization, and completion. The default shard size is 10,000 source rows, so there is one start/completion pair per shard and no per-document spam. `--quiet` suppresses normal progress without changing the corpus. ETA uses source rows completed during the current invocation divided by elapsed time; it is approximate and becomes `unknown` when the total or rate is unavailable. Pilot percentages use the per-split row limit; full percentages use Hub expected split sizes. Resume counts already completed rows in the numerator, while ETA uses the rate measured after resuming.

This excerpt came from the tiny offline UVW-shaped fixture; its all-`a` revision is a test SHA, not the live dataset revision:

```text
[Corpus V1] dataset=undertheseanlp/UVW-2026 config=default requested_revision=main resolved_revision_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
[Corpus V1] scope=pilot splits=train,validation,test expected_split_sizes=train:2,validation:2,test:2 limit_per_split=1
[train] shard=1 starting rows=0-0 expected_rows=1
[train] shard=1 completed processed=1 keep=1 review=0 drop=0 documents=1 chunks=1 duplicate_documents=0 duplicate_chunks=0 elapsed=00:00:00 rate=9.2 rows/s
[overall] progress=1/3 percent=33.3% elapsed=00:00:00 rate=5.3 rows/s eta=00:00:01
[train] pilot_limit_reached observed=1 expected=2 source_complete=false
[finalize] aggregating documents starting
[finalize] aggregating documents completed
[done] Corpus V1 build complete scope=pilot source_rows=3 documents=1 chunks=1 keep=2 review=0 drop=1 duplicate_documents=1 duplicate_chunks=0 ...
```

For Colab subprocess calls, invoke `python -u -m scripts.corpus_v1.cli ...` as an extra buffering safeguard. For example, `subprocess.run([sys.executable, "-u", "-m", "scripts.corpus_v1.cli", "build", ...], check=True)` inherits live stderr/stdout. `capture_output=True` holds both streams until the subprocess finishes; use an inherited stream or iterate over a `Popen` pipe when displaying progress in a notebook. On `--resume`, logs show completed shard validation, rows skipped, any uncommitted files cleared, dedup reconstruction, and the row where each split continues.

## Pilot, review, and audit

The real first-N pilot processed 10,000 rows from each split (30,000 articles). The previous filter marked 11,407 KEEP, 17,682 REVIEW, and 911 DROP, retaining about 96.96%. A year or period appeared in about 24,586 articles and was far too common to establish relevance. A single historical word in the body could also promote unrelated subjects. These are observations from **pilot-001**, not measured outcomes for the revised filter. The first-N selection may be order-biased.

The revised filter is `broad_history_category_v3`. Its year/period signal is diagnostic and can only add support when independent historical or contextual evidence exists; year extraction in document/chunk metadata is unchanged. Body text needs multiple distinct historical cues for a strong `historical_text` signal. Historical title/category evidence remains strong. Unicode NFC, case folding, normalized category whitespace, and Vietnamese/English keyword families recognize related categories. Wikimedia maintenance and disambiguation pages drop; historically meaningful list pages go to REVIEW. Generic commune/municipality stubs with only location or population facts drop, while cities, countries, historical places, and people retain paths to KEEP or REVIEW. Technical, taxonomy, routine sports, software/product, and fictional topics are handled more cautiously. `quality_score` remains diagnostic only.

**Do not resume `uvw-2026-pilot-001` with this filter.** The filter version is part of the build fingerprint, and resume rejects an older saved version. Use a fresh `uvw-2026-pilot-002` output directory and the pinned revision below. `--resume` can continue a pilot-002 run made with the same filter and settings.

```bash
python -m scripts.corpus_v1.cli build --preset uvw-2026 --dataset-revision a0a79294e4568137e25828bb3f2a4cde8546e1fb --split all --max-records-per-split 10000 --shard-size 10000 --cache-dir /content/drive/MyDrive/vn_history_llm/cache --scratch-dir /content/corpus_v1_scratch --output /content/drive/MyDrive/vn_history_llm/corpus_v1/uvw-2026-pilot-002 --resume
python -m scripts.corpus_v1.cli sample --corpus /content/drive/MyDrive/vn_history_llm/corpus_v1/uvw-2026-pilot-002 --decision review
python -m scripts.corpus_v1.cli sample --corpus /content/drive/MyDrive/vn_history_llm/corpus_v1/uvw-2026-pilot-002 --decision dropped
python -m scripts.corpus_v1.cli audit --corpus /content/drive/MyDrive/vn_history_llm/corpus_v1/uvw-2026-pilot-002
```

Review `filter_audit.json`, `stats.json`, and `samples/`. The pilot reports rows per split; KEEP/REVIEW/DROP counts and rates; top categories by decision; reason combinations; date-only, body-history-only, weak-signal REVIEW, generic-municipality-stub DROP, and Wikimedia-maintenance DROP counts; quality score distribution; Wikidata coverage; text sizes; duplicate counts; sampled examples; output size; and processing throughput. A pilot has `build_scope: pilot`, its per-split limit, and `source_complete: false` even if a tiny fixture happens to contain fewer rows. **A pilot is not an accepted production corpus.** It takes the first N rows of each split in stable source order; that selection can be biased and is not a random or representative sample. Sample contents should guide filter review before a full run.

The filter uses title, article text, and `main_category`. KEEP requires strong historical evidence. REVIEW requires plausible historical utility, especially for people, significant places, heritage, political institutions, religion, and culture. DROP covers clearly unrelated topics and low-value maintenance or stubs. Missing/unknown categories receive decisions from the available title and text evidence; they are not automatically dropped or reviewed. The score is a deterministic heuristic, not a calibrated probability. `quality_score` is preserved for diagnostics only; it does not determine inclusion. `wikidata_id` is recorded locally without per-article API calls. Inspect false positives and false negatives in pilot-002 before deciding on a full build.

## Full build and resume

Only after reviewing the pilot, use a **different** run directory without a limit:

```bash
python -m scripts.corpus_v1.cli build --preset uvw-2026 --split all --shard-size 10000 --cache-dir /content/drive/MyDrive/vn_history_llm/cache --scratch-dir /content/corpus_v1_scratch --output /content/drive/MyDrive/vn_history_llm/corpus_v1/uvw-2026-full-v1 --resume
```

After a Colab disconnect, rerun that exact command. The default shard size is 10,000 source rows. Each completed shard has per-split source row range, count, SHA-256 for its record/document/chunk files, and a configuration fingerprint. Shard data are built under optional local scratch, copied to the Drive run directory using `.partial` names, checked, atomically renamed, and committed by writing the manifest last. Only manifested, hash-valid shards are reused. Uncommitted `.partial` or orphan files are cleared on resume, so at most the current shard is recomputed. A corrupted completed shard or configuration mismatch fails clearly; use a trusted copy or a new run directory. Resume reconstructs global exact-dedup state from completed shards before continuing. Streaming `.skip()` is used when supported; otherwise earlier rows are traversed sequentially, which can cost time on late resumes.

```bash
python -m scripts.corpus_v1.cli audit --corpus /content/drive/MyDrive/vn_history_llm/corpus_v1/uvw-2026-full-v1
```

The full build compares observed source rows to Hub metadata counts when available and refuses to write a complete manifest on mismatch. `source_manifest.json` and `manifest.json` record requested and resolved revisions, config, split counts, card/license provenance, field mapping, software and pipeline versions, build scope, and output hashes. The resolved repository SHA is the primary remote source identity; any local Hugging Face fingerprint is auxiliary. A full build is still a candidate for human review and evaluation, not an automatic replacement for V0.

## Output and capacity

Schema version 2 adds first-class source identity. `documents.jsonl` rows include `schema_version`, `document_id`, `source_id`, `source_type`, `source_dataset_id`, `source_article_id`, `source_split`, `source_revision_sha`, `title`, `url`, `language`, normalized `text`, `raw_record_index`, `raw_sha256`, `source_metadata`, `historical_relevance_score`, `historical_filter_decision`, and `historical_filter_reasons`. Interpret `raw_record_index` together with `source_split`. `chunks.jsonl` rows include `chunk_id`, `document_id`, those source identity fields, `title`, `section`, `url`, `text`, `language`, `chunk_index`, `token_count`, `char_count`, `years`, score, and metadata containing the filter decision and source metadata. Schema 1 readers must be checked before using schema 2. Source IDs hash dataset ID, resolved SHA, split, and article ID; document IDs derive from source ID; chunk IDs derive from document ID, order, section, and normalized text hash. Exact document and chunk deduplication is global across all selected splits with deterministic first-wins order.

The configured `intfloat/multilingual-e5-base` tokenizer and its resolved model revision enforce the 384-token default chunk limit; the default overlap is 48 tokens. Chunking prefers preserved section, paragraph, and sentence boundaries, then splits long units without semantic rewriting. Unicode NFC normalization retains Vietnamese diacritics. The V1 run stays separate from the frozen V0 artifacts.

Plan for several hundred gigabytes of possible source/cache/intermediate/final/index activity until the pilot provides measured sizes; Drive copies and remote streaming may dominate elapsed time. Local scratch needs space for one shard and the dedup SQLite database; Drive holds durable checkpoints plus final outputs. Final aggregation temporarily needs another copy of each final JSONL. The existing `scripts/retrieval/build_index.py` reads all chunk rows and texts into RAM and then creates embeddings and BM25 tokens, so indexing a full corpus may require substantially more RAM than corpus construction. It accepts the schema 2 `title`, `text`, and `chunk_id` fields and retains ordered chunk ID hashing and V0 write protection. Its memory capacity and the full run's real throughput must be checked in the later Colab workflow; this task does not build indexes or the full corpus.
