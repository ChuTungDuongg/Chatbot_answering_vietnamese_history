# Corpus V1: Colab and Google Drive

Corpus V1 is an offline, reproducible Vietnamese Wikipedia build. It does not change the runtime or V0. Pin a verified Hugging Face dataset and revision before a comparison run; the example below uses placeholders because dataset IDs and revisions vary. The source schema is inspected before processing.

```bash
git clone https://github.com/ChuTungDuongg/Chatbot_answering_vietnamese_history.git
cd Chatbot_answering_vietnamese_history
pip install -r requirements-corpus.txt
python -m scripts.colab.bootstrap --mount-drive --drive-root /content/drive/MyDrive/vn_history_llm
python -m scripts.corpus_v1.cli inspect-source --dataset-id <HF_DATASET_ID> --dataset-config <OPTIONAL_CONFIG> --dataset-revision <PINNED_REVISION> --split train
python -m scripts.corpus_v1.cli build --dataset-id <HF_DATASET_ID> --dataset-config <OPTIONAL_CONFIG> --dataset-revision <PINNED_REVISION> --split train --cache-dir /content/drive/MyDrive/vn_history_llm/cache --output /content/drive/MyDrive/vn_history_llm/corpus_v1/run-001 --streaming --resume
```

Omit `--dataset-config` if the source has no config. Override field names with `--title-field`, `--text-field`, `--id-field`, and `--url-field` when auto-detection differs. `inspect-source` reads only a streaming sample and displays fields and mapping. Use `--no-streaming` only for a source that cannot stream. Outside Colab, omit `--mount-drive` and pass a normal local root; the CLI imports `google.colab` only when mounting is requested.

The bootstrap creates `raw/`, `cache/`, `intermediate/`, `corpus_v1/`, `logs/`, and `reports/` under the chosen Drive root. Each V1 run has its own `corpus_v1/run-NNN/` directory. The build writes `config.json`, `source_manifest.json`, `intermediate/` stage JSONL/checkpoints, `documents.jsonl`, `chunks.jsonl`, `stats.json`, `filter_audit.json`, `samples/`, `hashes.json`, and `manifest.json`. The runtime V0 artifacts remain under their original paths.

The builder streams source rows into atomic stage files: normalize → classify → exact document dedup → tokenize/chunk and exact chunk dedup → final copies, statistics, and hashes. Completed stage manifests contain count, SHA-256, and config fingerprint. On disconnect, rerun the exact build command with `--resume`; verified completed stages are reused. If a `.partial` file remains, inspect and remove only that run's incomplete `.partial` file before rerunning. A changed configuration or corrupt completed stage fails and requires a fresh run directory or repair from a trusted copy.

The high-recall filter uses title, text, date/period, person/place, cultural and archaeological signals. `KEEP` has clear historical cues; `REVIEW` is ambiguous and included by default; `DROP` is clearly unrelated or unusably empty. The score is a deterministic heuristic, not a calibrated probability. Inspect its distribution and deterministic samples before using the run:

```bash
python -m scripts.corpus_v1.cli sample --corpus /content/drive/MyDrive/vn_history_llm/corpus_v1/run-001 --decision review
python -m scripts.corpus_v1.cli audit --corpus /content/drive/MyDrive/vn_history_llm/corpus_v1/run-001/chunks.jsonl
```

The baseline chunk budget is 384 multilingual E5 tokenizer tokens with 48 tokens of sentence-unit overlap. These are comparison starting values, not optimized parameters. Unicode NFC normalization retains Vietnamese diacritics. Exact dedup uses normalized document/chunk text SHA-256. IDs derive from dataset identity, pinned revision, split, source ID or row index, and chunk position/text; the same source order and configuration reproduce IDs.

To build independent V1 FAISS/BM25S indexes with the existing E5 baseline, install runtime retrieval dependencies and run:

```bash
python -m scripts.retrieval.build_index --corpus /content/drive/MyDrive/vn_history_llm/corpus_v1/run-001/chunks.jsonl --output-dir /content/drive/MyDrive/vn_history_llm/corpus_v1/run-001/retrieval --embedding-model intfloat/multilingual-e5-base
```

The index builder currently loads the chunk rows into memory, so index building may need more RAM than corpus construction. Keep the V1 index separate and do not point production at it until human-reviewed V0/V1 evaluation. Corpus output paths containing `vn_history_deployment` or `vn_history_modal` are rejected.

## Row schemas

`documents.jsonl` rows have `schema_version` (1), `document_id`, `source_id`, `source_type` (`wikipedia`), `title`, `url`, `language` (`vi`), `text`, `raw_record_index`, `raw_sha256`, `source_metadata`, `historical_relevance_score` (integer heuristic), `historical_filter_decision` (`KEEP`/`REVIEW`), and `historical_filter_reasons` (list of strings). Filtered stage rows can also have `DROP`.

`chunks.jsonl` rows have `schema_version` (1), `chunk_id`, `document_id`, `source_id`, `source_type`, `title`, `section`, `url`, `text`, `language`, `chunk_index`, `token_count`, `char_count`, `years` (date/period matches), `historical_relevance_score`, and `metadata` (`raw_record_index`, `historical_filter_decision`, `source_metadata`). URLs are taken from source or derived from the Wikipedia title when absent.
