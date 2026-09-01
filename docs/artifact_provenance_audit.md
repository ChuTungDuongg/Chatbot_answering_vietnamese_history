# Artifact Provenance Audit

Date: 2026-08-28

## Baseline

- Starting branch: `main`
- Starting HEAD: `e2091b14e97003f53e70fc5e88f2638a6544896b`
- Initial status: clean (`## main...origin/main`)
- Canonical local root rebuilt: `artifacts/vn_history_deployment`
- Deployment ID: `qwen3-3f5f81301b0ca4ac`
- Shared base: `Qwen/Qwen3-4B-Instruct-2507`
- Backend: `transformers`

## Weight Verification

| Role | Expected local SHA256 | Remote SHA256 | Status |
|---|---|---|---|
| research | `0d36e09fb947a6b077ee493f3589a36bf68dba0403f7eac91f684d070d399086` | `0d36e09fb947a6b077ee493f3589a36bf68dba0403f7eac91f684d070d399086` | MATCH |
| evidence | `39385ca7c82b57b5ff8c9a531b5359509ea185b15e5a0adb0724626c58ed7ff6` | `39385ca7c82b57b5ff8c9a531b5359509ea185b15e5a0adb0724626c58ed7ff6` | MATCH |
| history | `70d873e15c48f5802e26d0c32eab7c63ea7b83f713be3192092476a0dac746a3` | `70d873e15c48f5802e26d0c32eab7c63ea7b83f713be3192092476a0dac746a3` | MATCH |

Remote hashes were computed inside a Modal function with `vn-history-artifacts` mounted at `/artifacts`; no adapter weights were downloaded to the workstation.

## Canonical Export

`training/scripts/export_artifacts.py` now exports through a temporary sibling root, validates the temporary bundle, and only then replaces `artifacts/vn_history_deployment`.

The exporter copies only production-required adapter files:

- `adapter_config.json`
- `adapter_model.safetensors`

It excludes training/runtime residue such as:

- `checkpoint-*`
- `optimizer.pt`
- `scheduler.pt`
- `trainer_state.json`
- `training_args.bin`
- Qwen2.5 active-looking legacy paths

Local stale entries removed during rebuild: `138`.

## Artifact Lock

Created: `artifacts/vn_history_deployment/artifact_lock.json`

The lock contains:

- `schema_version`
- deterministic `deployment_id`
- shared base model
- per-role adapter config hash, weight hash, and weight size
- corpus hash and count
- FAISS hash, `ntotal`, and dimension
- BM25 manifest hash and count
- inference config hash
- model registry hash

`deployment_id` derivation is deterministic: SHA256 over the canonical artifact lock payload excluding `deployment_id`, prefixed with `qwen3-`, truncated to 16 hex chars.

Important lock values:

- Corpus SHA256: `7192e64be17fdaab2ce91c04a60c00367e3ceb2475a936e69f11fe5c0df098cb`
- Corpus count: `58603`
- FAISS SHA256: `7fb50c73bf3a8f34cdf705933409254f0bbf68fcaccb3f560892e43049407f54`
- FAISS ntotal/dim: `58603` / `768`
- BM25 manifest SHA256: `c83001bf20177e7f5e42685480caf9cf38a94360218cf2ce1d2cc01a3e784b3e`
- BM25 count: `58603`

## Modal Sync

Dry-run plan before mutation:

- UNCHANGED: all three adapter configs, all three `adapter_model.safetensors`, inference config, model registry, corpus, FAISS, BM25 files, and `EXPORT_SUCCESS.txt`
- UPLOAD: `/artifact_lock.json`
- REPLACE: `/manifest.json`
- DELETE_STALE: 31 adapter-side training/tokenizer metadata files

Remote mutation performed:

- Uploaded `/artifact_lock.json`
- Replaced `/manifest.json` to include `deployment_id`
- Deleted 31 stale files under `/adapters/{research,evidence,history}`

Remote files unchanged:

- `/adapters/research/adapter_model.safetensors`
- `/adapters/evidence/adapter_model.safetensors`
- `/adapters/history/adapter_model.safetensors`
- `/corpus/vn_history_rag_chunks_enriched.jsonl`
- `/retrieval/faiss/chunks.index`
- `/retrieval/bm25s_index/*`
- current config files

Remote files replaced:

- `/manifest.json`

Remote files deleted:

- adapter `README.md`, tokenizer files, `split_manifest.json`, `training_args.bin`, and `training_log.jsonl` residue where present

Were any `adapter_model.safetensors` files modified or uploaded?

NO.

## Post-Sync Sanity

`scripts/modal_artifact_sanity.py` passed after sync:

- Deployment ID: `qwen3-3f5f81301b0ca4ac`
- Artifact lock present: yes
- Manifest deployment ID present: yes
- Research/Evidence/History remote weight hashes: MATCH
- Corpus count: `58603`
- FAISS index exists and non-empty
- BM25 directory contains 6 files

## Working Tree

No commit was created and nothing was pushed. The final working tree intentionally contains implementation changes and these audit reports for review.
