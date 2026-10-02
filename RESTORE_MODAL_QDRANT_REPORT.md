# Restore + Modal + Qdrant report

Ngày kiểm tra: **2026-10-02**, múi giờ **Asia/Saigon**.

**DEFAULT MODAL CORPUS: V1**

**Exact V1 path on Modal Volume: `/artifacts/corpus_v1/chunks.jsonl`**.
Volume canonical: `vn-history-artifacts`, mount `/artifacts`, Volume-relative path `/corpus_v1/chunks.jsonl`.

## A. Environment

| Item | Actual value |
| --- | --- |
| OS | Microsoft Windows 10.0.26100, PowerShell 7.6.5 |
| Repository root | `C:\Users\Admin\Chatbot_answering_vietnamese_history` |
| Python environment | Repo-local `.conda`, Python **3.11.15** |
| Modal CLI/SDK | **1.5.3**, `.conda/Scripts/modal.exe` |
| Qdrant client | **1.19.1** |
| Git branch | `main` |
| Source commit | `f1693da741ae7ddf717b2252c5d3b76f0df3fe3b` |
| Modal workspace/profile | `tungduong156gli`, environment `main` |
| Authentication | PASS: existing login can list Volumes, Secrets and run remote functions |

Runtime paths resolve from `Path(__file__)`/repo root or explicit environment settings. The absolute Windows path above records this inspection only. Source does not depend on the previous machine's user directory.

Installed the missing packages from `requirements.txt`, plus `pytest`; `pip check` passed. No copied Python environment was treated as an artifact backup. This machine already had a functioning `.conda` interpreter; use it below. Modal GPU image installs the existing pinned requirements from the Dockerfile.

## B. Backup archives

All three ZIPs were inspected before extraction. Each has one wrapper directory, `README_RESTORE.md`, `restore_bundle.py`, `bundle_manifest.json`, `SHA256SUMS.txt`, and `payload/` containing **repo-relative paths**. Only payload contents were restored to repo root. Backup helper code was preserved as metadata rather than blindly executed.

| Archive, now under `portable_backup/` | ZIP bytes | Archive files / payload files | Payload bytes | Purpose | Restored location | Verification |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `vn_history_v1_research_runtime_2026-09-30_release.zip` | 3,261,746,127 | 34 / 30 | 4,816,646,573 | V1 current research runtime plus frozen research datasets/caches | `artifacts/corpus_v1/`, original evaluation/training paths | PASS: all payload SHA-256 and sizes |
| `vn_history_v0_legacy_runtime_2026-09-30_release.zip` | 373,761,983 | 15 / 11 | 585,546,807 | Legacy V0 runtime | `artifacts/vn_history_deployment/` | PASS: all payload SHA-256 and sizes |
| `vn_history_peft_adapter_2026-09-30_release.zip` | 132,200,724 | 7 / 3 | 132,191,861 | Existing Qwen3-4B PEFT adapter | `artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/` | PASS: all payload SHA-256 and sizes |

**Total restored payload: 44 files / 5,534,385,241 bytes.** V1's 30 files include 14 serving files and 16 frozen evaluation/training/cache files, all restored locally. Research-only datasets are not needed by the inference container and were not uploaded as runtime data.

Mapping reasons:

| ZIP | Content → destination | Code contract |
| --- | --- | --- |
| V1 | `payload/artifacts/corpus_v1/...` → `<repo>/artifacts/corpus_v1/...`; other payload paths preserved exactly | `Settings.corpus_path`, `retrieval_dir`, V1 runtime manifests; `RAGService` checks corpus/order/index fingerprints |
| V0 | `payload/artifacts/vn_history_deployment/...` → same repo-relative path | Explicit legacy `ARTIFACT_ROOT` retains the original `corpus/`, `config/`, `retrieval/` layout |
| Adapter | `payload/artifacts/models/...` → same repo-relative path | `app/models/qwen.py` needs `adapter_config.json` and `adapter_model.safetensors` for `MODEL_VARIANT=sft` |

Existing adapter JSON files differed only in Git's CRLF checkout line endings. Exact LF ZIP bytes were restored **after** verifying that replacing CRLF with LF yielded the manifest hash. Original bytes are kept in `portable_backup/pre_restore/`. No model/index/corpus was transformed. Subsequent `--verify-only` passed all 44 files.

At the user's request the three ZIPs were moved from repo root into `portable_backup/`. Full archive hashes were checked before and after moving:

```text
V1      f4673cdc4074ef51e6977498c736076040cefd3a40427cfc51a51b1224fcc042
V0      a75a75074b4f0f1a29a463963b703c398d3cdbd17f3c394bb6f96d6b00758913
Adapter 67e90ac1bd2edb608f4f580d8539f39e31698b05778cb3fb80eda06e14994cce
```

## C. Restored local structure

```text
<repo>/
  portable_backup/
    vn_history_*_2026-09-30_release.zip     # 3 preserved archives
    vn_history_v1_research_runtime/        # original README/manifests/checksums/helper
    vn_history_v0_legacy_runtime/
    vn_history_peft_adapter/
    pre_restore/artifacts/models/...      # 2 original CRLF JSON files
  artifacts/
    corpus_v1/
      chunks.jsonl
      documents.jsonl
      retrieval/
        faiss/{chunks.index,manifest.json}
        bm25s_index/{*.npy,params.index.json,phase9_manifest.json,vocab.index.json}
        qdrant/manifest.json
        index_manifest.json
      runtime/{inference_config.json,manifest.json}
    vn_history_deployment/                # preserved V0, 11 files
      corpus/vn_history_rag_chunks_enriched.jsonl
      retrieval/{faiss,bm25s_index}/...
      config/inference_config.json
      manifest.json
    models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/
      adapter/{adapter_config.json,adapter_model.safetensors}
      manifest.json
    evaluation/reports/six_way_best_b4_ga4_e2/retrieval_cache_{faiss,qdrant}.*
  evaluation/datasets/v1_silver/{questions_3000.jsonl,questions_5000.jsonl,splits/v1_seed42/...}
  evaluation/annotation/workspace_codex_5000/frozen_v1.json
  training/datasets/vn_history_rag_sft_v1/{train_sft.jsonl,validation_sft.jsonl,manifest.json,stats.json}
  reports/restore/                        # ignored operational verification records
```

## D. V1 corpus

| Item | Verified value |
| --- | --- |
| Exact local corpus | `<repo>/artifacts/corpus_v1/chunks.jsonl` |
| Exact Modal corpus | `/artifacts/corpus_v1/chunks.jsonl` |
| Corpus format | UTF-8 JSONL, one retrieval chunk per record |
| Chunks | **624,288**, counted by existing validator |
| Chunk bytes | **1,331,391,190** |
| Documents | **145,648**, nonblank JSONL records counted directly |
| Document bytes | **926,690,133** |
| Complete V1 serving tree | **14 files / 4,783,916,770 bytes** |
| Corpus SHA-256 | `4d1700c20c25e8f2c77b801e6ef6242224a8f59a37469ab13fdbe1dc62266255` |
| Ordered chunk-ID SHA-256 | `b42f792bc8b59795861a5329df5fc0835abdfd3449028dcf2eef8e86f3b55d8f` |
| Embedding | `intfloat/multilingual-e5-base`, revision `d128750597153bb5987e10b1c3493a34e5a4502a`, **768 dimensions** |
| FAISS | `retrieval/faiss/chunks.index`, **624,288** vectors, `IndexFlatIP`, 1,917,812,781 bytes |
| BM25 | `retrieval/bm25s_index`, **624,288** documents; 3 `.npy` arrays plus vocab/params/manifest |
| BM25 settings | Existing Lucene method, k1=1.5, b=0.75, delta=0.5, float32/int32 |
| Reranker | `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Runtime metadata | `runtime/inference_config.json`, `runtime/manifest.json`, retrieval component manifests |

No SQLite/Parquet corpus, standalone embedding export, or Qdrant snapshot is contained in the V1 backup. Embeddings are already present in the portable FAISS index; no corpus re-embedding was run. Upstream E5/reranker/base Qwen weights and tokenizer can be pulled from Hugging Face into the separate HF cache Volume. No custom local reranker weights were excluded from the restored serving tree.

The existing `scripts.retrieval.validate_indexes --components local` passed corpus SHA-256, chunk order, index manifests and FAISS/BM25 counts. The restored manifests' historical `/content/drive/...` paths are provenance strings; runtime reads the configured paths, not those strings.

## E. V0 legacy

Local: `<repo>/artifacts/vn_history_deployment/`, **58,603 chunks**, **11 files / 585,546,807 bytes**.
Release V0 on Modal: `/artifacts/v0/` (Volume-relative `/v0/`).

An older tree already exists at `/artifacts/vn_history_deployment/`. Its inference config and runtime manifest differ from the release (4,206/4,523 bytes versus 1,140/1,760 bytes). They were not overwritten or deleted. The release uses `/v0/` to preserve both generations. Existing old files elsewhere in the Volume are outside the upload inventory.

**V0 is not the default.** Local reproduction can explicitly set `ARTIFACT_ROOT=artifacts/vn_history_deployment` and clear all V1 path overrides. Modal's primary entrypoint pins V1 paths and has no fallback to V0. V0 bytes remain available for an explicitly configured separate reproduction job.

## F. PEFT adapter

Source: `portable_backup/vn_history_peft_adapter_2026-09-30_release.zip`.

Local: `<repo>/artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter`.
Modal: `/artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter`.

Weights: `adapter_model.safetensors`, **132,187,888 bytes**. Config: `adapter_config.json`, **1,160 bytes**. Parent `manifest.json` supplies preserved training provenance. Base model remains `Qwen/Qwen3-4B-Instruct-2507`; tokenizer comes from that base model. No merge, retraining, new adapter, or base-model change was performed. `MODEL_VARIANT=vanilla` remains the established default; `MODEL_VARIANT=sft` attaches this restored adapter.

Historical warning: backed-up adapter fingerprint `742c5ffdfb6f0bc0ff8be65558b1836ad7c008033ad3828473064ebb67fe3ece` differs from the frozen six-way benchmark's `ab11d1371a3045c4be75bbb1612df6785ab13ad6c3b508ca5c56cc30a9f91ebc`. Do not claim exact SFT benchmark reproduction or resume its output directory. The backup also records an older SFT index-manifest hash; neither historical record was rewritten.

## G. Modal

| Item | Actual contract |
| --- | --- |
| Main App | `vn-history-rag-api` |
| Entrypoint | `modal_app.py` |
| API function | `fastapi_app`, `@app.function` + `@modal.asgi_app()` |
| Run-once function | `runtime_smoke` |
| Artifact Volume | `vn-history-artifacts` → `/artifacts` |
| HF cache Volume | `vn-history-hf-cache` → `/hf-cache` |
| Conversation Volume | `vn-history-chat-data` → `/data`, SQLite `/data/chat.sqlite3` |
| V1 root/corpus | `/artifacts/corpus_v1` / `/artifacts/corpus_v1/chunks.jsonl` |
| V1 retrieval/runtime config | `/artifacts/corpus_v1/retrieval` / `/artifacts/corpus_v1/runtime/inference_config.json` |
| Default dense backend | `faiss`, established default; explicit `RETRIEVAL_DENSE_BACKEND=qdrant` supported |
| Default model variant | `vanilla`; explicit `MODEL_VARIANT=sft` supported |
| Model artifacts | Adapter path in section F; upstream base weights in `/hf-cache/hub` |
| GPU/resources | A100, 4 CPU, 32 GiB host memory, 0 minimum / 1 maximum API container |
| API timeouts | 600 s invocation, 900 s startup; existing cold-start model downloads can take several minutes |
| Idle behavior | API scales to zero after 120 s; Central uses lazy model loading |
| Production endpoint | `https://tungduong156gli--vn-history-rag-api-fastapi-app.modal.run` |

The application was deployed successfully. Docker `EXPOSE` and `HEALTHCHECK` are ignored by Modal; the ASGI function supplies the web endpoint. Source sends explicit V1 container paths and a fixed Volume adapter path, so host Windows path overrides cannot leak into the container.

Uploaded and SHA-256 verified runtime inventory:

| Namespace on Volume | Files | Bytes |
| --- | ---: | ---: |
| `/corpus_v1/` | 14 | 4,783,916,770 |
| `/v0/` | 11 | 585,546,807 |
| `/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/` | 3 | 132,191,861 |
| `/restore_manifests/` | 6 | 20,358 |
| **Total** | **34** | **5,501,675,796** |

Only extracted files and small manifests are uploaded. ZIPs are portable backups, excluded from Git and Docker context; containers never unzip at cold start. The uploader verifies local hashes, audits existing remote hashes, refuses conflicts, commits each missing file separately, then verifies all remote SHA-256s using a CPU Modal function. Re-running skips identical files. Existing remote V0 data is preserved.

Operational recovery: Windows initially used cp1252 and failed rendering Modal's checkmark; utility output and terminal commands now use UTF-8. An initial all-files concurrent upload encountered gRPC/heartbeat failures. Uploads now run per file outside an App heartbeat, with resumable commits. Terminal setup uses `https://api.modal2.com`, the official failover endpoint in installed SDK 1.5.3 (`modal/config.py`, `DEFAULT_SERVER_URL`). Final deployment and GPU smoke succeeded through it. The final large-file retry uses process-local IPv4 DNS; `--ipv4` exposes this optional workaround in the uploader without modifying machine networking. Earlier primary-endpoint/mixed-address transfers stalled or timed out. No credentials/TLS checks were bypassed.

Upload/final remote verification status: **PASS** at 12:28 Asia/Saigon. All **34 files / 5,501,675,796 bytes** on `vn-history-artifacts` match local manifest sizes and SHA-256 hashes, including the complete V1 corpus/FAISS/BM25 tree, release V0, adapter and backup manifests. Evidence: `reports/restore/modal_volume_verification.json`.

## H. Qdrant

No API key is recorded here, in source, or in Git. User created Modal Secret `vn-history-qdrant` with `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`; all three required variable names were present. The dotenv used to create it is outside the repo in the user's temporary directory.

| Item | Verified value |
| --- | --- |
| URL hostname | `50997117-0583-4a9d-a9af-513bf2c98932.australia-southeast1-0.gcp.cloud.qdrant.io` |
| Collection | `vn_history_v1_e5` |
| Exact point/vector count | **624,288** points, one `dense_e5` vector per point |
| HNSW-indexed vectors reported by collection info | **621,952**, distinct from stored vector presence |
| Vector dimension | **768** |
| Distance | **COSINE** |
| Quantization | None, full precision |
| Status | `green` |
| Modal DNS/network/authentication | **PASS**, remote `connection_probe` using the Secret |
| Local V1 sample compatibility | **PASS**, 7 point payloads/vectors against restored corpus/FAISS |
| Modal V1 sample compatibility | **PASS**, same 7 payload/vector samples and basic search using the injected Secret |
| Basic vector search | **PASS**, 3 hits, first point score 1.0 for its own saved vector |

Payload schema version 1: `chunk_id`, `document_id`, `source_id`, `title`, `url`, `source_article_id`, `source_split`, `historical_filter_decision`, `historical_relevance_score`, `token_count`. Payload omits article text; the application uses zero-based point IDs and the local V1 corpus to recover text and checks chunk IDs. Sample IDs were 0, 1, 12,345, 100,000, 312,144, 500,000 and 624,287; all vector differences versus FAISS were **0.0** locally. This is a sample compatibility check, not an exhaustive audit of all 624,288 remote vectors.

No collection creation/deletion or re-ingestion was necessary. No embedding/chunking/retrieval model was changed. If the remote cluster is lost later, use a genuine Qdrant snapshot if available, or restore normalized vectors from the existing `IndexFlatIP` via its official `reconstruct` API and preserve the saved payload/row-ID contract. The repo's `scripts/retrieval/build_index.py --component qdrant` embeds the corpus again; do not run it as a routine restore while valid saved vectors exist. Such a fallback would involve 624,288 vectors × 768 float32 dimensions (~1.918 GB) and 624,288 payload upserts, not a new embedding model.

An exact `count` with `HasVectorCondition(has_vector="dense_e5")` confirms **all 624,288 points have the stored vector**. The smaller HNSW-indexed count does not indicate missing stored vectors. Qdrant can search small segments by full scan, so indexed and stored counts can differ; see the [official Qdrant FAQ](https://qdrant.tech/documentation/faq/qdrant-fundamentals/#why-the-amount-of-indexed-vectors-doesnt-match-the-amount-of-vectors-in-the-collection). No indexing settings were changed.

Remote corpus/vector sample smoke status: **PASS**. All seven payloads match the V1 schema/source records and vector differences versus restored FAISS are **0.0** from Modal as well. Evidence: `reports/restore/qdrant_smoke.json`.

## I. Source code changes

| File | Change | Reason |
| --- | --- | --- |
| `app/config.py` | V1 default root; paths anchored to repo; V1 runtime paths and strict validation enabled; explicit V0 layout kept | Portable paths and current-corpus default |
| `modal_app.py` | Explicit V1 Volume paths; canonical adapter path; optional Qdrant Secret/backend; remote runtime smoke; startup timeout 900 s | Cloud serving independent of local drive |
| `app/models/qwen.py` | Only set cancellation when a stream ends before `ModelDone`; always join worker | Real Modal chat returned HTTP 500 because successful generation was incorrectly marked cancelled |
| `Dockerfile` | Default `ARTIFACT_ROOT=/artifacts/corpus_v1` | Match current runtime |
| `.env.example` | V1 default, explicit V0 instructions, Secret name | Avoid inadvertent legacy selection |
| `.gitignore` | Release ZIPs and `portable_backup/` ignored | Preserve backups outside Git |
| `.dockerignore` | ZIPs and backup folder excluded | Avoid uploading backups in image context |
| `README.md`, `artifacts/README.md` | Current default and report links; older build claims marked historical | Prevent obsolete serving instructions |
| `scripts/restore_portable_backup.py` | Inspect, safe restore, verify-only, dry-run; manifest/hash/conflict checks | Repeatable 3-archive recovery |
| `scripts/upload_modal_volume.py` | Idempotent missing-file upload + remote hashes, dry-run/verify-only | Safe Volume sync |
| `scripts/modal_artifact_checks.py` | CPU remote SHA-256 audit | Verify actual mounted bytes |
| `scripts/smoke_test_qdrant.py` | Read-only count/schema/payload/vector/search test | Detect old/wrong collection |
| `scripts/modal_qdrant_smoke.py` | Secret-backed remote connection/sample tests | Verify from Modal without exposing credentials |
| `tests/test_qdrant_lanes.py` | Isolate missing-manifest test from restored real artifacts | Existing test assumed absent default corpus |
| `tests/test_restore_config.py` | Default V1, repo-relative and explicit V0 path contracts | Guard deployment configuration |
| `tests/test_six_way_memory.py` | Verify completed generation leaves caller cancellation clear | Regression coverage for the observed HTTP 500 |
| `RESTORE_MODAL_QDRANT_REPORT.md`, `MODAL_QUICKSTART.md` | Long report and short terminal workflow | Fresh-terminal operation |

Adapter JSONs have exact restored LF bytes and no semantic/source diff; preserved original CRLF bytes are ignored backup files. `requirement.txt` appeared as an unrelated untracked file during this session and was left untouched. No commit, staging, or push was performed.

The repository already tracks `training/InvestigatingDataset.zip` (52,273 bytes) in the original commit. It is an unrelated historical dataset archive and was preserved unchanged. None of the three release backup ZIPs, restored model weights, `.env` files or credentials is tracked or staged.

Hard-coded path audit:

| Occurrence group | Classification | Action |
| --- | --- | --- |
| Old Modal root `/artifacts` selected incompatible bundle layout | runtime problem / must change | Set explicit `/artifacts/corpus_v1/...` paths |
| Default `artifacts/vn_history_deployment` in config/Docker/env example | runtime problem / must change | Default now V1; V0 remains explicit |
| Absolute Colab paths in restored retrieval/training/benchmark manifests | backup metadata / safe to keep | Preserve bytes and fingerprints |
| `docs/CORPUS_V1_COLAB.md`, `docs/CORPUS_V1_INDEXING.md`, `docs/TRAINING_AND_EVALUATION.md` Colab examples | documentation only | Preserve historic instructions |
| `scripts/colab/bootstrap.py` `/content` paths | safe to keep | Explicit Colab utility, not serving runtime |
| Repository name in package/GitHub links | safe to keep | Repository identity, no local path dependency |
| `portable_backup` in restore/upload utilities and ignore files | safe to keep | Dynamic repo-relative backup location |
| `C:\Users\PC\...`, old Windows drive paths in deployed `app/`/Modal entrypoint | None found | No runtime dependency remains |

The source search and a per-file/per-line inventory are stored in ignored `reports/restore/path_audit.json`. Git-ignored corpus/index manifests were inspected through their ZIP inventories as well. Historical docs and metadata were not mass-rewritten.

## J. HOW TO RUN FROM A FRESH TERMINAL

All commands below are **PowerShell**, verified against installed Modal **1.5.3**. Start in this repository (or any subdirectory of it). The first block puts the existing `.conda` runtime on PATH; no globally installed `python`/`modal` is assumed.

### First-time setup

```powershell
Set-Location (git rev-parse --show-toplevel)
$repo = (Get-Location).Path
$env:Path = "$repo\.conda;$repo\.conda\Scripts;$repo\.conda\Library\bin;$env:Path"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MODAL_SERVER_URL = 'https://api.modal2.com'
python --version
modal --version
# Dependencies are installed already; run only when rebuilding the environment:
python -m pip install -r requirements.txt modal==1.5.3
```

Authentication is already working. On a new/unlogged-in profile only:

```powershell
modal token new
```

Volumes already exist. On another Modal environment, create missing canonical names after inspecting `modal volume list --json`:

```powershell
modal volume create vn-history-artifacts
modal volume create vn-history-hf-cache
modal volume create vn-history-chat-data
```

`volume create` has no `--allow-existing` flag in this CLI; don't blindly recreate existing names. The API also creates the chat Volume if missing.

Secret already exists. To create it on a different environment without typing a key in shell history, put three variables into a local ignored/outside-repo dotenv via an editor, then:

```powershell
# Dotenv contents: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION=vn_history_v1_e5
modal secret create vn-history-qdrant --from-dotenv "$env:TEMP\vn_history_qdrant.env"
modal secret list --json
```

Don't add `--force` unless intentionally rotating an existing Secret. Modal 1.5.3 also accepts `QDRANT_API_KEY=-` to open an editor instead of exposing it in the command.

### Verify environment

```powershell
modal --help
modal volume --help
modal secret --help
modal volume list --json
modal secret list --json
python -m pip check
python -c "from app.config import settings; print(settings.corpus_path); print(settings.retrieval_dir); print(settings.manifest_path)"
python -m scripts.restore_portable_backup --verify-only
python -m scripts.retrieval.validate_indexes --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --components local
```

### Restore / upload / sync artifacts

No restore is needed on this already restored machine. For a fresh checkout, place the three ZIPs in repo root **or** `portable_backup/`, then:

```powershell
python -m scripts.restore_portable_backup --restore-line-endings --dry-run
python -m scripts.restore_portable_backup --restore-line-endings
python -m scripts.restore_portable_backup --verify-only
python -m scripts.upload_modal_volume --include-v0 --dry-run
python -m scripts.upload_modal_volume --include-v0 --ipv4
python -m scripts.upload_modal_volume --include-v0 --verify-only
modal volume ls vn-history-artifacts /corpus_v1 --json
modal volume ls vn-history-artifacts /corpus_v1/retrieval/faiss --json
modal volume ls vn-history-artifacts /models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter --json
```

Official recursive CLI syntax is also available: `modal volume put vn-history-artifacts artifacts/corpus_v1 /corpus_v1` (no extra `--recursive`). Use it only for a new empty destination; the utility above provides hash-aware conflict checks and interrupted-upload recovery. A trailing `/` on a remote path changes CLI basename handling; use exact destination roots to avoid `corpus_v1/corpus_v1` nesting. Do not use `--force` to hide conflicts.

### Run once

```powershell
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'
$env:MODEL_VARIANT = 'vanilla'
New-Item -ItemType Directory -Force reports/restore | Out-Null
modal run --write-result reports/restore/runtime_retrieval.json modal_app.py::runtime_smoke
modal run --detach --write-result reports/restore/runtime_generation.json modal_app.py::runtime_smoke --generate --central
```

`runtime_smoke` returns a JSON string, compatible with CLI `--write-result`. It starts the actual FastAPI lifespan, checks V1 readiness, calls the real retrieval endpoint and optionally both generation modes. It uses a temporary SQLite store rather than changing user conversations. `fastapi_app` is an ASGI factory: use `serve`/`deploy` for its HTTP behavior, rather than treating its return value as a chat answer.

To check SFT with Qdrant:

```powershell
$env:RETRIEVAL_DENSE_BACKEND = 'qdrant'
$env:MODAL_QDRANT_SECRET_NAME = 'vn-history-qdrant'
$env:MODEL_VARIANT = 'sft'
modal run --detach --write-result reports/restore/runtime_qdrant_sft.json modal_app.py::runtime_smoke --generate --central
```

### Serve development

```powershell
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'  # or explicit 'qdrant'
$env:MODEL_VARIANT = 'vanilla'         # or explicit 'sft'
modal serve modal_app.py
```

CLI prints a development URL with `-dev.modal.run`; leave the process running. `Ctrl+C` ends that development App. From a second terminal test the printed URL. `npm run dev` also starts the frontend plus this Modal command if npm dependencies are installed, but is not required for terminal API operation.

### Deploy Modal

```powershell
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'  # set 'qdrant' before deploy to select remote dense backend
$env:MODEL_VARIANT = 'vanilla'
modal deploy modal_app.py
modal app list --json
```

Production URL created on this workspace:
`https://tungduong156gli--vn-history-rag-api-fastapi-app.modal.run`.
Deployment persists after the terminal closes, while idle containers scale down. Changing local environment variables affects the next `serve`/`run`/`deploy`, not an existing deployed image. For Qdrant deployment, explicitly set the backend and Secret name first; a missing Secret/key/schema fails instead of silently switching to FAISS or V0.

### Check logs / shutdown

```powershell
modal app logs vn-history-rag-api --tail 100
modal app logs vn-history-rag-api -f
# To intentionally stop the deployed API:
modal app stop vn-history-rag-api
# Start it again by deploying:
modal deploy modal_app.py
```

In Modal 1.5.3 `app logs` fetches recent logs and exits by default; `-f` follows. Stopping the deployed App terminates its containers but leaves Volumes/Secrets intact. No stop command was run against user deployments as part of restore.

### Test endpoint

```powershell
$base = 'https://tungduong156gli--vn-history-rag-api-fastapi-app.modal.run'
Invoke-RestMethod "$base/health" -ConnectionTimeoutSeconds 60 -OperationTimeoutSeconds 1200
Invoke-RestMethod "$base/ready" -ConnectionTimeoutSeconds 60 -OperationTimeoutSeconds 1200
$retrievalBody = @{question='Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?'; final_k=3} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/api/v1/retrieve" -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($retrievalBody)) -ConnectionTimeoutSeconds 60 -OperationTimeoutSeconds 1200
$headers = @{'X-Client-ID'='terminal-smoke'}
$c = Invoke-RestMethod -Method Post -Uri "$base/api/v1/conversations" -Headers $headers -ContentType 'application/json' -Body '{}'
$body = @{conversation_id=$c.id; question='Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?'; mode='hybrid'; final_k=3} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/api/v1/chat" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ConnectionTimeoutSeconds 60 -OperationTimeoutSeconds 1200
```

Use `mode='central'` with a separate conversation for Central. `/api/v1/chat/stream` is the real SSE endpoint (`status`, `answer_delta`, `done`/`error`). Keep `X-Client-ID` consistent for conversation ownership. Expected `/ready`: `ready=true`, `corpus_chunks=624288`, `bm25_documents=624288`, selected `dense_backend`, and Hybrid variant/adapter attachment flags. `/health` alone does not prove retrieval/model readiness. For `modal serve`, substitute the CLI's development URL.

### Qdrant smoke test

The preferred command uses Secret injection entirely inside Modal:

```powershell
modal run --write-result reports/restore/qdrant_probe.json scripts/modal_qdrant_smoke.py::connection_probe
modal run --write-result reports/restore/qdrant_smoke.json scripts/modal_qdrant_smoke.py::qdrant_smoke
```

Local alternative: populate environment variables or an ignored `.env` with the three Qdrant variables, then `python -m scripts.smoke_test_qdrant`. Do not paste API keys into chat, report or tracked files. The sample smoke is read-only and compares preserved vectors; it does not embed/reindex the corpus.

## K. Smoke-test results

| Component | Status | Evidence |
| --- | --- | --- |
| Three archives and safe mapping | PASS | All inventories/manifests/checksums inspected before extract |
| Local restore and re-verification | PASS | 44 payload files, 5,534,385,241 bytes, SHA-256 match |
| Corpus V1 count/hash/order | PASS | Existing validator: 624,288; exact corpus/ordered IDs hashes above |
| V1 FAISS + BM25 | PASS | Both 624,288; artifact bytes preserved |
| Local real index loading/search | PASS | Actual `RAGService` loads corpus/FAISS/BM25; BM25 question returns 80 candidates, saved-vector FAISS query returns 3 hits with row 0 as self-hit |
| Documents | PASS | 145,648 records, 926,690,133 bytes |
| Project/config/Modal imports | PASS | No missing default V1 paths; `app.main` and `modal_app` import |
| Dependencies | PASS | `pip check`: no broken requirements |
| Focused tests | PASS | 57 passed, including successful generation, cancellation, worker cleanup, model variants and V1 configuration |
| Modal authentication/Volumes | PASS | Canonical Volumes already present |
| Modal Secret + Qdrant connection | PASS | Remote probe using Secret, collection green/count/dimension correct |
| Qdrant local payload/vector/search | PASS | Seven vectors match FAISS exactly, basic search 3 hits |
| Modal deployment | PASS | Production ASGI URL created |
| Remote upload SHA-256 | PASS | 34 files / 5,501,675,796 bytes; all remote hashes match local manifests |
| Modal Qdrant V1 sample smoke | PASS | Seven exact payload/vector matches, zero difference vs FAISS, 3 search hits |
| Real Hybrid/Central retrieval/generation | PASS | A100, Qdrant V1 + BM25 + reranker; SFT adapter attached; both modes status `done`, 3 evidence sources each |
| External HTTP endpoint readiness | PASS | Production `/health`, `/ready`, `/api/v1/retrieve`, `/api/v1/chat` HTTP 200; 3 evidence sources; status `done`; own smoke conversation deleted with HTTP 204 |

Operational result files are kept under ignored `reports/restore/`: restore verification, Volume inventory/hash audit, Qdrant probe/sample records, runtime smoke and path audit. They contain no API key.

The first real GPU chat exposed an existing cancellation bug: `QwenRuntime.stream()` set the caller's cancellation event in cleanup even after producing `ModelDone`. The API discarded that completed response and returned HTTP 500. Cleanup now cancels only an unfinished stream and still joins the generation worker. This is a runtime blocker fix; model weights, prompting, chunking, retrieval and evaluation methods were preserved. Regression coverage verifies successful completion, cancellation, errors and worker cleanup. `--detach` keeps the bounded smoke function alive if its terminal disconnects.

## L. Remaining issues

- **BLOCKER:** none remaining. Upload, remote hashes, Qdrant compatibility, real Hybrid/Central generation and production HTTP smoke all passed.
- **WARNING:** historical adapter/benchmark fingerprint mismatch; exact SFT benchmark identity unverified, as recorded by the backup.
- **WARNING:** earlier build/audit docs describe old states; use this report/quick start for current deployment. The historical preservation inventory includes intentionally excluded files and is not a requirement to reconstruct every old training workspace.
- **OPTIONAL:** freeze upstream base model revisions for a separately reviewed reproducibility experiment. This restore preserves existing model choices/revisions and does not change evaluation methodology.
- **OPTIONAL:** create an official Qdrant snapshot for future portability; existing ZIPs carry metadata and FAISS vectors, not the remote database.

## Acceptance checklist

- [x] 3 ZIPs identified and preserved under `portable_backup/`.
- [x] V1, V0 and PEFT adapter restored and locally hash-verified.
- [x] Runtime independent of old-machine paths; V1 default and explicit container paths.
- [x] Canonical Modal Volumes exist; no namespace/model/retrieval algorithm changed.
- [x] V1, V0 release and model runtime files fully uploaded and remote-hash-verified.
- [x] Qdrant credentials in Modal Secret, absent from source/report/Git.
- [x] Modal connects to green V1 collection, 624,288 × 768 COSINE.
- [x] Remote payload/vector comparison and real pipeline smoke complete.
- [x] Production HTTP readiness, retrieval and generation use V1; no legacy fallback.
- [x] Exact terminal commands, long report and quick start created.

Final Git review is recorded below. No large payload/ZIP/secret/cache is staged or committed.

Real smoke outcomes (operational checks, not a benchmark): Qdrant/SFT Hybrid returned `done`, 3 sources, 141 answer characters; Central returned `done`, 3 sources, 290 characters. The public FAISS/vanilla API returned the same successful response contract. Its first cold start took about 102 seconds; subsequent retrieval and chat completed in about 3 and 14 seconds in this run. The test's conversation was removed; user conversations were left intact. Evidence: `runtime_qdrant_sft_smoke.json` and `http_smoke.json` under ignored `reports/restore/`.

## Final Git review

`git status --short` and full `git diff` were reviewed. `git diff --check` passed. The tracked semantic diff is **11 files, 143 insertions, 34 deletions**; new utility/test/docs files are untracked and listed below. The two restored JSON files appear modified due to checkout line endings and have no semantic Git diff. No files were staged or committed.

```text
 M .dockerignore
 M .env.example
 M .gitignore
 M Dockerfile
 M README.md
 M app/config.py
 M app/models/qwen.py
 M artifacts/README.md
 M artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter/adapter_config.json
 M artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/manifest.json
 M modal_app.py
 M tests/test_qdrant_lanes.py
 M tests/test_six_way_memory.py
?? MODAL_QUICKSTART.md
?? RESTORE_MODAL_QDRANT_REPORT.md
?? requirement.txt                       # unrelated, left untouched
?? scripts/modal_artifact_checks.py
?? scripts/modal_qdrant_smoke.py
?? scripts/restore_portable_backup.py
?? scripts/smoke_test_qdrant.py
?? scripts/upload_modal_volume.py
?? tests/test_restore_config.py
```

Credential comparison against the actual local Qdrant key passed for all 20 semantic changed/untracked files; the key was never printed. Release ZIPs are ignored under `portable_backup/`, with no ZIP left at repo root. Large corpus/index/adapter files and generated operational records are ignored. The final status and full diff are also saved in ignored `reports/restore/git_status.txt` and `git_diff.patch`.
