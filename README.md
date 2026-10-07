# 🇻🇳 Vietnamese History LLM ✨

**🐍 Python 3.11 · ⚡ FastAPI · ⚛️ React · ☁️ Modal · 📚 RAG**

A Vietnamese history chatbot with cited answers, a React interface, and a FastAPI backend. It supports two inference modes, a shared history retriever, optional Hybrid fine-tuning, and reproducible quality and latency evaluation.

| Mode | Execution path | Model |
| --- | --- | --- |
| 🔎 `hybrid` | Hybrid retrieval → cited prompt → streamed answer | `Qwen/Qwen3-4B-Instruct-2507`, vanilla or a local SFT PEFT adapter |
| 🧠 `central` | Tool planner → permitted evidence tools → cited final answer | Vanilla `Qwen/Qwen3-8B` |

Both modes use the same Corpus V1 history retriever: multilingual E5 dense search with FAISS or Qdrant, BM25S lexical search, weighted RRF fusion, cross-encoder reranking, and context selection. The UI supports conversations, PDF/image attachments, source citations, request-level retrieval selection, Central tool permissions, and a developer trace.

Transformers is the default inference backend and supports Hybrid vanilla/SFT and Central. The optional **vLLM backend requires Linux CUDA, vanilla models, and a deployment with only one inference mode enabled**. See [vLLM experiments](docs/VLLM_LATENCY_EXPERIMENTS.md). SSE `answer_delta` events carry actual generated text; status and retrieval progress can arrive before the first answer token.

## 🧭 Start here

- 💻 [Local setup](#run-locally): start with `api-only`, then enable retrieval or generation after restoring artifacts.
- ☁️ [Modal setup](#run-on-modal): run GPU inference remotely and the frontend locally.
- ⚙️ [Runtime configuration](#runtime-configuration): models, SFT, retrieval backends, and tools.
- 📊 [Evaluation and benchmarks](#evaluation-and-benchmarks): score saved answers and measure real HTTP/SSE latency.
- 🎓 [Training](#training): frozen SILVER data, QLoRA, and six-system comparisons.
- 🏗️ [Architecture](docs/ARCHITECTURE.md), [Modal quick start](MODAL_QUICKSTART.md), and [runtime telemetry](docs/RUNTIME_TELEMETRY.md): detailed guides.

## 🗂️ Repository map

| Location | Purpose |
| --- | --- |
| [`app/`](app/README.md) | FastAPI routes, Hybrid/Central runtimes, model streaming, conversations, attachments, and tools. |
| [`frontend/`](frontend/README.md) | React/Vite chat interface. |
| [`training/`](training/train_qwen3.py) | QLoRA training, adapter merging, and preserved legacy source data. |
| [`benchmarks/latency/`](docs/BASELINE_BENCHMARK.md) | HTTP/SSE latency runner, frozen workloads, and comparison reports. |
| [`evaluation/`](docs/EVALUATION.md) | Retrieval, answer, grounding, citation, and six-system evaluation. |
| [`config/`](config/mcp_servers.example.json) | Example MCP server configuration. |
| [`configs/`](configs/benchmarks/vllm_latency_hybrid_v1.json) | Corpus, annotation, and benchmark experiment settings. |
| [`scripts/corpus/`](scripts/corpus/audit_corpus.py) | Read-only preservation audit and explicit corpus utilities. |
| [`scripts/corpus_v1/`](scripts/corpus_v1/cli.py) | Standalone, resumable Vietnamese Wikipedia Corpus V1 construction. |
| [`scripts/retrieval/`](scripts/retrieval/build_index.py) | Explicit index building, validation, and FAISS/Qdrant comparison. |
| [`tools/drive_cli.py`](tools/drive_cli.py) | Verified transfers of datasets, adapters, and evaluation/benchmark results through a mounted Drive directory. |
| [`docs/`](docs/EXPERIMENTS.md) | Architecture, preservation, annotation, training, and experiment guides. |

## 📦 Data and artifacts

> 📌 **A fresh clone can run `api-only`.** Restore the runtime corpus, indexes, base-model weights, and adapter weights separately for retrieval and generation.

Startup validates the configured files and does not rebuild indexes or fall back to V0.

| Data | Purpose and availability |
| --- | --- |
| Corpus V1 | Default runtime: 624,288 chunks with FAISS/BM25S indexes and Qdrant metadata under `artifacts/corpus_v1/`; Git-ignored. |
| Corpus V0 | Preserved legacy runtime: 58,603 enriched chunks and protected indexes; available through a separately restored legacy bundle. |
| `training/Dataset/` | Preserved historical source packs; the merged JSONL contains 520 chunk-ID records. This is separate from the runtime corpus and current SFT dataset. |
| `evaluation/datasets/fixtures/questions.jsonl` | Two tracked, unlabeled questions for pipeline smoke tests. |
| `evaluation/datasets/v1_silver/` | Frozen 3,000-question SILVER benchmark, with 2,400/300/300 train/validation/test splits; local generated files are Git-ignored. |
| `training/datasets/vn_history_rag_sft_v1/` | Git-ignored generated SFT data: 3,000 training rows from 2,400 train IDs and 300 validation rows. |
| `evaluation/datasets/latency_v1/` | Tracked 100-question latency workload and provenance manifest, selected from the frozen SILVER test split. |
| `artifacts/models/qwen3_4b_sft_v1/` | Tracked adapter configuration and run metadata; adapter weights are supplied separately. |
| `artifacts/evaluation/reports/six_way_best_b4_ga4_e2/` | Tracked six-system predictions, metrics, and summaries for a recorded experiment. |

SILVER labels are automatically annotated and are not human-reviewed GOLD. The two fixture questions have no quality labels. Existing results apply to their recorded model, dataset, prompts, and runtime; they do not establish historical correctness or a vLLM performance improvement.

The expected V1 layout is:

```text
artifacts/corpus_v1/
├── chunks.jsonl
├── retrieval/
│   ├── index_manifest.json
│   ├── faiss/                 # chunks.index and manifest.json
│   ├── bm25s_index/           # BM25S files and phase9_manifest.json
│   └── qdrant/                # manifest.json; vectors live in Qdrant
└── runtime/
    ├── inference_config.json
    └── manifest.json
```

Restore the existing release ZIPs and Modal Volumes using [the restoration report](RESTORE_MODAL_QDRANT_REPORT.md) and [Modal quick start](MODAL_QUICKSTART.md). These guides include details of the restored machine; use your own paths, credentials, and endpoint URLs on a new checkout. [The V1 indexing guide](docs/CORPUS_V1_INDEXING.md) covers explicit artifact preparation.

<a id="run-locally"></a>

## 💻 Run locally

Use **Python 3.11** (the Docker runtime version), **Node.js 24**, and npm. Run the following PowerShell commands from the repository root. Full generation needs a compatible CUDA/PyTorch installation and enough GPU memory for the enabled models and retriever. OCR for image attachments also needs Tesseract with Vietnamese/English language data; the Docker images install it.

### 1. 🛠️ Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm install
npm --prefix frontend install

if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path frontend/.env)) { Copy-Item frontend/.env.example frontend/.env }
```

The runtime dependency bundle does not install the Modal CLI, pytest, or the training stack. Install those separately when needed. Keep credentials in Git-ignored environment files or deployment secrets.

### 2. 🎛️ Choose the API mode

Edit the root `.env`:

| `APP_MODE` | Behavior | Required artifacts |
| --- | --- | --- |
| `api-only` | API health, conversations, and lightweight features; no history retrieval or answer generation. | None. |
| `retrieval-only` | History retrieval without Qwen answer generation. | Corpus and the selected retrieval indexes/configuration. |
| `full` | Enabled Hybrid/Central inference modes. | Retrieval artifacts and base models; local adapter weights for Hybrid SFT. |

For an initial local API smoke test:

```dotenv
APP_MODE=api-only
DEVICE=cpu
INFERENCE_BACKEND=transformers
MODEL_VARIANT=vanilla
MODEL_ADAPTER_PATH=
```

For full local inference after restoring V1 artifacts, change `APP_MODE=full` and `DEVICE=cuda`. Keep `ARTIFACT_ROOT=./artifacts/corpus_v1`. Base models can download into the configured Hugging Face cache; `MODEL_LOCAL_FILES_ONLY=true` requires them to be cached already. An explicit `ARTIFACT_ROOT=./artifacts/vn_history_deployment` selects the legacy V0 layout.

### 3. 🚀 Start the API and frontend

In one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Set this in **`frontend/.env`**, then start the frontend in another terminal:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

```powershell
npm run frontend
```

Open `http://localhost:5173`. Vite reads its API URL from `frontend/.env`; the similarly named root `.env` variable does not configure the frontend. Restart Vite after changing the URL, and rebuild for a production frontend build.

Check API health and capabilities:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`. `/health` confirms that the API responds; `/ready` describes the configured runtime, available retrieval backends, tools, and model load state. With lazy loading, Central can remain unloaded until its first request.

<a id="run-on-modal"></a>

## ☁️ Run on Modal

Modal runs the backend in Linux/CUDA while the frontend stays local. Install and authenticate the CLI in your Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install modal
.\.venv\Scripts\python.exe -m modal token new
```

Before starting, provision the existing `vn-history-artifacts` and `vn-history-hf-cache` Volumes using the restoration guide. The app opens them with `create_if_missing=False`; `vn-history-chat-data` is created automatically. V1 must exist at `/artifacts/corpus_v1/` on the artifact Volume.

For the combined Transformers Hybrid/Central runtime, set these values in the root `.env`:

```dotenv
INFERENCE_BACKEND=transformers
MODEL_VARIANT=vanilla
MODAL_GPU_CLASS=A100-40GB
ENABLE_HYBRID_MODE=true
ENABLE_CENTRAL_MODE=true
```

`modal_app.py` reads this file before creating the Modal app. Existing shell environment variables take precedence. Modal configures `APP_MODE=full`, `DEVICE=cuda`, and the mounted V1 paths independently of the local API settings. It uses lazy Central loading by default.

Start an ephemeral development backend:

```powershell
.\.venv\Scripts\python.exe -m modal serve modal_app.py
```

Copy the `https://...-dev.modal.run` URL printed by the CLI into `VITE_API_BASE_URL` in `frontend/.env`, then run `npm run frontend`. Stop the backend with `Ctrl+C`; restart it after changing root `.env` settings. Once the URL is configured and the virtual environment is activated with `.\.venv\Scripts\Activate.ps1`, **`npm run dev` starts both Vite and `modal serve`**. Use `npm run frontend` for a separately running backend.

For a persistent deployment:

```powershell
.\.venv\Scripts\python.exe -m modal deploy modal_app.py
```

Use the deployment URL printed by Modal in the frontend configuration. [Modal quick start](MODAL_QUICKSTART.md) covers volume checks, Qdrant secrets, runtime smoke tests, and deployment diagnostics.

<a id="runtime-configuration"></a>

## ⚙️ Runtime configuration

### 🧠 Models and Hybrid SFT

Keep `HYBRID_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507` and `CENTRAL_MODEL_ID=Qwen/Qwen3-8B`. Central uses the vanilla 8B model. The following Hybrid configurations use the same 4B base tokenizer:

| Setting | Vanilla | SFT |
| --- | --- | --- |
| `MODEL_VARIANT` | `vanilla` | `sft` |
| Local `MODEL_ADAPTER_PATH` | Empty | Directory containing `adapter_config.json` and `adapter_model.safetensors`. |
| Modal adapter path | Adapter is unused. | `/artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter` on the artifact Volume. |

For local SFT, use `MODEL_ADAPTER_PATH=./artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter`. For Modal, change only `MODEL_VARIANT=sft`; the app supplies the Volume path. Restart the backend after changing the variant or adapter. Missing or incompatible adapter files cause an explicit startup error. Use Transformers for SFT; the current vLLM implementation rejects adapters.

Generation defaults to `DO_SAMPLE=false` and `ENABLE_THINKING=false`. Hybrid vanilla/SFT share `HYBRID_MAX_NEW_TOKENS=1536`; Central defaults to a 256-token planner action budget, two action rounds, and a 1,536-token final answer budget. These are ceilings, and models can stop earlier. Set `HYBRID_MAX_NEW_TOKENS` before local startup or Modal serve/deploy to override it. [Runtime telemetry](docs/RUNTIME_TELEMETRY.md) describes effective budgets, planner timings, finish reasons, and truncation fields.

### 🔎 FAISS and Qdrant

FAISS is the default dense backend. To expose both validated V1 backends in the UI, set:

```dotenv
RETRIEVAL_DENSE_BACKEND=faiss
RETRIEVAL_AVAILABLE_BACKENDS=faiss,qdrant
QDRANT_COLLECTION=vn_history_v1_e5
```

Local Qdrant also requires `QDRANT_URL` and `QDRANT_API_KEY` in the environment and the matching Qdrant manifest. Modal reads `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` from the Secret named by `MODAL_QDRANT_SECRET_NAME` (default `vn-history-qdrant`). Retain `RETRIEVAL_AVAILABLE_BACKENDS=faiss` if Qdrant is not provisioned.

The composer reads available backends from `/ready` and sends `retrieval_backend` per request. Central's `search_history` honors the same selection. Both dense backends share BM25S, fusion, and reranking. A Qdrant failure produces an error; it does not silently switch to FAISS. See [dynamic retrieval](DYNAMIC_RETRIEVAL_REPORT.md) for setup, SSE stages, and recorded measurements.

### 🧰 Central tools and MCP

Central can use history retrieval, uploaded documents, Wikipedia, and an optionally configured web search provider. Its Tools popover applies permissions to each request. Web search is disabled by default and needs a supported provider/key; see [built-in tools](app/tools/README.md).

MCP is disabled by default. Copy `config/mcp_servers.example.json` to the private, Git-ignored `config/mcp_servers.local.json`, configure selected servers and environment-based credentials, then set `MCP_ENABLED=true` and `MCP_CONFIG_PATH`. Modal may also need `MODAL_MCP_SECRET_NAME` for server credentials. See [MCP integration](MCP_INTEGRATION_REPORT.md) for transports, configuration, and smoke tests.

### ⚡ Optional vLLM

Use the isolated `requirements-vllm.txt` bundle and `Dockerfile.vllm` on Linux CUDA. Keep it in a separate environment from `requirements.txt`; the two bundles pin different Torch/FastAPI/NumPy versions. Native Windows is suitable for the benchmark client and Modal CLI.

For an isolated vanilla Hybrid vLLM development endpoint on Modal:

```powershell
.\.venv\Scripts\python.exe -m scripts.deploy_inference_benchmark --backend vllm --mode hybrid --name vn-history-bench-vllm-hybrid --max-model-len 32768 --serve
```

The helper enables only the requested model, selects the vLLM-compatible image, and defaults to L4 for isolated benchmarks. That GPU setting is separate from the combined application configuration above. Point the existing frontend at the returned URL. Full vLLM startup rejects enabling Hybrid and Central together; SFT/LoRA is unsupported. Context capacity and GPU memory settings must fit the chosen model/hardware. [The vLLM guide](docs/VLLM_LATENCY_EXPERIMENTS.md) covers reference deployments, parity checks, and concurrent experiments; [the implementation report](VLLM_INFERENCE_LATENCY_REPORT.md) records validation limits and failed GPU attempts.

<a id="evaluation-and-benchmarks"></a>

## 📊 Evaluation and benchmarks

Start a full-mode API with the intended artifacts and models. The latency runner makes real HTTP requests to `POST /api/v1/chat/stream`; use a fresh output directory for each experiment:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.latency.runner --mode hybrid --base-url http://127.0.0.1:8000 --dataset evaluation/datasets/fixtures/questions.jsonl --warmup 2 --runs 5 --concurrency 1 --output reports/baseline/hybrid-run-001
.\.venv\Scripts\python.exe -m benchmarks.latency.runner --mode central --base-url http://127.0.0.1:8000 --dataset evaluation/datasets/fixtures/questions.jsonl --warmup 2 --runs 5 --concurrency 1 --output reports/baseline/central-run-001
```

These two-question runs are pipeline smoke measurements. For the frozen 100-question workload, replace `--dataset` with `evaluation/datasets/latency_v1/questions_100.jsonl`. Replace `--base-url` with your Modal URL for remote inference, and choose a timeout appropriate for cold startup (for example `--timeout 900`). Use `--cold-start` to record a separate cold request. Run different concurrency values in separate experiments.

The runner writes `latency_records.jsonl`, `run_metadata.json`, `latency_summary.json`, and `latency_summary.md`. [Benchmark methodology](docs/BASELINE_BENCHMARK.md) distinguishes first status/response from first answer token, keeps cold/warmup/warm populations separate, and reports unavailable server metrics as null. [vLLM experiments](docs/VLLM_LATENCY_EXPERIMENTS.md) adds frozen workloads, provenance guardrails, and C1/C2/C4/C8 comparison matrices. Successful GPU parity and the complete load matrix remain separate validation work; no speedup is assumed.

Score saved answers with the same dataset used to produce them:

```powershell
.\.venv\Scripts\python.exe -m evaluation.runner --dataset evaluation/datasets/fixtures/questions.jsonl --predictions reports/baseline/hybrid-run-001/latency_records.jsonl --output reports/evaluation/hybrid-run-001
```

Unlabeled fixtures yield N/A for metrics requiring reference answers, relevant IDs, or citation labels. Use reviewed labels for historical-quality claims. See [evaluation metrics](docs/EVALUATION.md), [GOLD annotation](docs/GOLD_ANNOTATION_GUIDE.md), and [the experiment template](docs/EXPERIMENTS.md).

<a id="training"></a>

## 🎓 Training

The offline workflow trains a QLoRA adapter for Hybrid's 4B base model using frozen TRAIN-only SILVER examples, selects the best checkpoint by validation loss, and supports verified resume. Install `requirements-training.txt` in a separate CUDA training environment. Restore the canonical SILVER data, split manifests, SFT files, and required corpus artifacts before running it.

The six-system evaluator compares vanilla/SFT across no-RAG, FAISS, and Qdrant. It records dataset/model/adapter identities, predictions, retrieval/citation metrics, progress, and GPU-memory diagnostics. Recorded results are available in [`summary.csv`](artifacts/evaluation/reports/six_way_best_b4_ga4_e2/summary.csv).

See [training and evaluation](docs/TRAINING_AND_EVALUATION.md), [the SILVER freeze](docs/SILVER_V1_FREEZE_SFT.md), and [citation-aware SFT V2](evaluation/silver_citation_v2.py). Git tracks the directory as **`training/`**: use `python -m training.train_qwen3` and `python -m training.merge_adapter` on case-sensitive systems. A Windows checkout may display the directory as `Training/`. Offline SFT data retains its historical response-mode instructions; the live UI/API has no response-detail selector.

## 🛡️ Preserve and audit the corpus

[`docs/corpus_preservation_manifest.json`](docs/corpus_preservation_manifest.json) records the historical preservation snapshot and hashes. It is a legacy snapshot, separate from V1 runtime identity checks. Removed legacy SFT JSONL entries are marked `retired_legacy`. If protected V0 files have not been restored at their recorded paths, or were moved to `artifacts/old_corpus/`, the audit reports them missing even when V1 works.

Check that snapshot without changing data:

```powershell
.\.venv\Scripts\python.exe -m scripts.corpus.audit_corpus
```

The command prints JSON and returns a nonzero status for missing/changed protected files. For descriptive V1 diagnostics, write a separate report outside the corpus directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.corpus.audit_corpus --corpus artifacts/corpus_v1/chunks.jsonl --output reports/corpus/v1-audit.json
```

See [corpus preservation](docs/CORPUS_PRESERVATION.md) for interpretation. Corpus/index construction is always an explicit utility operation; server startup does not run it.

## 🧪 Verify changes

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
npm --prefix frontend test
npm run frontend:lint
npm run frontend:build
```

Tests use fake model streams and temporary fixtures and do not download Qwen weights. Some preservation/dataset checks inspect existing local frozen artifacts. Live historical quality, OCR, GPU memory, backend parity, and latency require separate checks on the intended deployment. See [test guidance](tests/README.md).

## 🔧 Common setup issues

| Symptom | Check |
| --- | --- |
| Chat generation is unavailable | Set `APP_MODE=full` locally and restore retrieval/model artifacts; `api-only` does not generate answers. |
| V1 startup reports missing or mismatched artifacts | Restore the correct bundle and verify the V1 layout/manifests; startup does not rebuild indexes. |
| SFT fails to load | Check the matching base model, adapter config and weights, `MODEL_VARIANT`, and the local or Modal adapter path. |
| UI calls the wrong backend or cannot connect | Set `VITE_API_BASE_URL` in `frontend/.env`, restart Vite, and check `/health` and `CORS_ORIGINS`. |
| `.env` changes appear ignored | Remove stale shell overrides and restart the backend; Modal supplies its own full-mode/device/Volume paths. |
| Qdrant is unavailable | Check its manifest, collection identity, URL/key, and Modal Secret; enable only provisioned retrieval backends. |
| vLLM fails to start | Use Linux CUDA/the optional image, vanilla, one enabled inference mode, and a context/memory configuration that fits the GPU. |
