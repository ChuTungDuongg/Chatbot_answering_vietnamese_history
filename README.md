# Vietnamese History LLM

A reproducible baseline for answering Vietnamese history questions with grounded citations. It has two inference modes:

| Mode | Path | Model |
|---|---|---|
| `hybrid` | Existing hybrid retrieval → cited prompt → answer | Vanilla `Qwen/Qwen3-4B-Instruct-2507` |
| `central` | Tool-using agent → history retrieval and other configured tools → final answer | Vanilla `Qwen/Qwen3-8B` |

Both modes use the same preserved historical corpus and shared retriever. The baseline keeps multilingual E5, FAISS, BM25S, weighted RRF, cross-encoder reranking, and context selection. Neither mode requires a LoRA adapter. Generation defaults to `do_sample=false` and `enable_thinking=false`.

The project includes real model-token SSE streaming, an HTTP latency benchmark, retrieval and answer evaluation, and a read-only corpus audit. See [the architecture](docs/ARCHITECTURE.md) for the execution paths.

## Repository map

| Location | Purpose |
|---|---|
| [`app/`](app/README.md) | FastAPI, both runtimes, retrieval, model streaming, conversations, and tools. |
| [`frontend/`](frontend/README.md) | React chat UI with Hybrid RAG and Central Agent selection. |
| [`benchmarks/latency/`](docs/BASELINE_BENCHMARK.md) | External HTTP/SSE latency measurement and reports. |
| [`evaluation/`](docs/EVALUATION.md) | Versioned questions and separate retrieval, answer, grounding, and citation metrics. |
| [`scripts/corpus/`](scripts/corpus/audit_corpus.py) | Read-only preservation audit and explicit corpus utilities. |
| [`scripts/retrieval/`](scripts/retrieval/build_index.py) | Explicit index-build utility; server startup does not run it. |
| [`scripts/corpus_v1/`](scripts/corpus_v1/cli.py) | Standalone, resumable Vietnamese Wikipedia Corpus V1 construction and audit. |
| [`scripts/colab/`](scripts/colab/bootstrap.py) | Optional Google Drive mount and workspace setup. |

## Data versions

| Data | Status | Meaning |
| --- | --- | --- |
| Corpus V0 | Frozen runtime baseline | 58,603 enriched chunks plus protected FAISS/BM25S indexes; current Hybrid and Central retrieval source. |
| Corpus V1 | Offline build pipeline, no accepted build in this repository | The `uvw-2026` preset uses all three splits of `undertheseanlp/UVW-2026`; build to a separate `corpus_v1/` directory, preferably Drive. No quality superiority claim until evaluation. |
| `training/Dataset/` | Protected legacy historical source data | 520 chunk-ID records in its merged JSONL; not the runtime corpus or a new SFT set. |
| `Dataset/` | Protected ignored historical sample packs only | Former 1,000-message RAG-SFT export removed from tracked tree. |
| `evaluation/datasets/` | Two unlabeled demonstration questions | Gold labels need human review; not training data. |
| `datasets/` | Former role SFT data removed | Research, Evidence, and History Answerer datasets are recoverable through Git history. |

See [Corpus V1 Colab workflow](docs/CORPUS_V1_COLAB.md), [legacy cleanup](docs/LEGACY_DATA_CLEANUP.md), [repository audit](docs/REPO_AUDIT.md), and [gold-set guidance](docs/GOLD_EVALUATION_DATASET.md).

## Run locally

Create an environment and install the runtime and frontend dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
npm install
npm --prefix frontend install
Copy-Item .env.example .env
Copy-Item frontend/.env.example frontend/.env
```

Keep credentials only in local environment files or deployment secrets; `.env` files are Git-ignored.

Set `APP_MODE` in `.env`:

- `api-only` serves lightweight API and conversation features without loading the corpus or Qwen.
- `retrieval-only` loads the preserved corpus and indexes without loading Qwen.
- `full` enables both inference modes and requires the retrieval artifacts and model files/cache.

The deployment artifact root defaults to `artifacts/vn_history_deployment`. Large corpus, index, and model files are not supplied by Git; use an existing project artifact copy. Keep `HYBRID_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507` and `CENTRAL_MODEL_ID=Qwen/Qwen3-8B` for comparable baseline measurements. The app reports missing required artifacts instead of silently rebuilding indexes or switching models.
Set `DEVICE=cuda` in `.env` when running full inference on a CUDA host.

Start the local API and frontend in separate terminals:

```powershell
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
npm run frontend
```

Set `VITE_API_BASE_URL=http://127.0.0.1:8000` in `frontend/.env` before starting Vite. For the Modal development backend, `npm run dev` starts the frontend and `modal serve modal_app.py` together. The stream endpoint is `POST /api/v1/chat/stream`; `answer_delta` carries actual generated text, while status events may arrive earlier.

## Preserve and audit the corpus

The historical data and index bytes were inventoried in [`docs/corpus_preservation_manifest.json`](docs/corpus_preservation_manifest.json). The five removed SFT JSONL paths retain recorded hashes as `retired_legacy` entries. Protected V0 files, `training/Dataset/`, and `training/InvestigatingDataset.zip` remain unchanged. V1 writes only to a separate root.

Check the preservation snapshot without modifying data:

```powershell
python -m scripts.corpus.audit_corpus
```

The command prints a JSON report and returns a nonzero status if a recorded file is missing or changed. See [the corpus preservation guide](docs/CORPUS_PRESERVATION.md) for diagnostics and interpretation.

For descriptive quality diagnostics, write a separate report without modifying the corpus:

```powershell
python -m scripts.corpus.audit_corpus --corpus artifacts/vn_history_deployment/corpus/vn_history_rag_chunks_enriched.jsonl --output reports/corpus/audit.json
```

## Measure the baseline

Run the API in `APP_MODE=full` with the required artifacts and models available. The benchmark makes real HTTP requests to its SSE endpoint. Use a fresh output directory for each run:

```powershell
python -m benchmarks.latency.runner --mode hybrid --base-url http://127.0.0.1:8000 --dataset evaluation/datasets/fixtures/questions.jsonl --warmup 2 --runs 5 --concurrency 1 --output reports/baseline/hybrid-run-001
python -m benchmarks.latency.runner --mode central --base-url http://127.0.0.1:8000 --dataset evaluation/datasets/fixtures/questions.jsonl --warmup 2 --runs 5 --concurrency 1 --output reports/baseline/central-run-001
```

The runner writes raw JSONL, run metadata, JSON summary, and Markdown summary. Use `--cold-start` for a separate cold request, and repeat runs with `--concurrency 2` or `4` as separate experiments. [Latency definitions and methodology](docs/BASELINE_BENCHMARK.md) distinguish first response/status from first answer token and keep unavailable server measurements null.

Score saved answers with the evaluation runner:

```powershell
python -m evaluation.runner --dataset evaluation/datasets/fixtures/questions.jsonl --predictions reports/baseline/hybrid-run-001/latency_records.jsonl --output reports/evaluation/hybrid-run-001
```

The included fixture questions are unlabeled and do not establish historical correctness. Add reviewed gold answers, relevant chunk/source IDs, or citation labels to obtain the corresponding metrics; missing labels are reported as N/A. [Evaluation metrics](docs/EVALUATION.md) and [experiment template](docs/EXPERIMENTS.md) describe reproducible comparisons.

## Verify changes

```powershell
.venv\Scripts\python -m pytest -q
npm --prefix frontend test
npm run frontend:lint
npm run frontend:build
```

The unit and fake-stream tests do not download Qwen weights. Live model quality and latency require the intended GPU host, model versions, and deployment artifacts.
