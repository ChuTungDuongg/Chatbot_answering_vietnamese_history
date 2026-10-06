# Optional vLLM inference and latency experiments

The production path remains: existing frontend → FastAPI
`POST /api/v1/chat/stream` → HybridRuntime/CentralRuntime → configured model →
`ModelDelta`/`ModelDone` → existing SSE `answer_delta`/`done` → frontend rendering.
There is no benchmark-only generation endpoint or new frontend implementation.

## Architecture and backend selection

`app/models/factory.py:build_model_runtime` selects one engine with
`INFERENCE_BACKEND=transformers` (default) or `INFERENCE_BACKEND=vllm`.
The backend-neutral protocol exposes `is_loaded`, async loading/shutdown,
stream/generate, model/revision/adapter identity, generation and engine metadata.
Transformers keeps `generate`, TextIteratorStreamer and its generation lock.
vLLM uses V1 AsyncLLM's async generator and aborts cancelled requests. GPU startup
and tokenization stay outside FastAPI's event loop.

Both engines use the same Qwen tokenizer chat template, messages, tool schemas,
and `enable_thinking`; Central retains HermesFunctionCallCodec and its planner/tool
policy. Greedy generation maps to vLLM temperature zero. Logical generation
settings remain unchanged in metadata. `generation_config="vllm"` disables unrelated
Hugging Face generation defaults, and the existing tokenizer EOS is supplied.
There is no quantization. The IDs remain `Qwen/Qwen3-4B-Instruct-2507` (Hybrid)
and `Qwen/Qwen3-8B` (Central).

**vLLM SFT/LoRA parity is not verified and adapters are explicitly rejected.**
Continue using Transformers for SFT. There is no conversion or retraining.

## Dependency compatibility

Default `requirements.txt` retains Torch 2.13.0, Transformers 4.57.6,
FastAPI 0.141.1 and NumPy 2.4.6. Unchanged common pins are included from
`requirements-runtime.txt`. The separate Linux CUDA `requirements-vllm.txt`
pins vLLM 0.23.0, Torch 2.11.0, FastAPI 0.136.3, NumPy 2.3.5 and the same
Transformers 4.57.6. Real Modal builds exposed the published wheel's FastAPI
<0.137 constraint and mistral-common's NumPy <2.4 constraint on Python 3.11.
These are resolved in the optional bundle, without forcing installation or
upgrading the production baseline. Newer vLLM requires Transformers 5.x.
See [pinned CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/cuda.txt)
and [pinned V1 async API](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/engine/async_llm.py).

For direct engine comparisons, **both reference deployments use Dockerfile.vllm
and the exact same dependency bundle**. A production Transformers 2.13.0 result
is a separate population. Metadata records actual installed versions.

## Modal deployments

vLLM reserves model/KV memory. Settings reject full vLLM deployments enabling
both models. Benchmark Hybrid and Central independently, and isolate Transformers
likewise. This is an experiment topology. Simultaneous production use should
later use independent inference services/endpoints. A later UI engine selector
should route to separate deployments rather than co-load engines.

From the repository root, with existing Modal credentials and artifact/HF volumes:

```bash
python -m scripts.deploy_inference_benchmark --backend transformers --mode hybrid --name vn-history-bench-tf-hybrid
python -m scripts.deploy_inference_benchmark --backend vllm --mode hybrid --name vn-history-bench-vllm-hybrid --max-model-len 32768
python -m scripts.deploy_inference_benchmark --backend transformers --mode central --name vn-history-bench-tf-central
python -m scripts.deploy_inference_benchmark --backend vllm --mode central --name vn-history-bench-vllm-central --max-model-len 16384 --gpu-memory-utilization 0.9
python -m scripts.deploy_inference_benchmark --backend vllm --mode hybrid --name vn-history-bench-vllm-hybrid-prefix --max-model-len 32768 --prefix-caching
```

On Windows activate `.conda` first. Use URLs printed by Modal. The helper selects
identical L4, optional image, vanilla model, eager loading, unchanged FAISS
retrieval and container HTTP capacity 8; only its child environment is changed.
It refuses the production app name. `--serve` creates an ephemeral dev endpoint.
`MODAL_APP_NAME`, `MODAL_RUNTIME_IMAGE`, `MODAL_GPU_CLASS`, `MODAL_MAX_INPUTS` can
also configure `modal deploy modal_app.py` directly. Default deployment behavior
remains Transformers with the original app name and input capacity.

Hybrid L4 reference prefix caching is explicitly false, GPU utilization 0.75, max sequences
8 and context capacity 32768. The native 262144 context failed on L4: it needed
36 GiB KV memory versus 8.1 GiB available. This disclosed deployment-capacity
restriction does not truncate prompts or change output tokens; over-capacity
requests fail explicitly. Transformers retains its native context capacity, so
comparison parity applies only to prompts that fit the vLLM deployment.
Central L4 presets use 16384 and utilization 0.9 and require a separate GPU
memory/parity check before accepting results. Engine-native context length applies unless `--max-model-len` is specified;
changing context capacity must be named/documented as a separate experiment.
`--max-num-seqs`, `--gpu-memory-utilization`, `--prefix-caching` and
`--enforce-eager` expose supported tuning variants. Readiness/metadata report the
effective initialized configuration; eager mode is an explicit debug experiment.

## Frozen 100-question workload

`evaluation/datasets/latency_v1/questions_100.jsonl` derives from the existing
frozen Silver `v1_seed42/test.jsonl` (300 questions), not a training split.
Seed: 2026. Source SHA256:
`f0a32500ef83e42f7d8748e02e41752c61fc4491afdb248dbe65c3198ec0b2a5`.
Result SHA256:
`9536de94eb02978e5728d9c900cab01345faa3cd4ea94661fa1d918684a53fb3`.

Selection validates IDs/schema, prefers accepted/reviewed pools when available,
stratifies joint category/difficulty/question_type/answerability/domain metadata,
preserves rare strata where possible, then fills proportional-deficit quotas.
SHA256 ranking fixes within-stratum choices; final order is by ID. Original
labels are retained. The manifest records source/split hashes, seed, policy,
selected IDs, distributions, count and output hash.

These are auto-reviewed Silver labels, not human gold. Coverage is 21 categories,
36 easy/43 medium/21 hard, one unanswerable and two false-premise questions.
All are in-domain: missing out-of-domain cases are not fabricated.

```bash
python -m benchmarks.latency.freeze --verify
```

A new source/seed belongs to a new version. Different frozen bytes cannot be
overwritten. A unit test verifies exact deterministic IDs/hash.

## HTTP/SSE runner and matrix

Any Python/network machine can run the existing CLI; no GPU/model download is
required on the benchmark client:

```bash
python -m benchmarks.latency.runner --base-url https://YOUR-ENDPOINT.modal.run --mode hybrid --dataset evaluation/datasets/latency_v1/questions_100.jsonl --output reports/latency/hybrid-c1 --warmup 2 --runs 1 --concurrency 1 --timeout 900 --strict-contract
```

Set deployed URLs for the matrix (PowerShell example):

```powershell
$env:TF_HYBRID_URL = "https://TRANSFORMERS-ENDPOINT.modal.run"
$env:VLLM_HYBRID_URL = "https://VLLM-ENDPOINT.modal.run"
$env:VLLM_HYBRID_PREFIX_URL = "https://PREFIX-ENDPOINT.modal.run"
python -m benchmarks.latency.compare --experiment configs/benchmarks/vllm_latency_hybrid_v1.json --output reports/latency/hybrid-v1
```

Presets run C1/C2/C4/C8 separately and compare Transformers reference → vLLM
reference → vLLM prefix caching. First establish reference parity. For scheduler
tuning, copy a manifest under a new experiment ID and add a unique variant ID,
URL and expected effective settings; change one parameter at a time. Keep model,
prompts, retrieval, output budget, dataset, hardware and warmup fixed.

Central has separate populations and outputs:

```bash
python -m benchmarks.latency.compare --experiment configs/benchmarks/vllm_latency_central_controlled_v1.json --output reports/latency/central-controlled-v1
python -m benchmarks.latency.compare --experiment configs/benchmarks/vllm_latency_central_realistic_v1.json --output reports/latency/central-realistic-v1
```

URL environment names appear in each JSON. Controlled Central uses request-local
`allowed_tools=["search_history"]`, MCP disabled and FAISS. Realistic Central
uses normal production routing/tools. Never pool these populations. Controlled
requests still run through the real planner/tools/API.

## Provenance, guardrails and interpretation

Preflight rejects differing/unknown critical dataset/model revisions,
model/adapter identity, corpus/index hashes, generation settings/output ceiling,
retrieval configuration, source fingerprint, hardware/software class, topology,
mode or concurrency. Expected backend/effective settings are verified. Raw SSE
done identity is checked against run metadata. Tool policy is captured for Central.

Runs keep `run_metadata.json`, `latency_records.jsonl`, `latency_summary.json`,
`latency_summary.md`, including failed requests and partial answers. Experiments
add `experiment_manifest.json`, `comparison.json`, `comparison.md`. Invalid
comparisons retain observations and suppress comparison deltas. Store large
outputs under ignored `reports/`.

Strict checks require successful HTTP/SSE completion, nonempty answer, ModelDone
metrics, correct model, and valid source/citation structures. Existing offline
evaluation creates a secondary quality report with Silver labels and no extra
model calls. It does not prove semantic parity. Median output-length differences
over 20% flag review; increased errors invalidate acceptance by default. Numerical
backend differences do not require identical answer strings.

Tables per mode/concurrency show answer TTFT, model TTFT, E2E, TPOT p50/p95/p99,
decode tokens/s, success rate, aggregate requests/s and output tokens/s. Signed
absolute differences use candidate − reference. Latency reduction percentage
uses (reference − candidate)/reference; throughput increase uses
(candidate − reference)/reference. Phase throughput includes conversation creation
and client scheduling. Cold/warmup/warm remain separate. C1 measures user latency;
C2/C4/C8 measure load behavior and failures.

Central also exposes per-round/aggregate planner model time, planner wall time,
tools, retrieval and final generation. vLLM ModelDone uses engine-core monotonic
scheduled/first/last token timestamps when available; its monotonic receipt fallback
is explicitly labeled. Queue time remains in answer TTFT/E2E. Transformers model
timing starts after its lock; queue delay likewise remains in answer TTFT. SSE
chunk gaps are not token intervals. See BASELINE_BENCHMARK.md for definitions.

## Existing frontend integration smoke

Verify `/health`, `/ready`, `/api/v1/baseline/metadata?mode=hybrid`, then an actual
streamed chat. Assert backend vLLM, multiple incremental answer_delta events and
ModelDone. Set `VITE_API_BASE_URL=https://YOUR-VLLM-ENDPOINT.modal.run` in the
existing ignored `frontend/.env`; start/restart `npm --prefix frontend run dev`.
Open localhost:5173, choose Hybrid, submit a history question and verify streaming
rendering/completion. Actual evidence is recorded in
`VLLM_INFERENCE_LATENCY_REPORT.md`. No SSE protocol change is needed.

## Google Drive and Colab

Reuse `tools/drive_cli.py` with mounted Drive, no duplicate API/OAuth client:

```bash
python -m tools.drive_cli init --drive-root /content/drive/MyDrive/vn_history
python -m tools.drive_cli push-benchmark-datasets --drive-root /content/drive/MyDrive/vn_history
python -m tools.drive_cli verify-benchmark-datasets --drive-root /content/drive/MyDrive/vn_history
python -m tools.drive_cli push-benchmark-results --drive-root /content/drive/MyDrive/vn_history --local-path reports/latency/hybrid-v1 --name hybrid-v1
python -m tools.drive_cli verify-benchmark-results --drive-root /content/drive/MyDrive/vn_history --local-path reports/latency/hybrid-v1 --name hybrid-v1
```

Matching pull commands are `pull-benchmark-datasets` and
`pull-benchmark-results`. Drive layout is `benchmarks/datasets/`,
`benchmarks/experiments/`, `benchmarks/results/`. Copies verify SHA256 and refuse
mismatched existing files without `--overwrite`. Secret-like files, symlinks and
path escapes are rejected. Transfer benchmark assets/results only, never `.env`,
credentials or a whole checkout.

In Colab mount Drive with `google.colab.drive.mount('/content/drive')`, clone the
experiment revision, `pip install pydantic jsonschema`, pull/verify assets if
needed, assign endpoint URLs and run the same CLI. Colab is optional; no GPU is
needed on the client. GPU integration requires Linux CUDA, credentials and
unchanged artifact/HF volumes. Normal unit tests stub vLLM without downloads.
Central GPU and LoRA parity require independent verification. No conclusion
should be drawn from a single-question smoke or C1 alone.
