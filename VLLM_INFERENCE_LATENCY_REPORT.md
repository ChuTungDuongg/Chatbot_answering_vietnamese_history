# vLLM inference and latency implementation report

## Status and starting point

**IMPLEMENTED:** optional inference backend, production API integration, frozen
workload, HTTP/SSE experiment matrix, provenance, comparison reports and verified
mounted-Drive transfers. **VERIFIED LOCALLY:** Python tests and frontend checks.
**REQUIRES GPU/MODAL RUN:** successful vLLM L4 HTTP smoke, Central GPU parity and
the full 100-question load matrix. Browser/UI smoke was explicitly waived by the
user near the end of this task because they needed to shut down their computer.
No browser chat success or performance improvement is claimed.

Fetched `origin/main` before editing. It remained
`0500caec90cfa3a7cdd61c207d797e696a471d51`. Local HEAD was already
`e8c71dd896520136775ae89bd1e9a123ebf82e65`, containing the earlier Central fixes;
those changes were preserved. Work is on `codex/vllm-latency-experiments`.
Deployment Git SHA identifies that local starting commit; a separate normalized
source SHA256 identifies the actual uncommitted application/dependency code.

## Files changed and created

Changed:

- `.env.example`, `Dockerfile`, `requirements.txt`, `modal_app.py`.
- `app/config.py`, `app/main.py`, `app/models/base.py`, `app/models/qwen.py`,
  `app/api/routes.py`, `app/services/metadata.py`.
- `benchmarks/latency/runner.py`, `sse_client.py`, `metrics.py`, `report.py`.
- `tools/drive_cli.py`, `docs/BASELINE_BENCHMARK.md`.
- Tests: `test_model_variants.py`, `test_mcp_integration.py`,
  `test_repository_hardening.py` (updated factory/dependency-include fixtures).

Created:

- `app/models/factory.py`, `app/models/vllm.py`.
- `Dockerfile.vllm`, `requirements-runtime.txt`, `requirements-vllm.txt`.
- `scripts/deploy_inference_benchmark.py`, `scripts/inference_http_smoke.py`.
- `benchmarks/latency/freeze.py`, `benchmarks/latency/compare.py`.
- `evaluation/datasets/latency_v1/questions_100.jsonl`, its `manifest.json`.
- Three `configs/benchmarks/vllm_latency_*_v1.json` manifests: Hybrid,
  controlled Central, realistic Central.
- `tests/test_inference_backends.py`, `tests/test_latency_experiments.py`.
- `docs/VLLM_LATENCY_EXPERIMENTS.md`, this report.

The ignored existing `frontend/.env` now points at the isolated vLLM endpoint.
No frontend source/protocol implementation changed. Generated logs/results are
in ignored `reports/`, not committed benchmark result trees.

## Inference abstraction and semantics

`build_model_runtime(settings=..., model_id=...)` is the sole application backend
selection point. Default is Transformers; invalid values are rejected.
`ModelRuntime` defines identities, generation/adapter/engine metadata, `is_loaded`,
async lifecycle, stream/generate and cancellation. Callers no longer inspect
`.model is not None` for readiness. Hybrid and Central continue consuming
ModelDelta/ModelDone through the existing routes.

Transformers preserves its model loader, PEFT attachment, `transformers.generate`,
TextIteratorStreamer, generation lock, sampling, EOS and token timing behavior.
The Qwen chat template helper is shared by both engines, including Central tool
schemas and thinking flag. HermesFunctionCallCodec, domain routing, planner early
exit, grounding, citations and request-local MCP/backend steering remain intact.

vLLM uses V1 AsyncLLM/AsyncEngineArgs, asynchronous DELTA output and actual token
IDs/finish reason. Cancellation aborts the unique engine request; loading and
tokenization run off the FastAPI event loop. Greedy decoding uses temperature zero;
logical settings remain `do_sample=false`, `enable_thinking=false`, original dtype
and output budget. Quantization and LoRA are disabled. Model IDs remain exactly
Qwen3-4B-Instruct-2507 and Qwen3-8B. No retrieval, training, corpus/index, prompt,
model-weight, adapter or frozen-evaluation-split changes were made.

SFT/LoRA remains supported by Transformers. It is **unsupported/unverified in this
vLLM implementation** and fails explicitly rather than converting adapters.

## Version and deployment compatibility

Pinned vLLM: **0.23.0**. Its supported V1 API and CUDA requirements were inspected
in [the pinned source](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/engine/async_llm.py)
and [CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/cuda.txt).
The installed wheel exposed additional FastAPI <0.137 and NumPy <2.4 conflicts
during actual image builds. The optional bundle therefore uses Torch 2.11.0,
FastAPI 0.136.3, NumPy 2.3.5 and existing Transformers 4.57.6. Newer vLLM's
Transformers 5.x requirement was not imposed on this application.

Default Transformers dependencies remain Torch 2.13.0, FastAPI 0.141.1,
NumPy 2.4.6 and Transformers 4.57.6. Both **experiment** reference engines use
the identical optional image to avoid software-stack confounding. The default
backend is unchanged. Modal GPU defaults and benchmark helper defaults were
changed to **L4 at the user's request**. Production deployment was not overwritten.

Only one model/runtime is enabled per benchmark app. vLLM rejects enabling both
models together. Benchmark apps allow eight concurrent HTTP inputs in one
container; Transformers still serializes model generation internally. A later
production architecture can use independent Hybrid/Central inference services.

## L4 memory evidence and explicit context capacity

The first L4 vLLM attempt with native Hybrid context **failed**: 262144 tokens
required 36.0 GiB KV cache, versus 8.1 GiB available. The engine estimated a maximum
length around 58960. This is recorded in
`reports/vllm_server_l4_native_context_failure.log`.

The isolated Hybrid vLLM app was redeployed with **max_model_len=32768**,
gpu_memory_utilization=0.75, max_num_seqs=8, prefix caching=false, eager=false.
This is a disclosed deployment-capacity restriction, not prompt truncation.
Over-capacity requests fail; final output still allows 1536 tokens. Transformers
retains native input capacity, so parity requires workload prompts fitting the
vLLM cap. The Central L4 preset proposes 16384 and utilization 0.9 and still
requires its own real memory/behavior check. No Central GPU success is claimed.

## Frozen latency dataset

Source: existing frozen `evaluation/datasets/v1_silver/splits/v1_seed42/test.jsonl`
(300 questions). Source SHA256:
`f0a32500ef83e42f7d8748e02e41752c61fc4491afdb248dbe65c3198ec0b2a5`.
Seed: **2026**. Joint strata cover category, difficulty, question type,
answerability and in-domain flag. Rare strata are preserved where possible,
remaining quotas use proportional deficit and within-stratum SHA256 ranking.
The manifest preserves selected IDs, policy, source/split hashes, distributions,
count and resulting hash. Original Silver data/splits are untouched.

Result: **100 questions**, SHA256
`9536de94eb02978e5728d9c900cab01345faa3cd4ea94661fa1d918684a53fb3`.
Coverage: 21 categories; 36 easy/43 medium/21 hard; one unanswerable and two
false-premise cases. All are in-domain. These are auto-reviewed Silver labels;
absent out-of-domain examples were not invented. Deterministic bytes/IDs/hash
are tested and `freeze --verify` passes.

## Experiment framework and metrics

Presets compare Transformers reference → semantics-matched vLLM reference →
vLLM prefix caching, independently at **C1/C2/C4/C8**. Scheduler tuning is added
with explicit variant IDs/expected settings and one changed parameter at a time.
Controlled Central restricts request-local tools to history retrieval and disables
MCP; realistic Central retains normal production routing. Populations never merge.

All measured work uses production HTTP/SSE, with cold/warmup/warm and raw failures
preserved. Preflight checks dataset/model/revision/adapter, corpus/index,
generation/output ceiling, retrieval, hardware/software, source identity, mode,
population, concurrency and deployment topology. Invalid comparisons are rejected
or flagged and deltas suppressed. Every run is self-describing in run_metadata.json.

Run outputs: run_metadata.json, latency_records.jsonl, latency_summary.json/.md.
Experiment outputs: experiment_manifest.json, comparison.json/.md. Metrics include
answer/model TTFT, E2E, TPOT p50/p95/p99, generation/decode tokens/s, aggregate
successful requests/s and output tokens/s, success/error rate, output tokens and
Central planner/tool/retrieval spans. Differences are signed candidate − reference;
percentages explicitly mean latency reduction or throughput increase.

Model timing uses server monotonic clocks: vLLM engine-core scheduled/first/last
token times where available, with an explicitly labeled receipt-clock fallback.
Planning telemetry and termination reason remain reusable across both modes.
Strict guardrails require correct model, nonempty answer, successful SSE done,
ModelDone metrics and valid source/citation structures. Existing offline evaluation
adds secondary Silver quality reports; output-length shifts >20% flag review.

## Commands, Drive and Colab

Deployment commands (repository root; activate `.conda` on Windows):

```bash
python -m scripts.deploy_inference_benchmark --backend transformers --mode hybrid --name vn-history-bench-tf-hybrid-smoke
python -m scripts.deploy_inference_benchmark --backend vllm --mode hybrid --name vn-history-bench-vllm-hybrid-smoke --max-model-len 32768
python -m scripts.deploy_inference_benchmark --backend transformers --mode central --name vn-history-bench-tf-central
python -m scripts.deploy_inference_benchmark --backend vllm --mode central --name vn-history-bench-vllm-central --max-model-len 16384 --gpu-memory-utilization 0.9
```

The first two were actually deployed on L4. Central commands are provided for
manual integration. Deployment names isolate experiments from production.
Use URLs printed by Modal and set preset URL environment variables, then:

```bash
python -m benchmarks.latency.compare --experiment configs/benchmarks/vllm_latency_hybrid_v1.json --output reports/latency/hybrid-v1
python -m benchmarks.latency.freeze --verify
```

Drive uses existing SHA256-verified mounted-directory copies. New commands are
push/pull/verify-benchmark-datasets (also experiment configs) and
push/pull/verify-benchmark-results with --local-path/--name. The layout is
benchmarks/datasets, benchmarks/experiments, benchmarks/results. Secret-like files
and path/symlink escapes are refused. No Drive API/OAuth client was added.
In Colab mount Drive, clone this revision, install pydantic/jsonschema, pull/verify
assets, set endpoint URLs and invoke the same CLI. Any networked Python machine
can run it; Colab/GPU on the benchmark client is optional. Full instructions are
in docs/VLLM_LATENCY_EXPERIMENTS.md.

## Validation actually executed

PowerShell commands used the existing `.conda/python.exe`; PYTHONCASEOK=1 handles
the existing Windows Training/training import casing.

| Command | Actual result |
|---|---|
| `python -m benchmarks.latency.freeze` | 100 questions created, expected hash |
| `python -m benchmarks.latency.freeze --verify` | Passed |
| `pytest -q tests/test_inference_backends.py tests/test_model_variants.py tests/test_mcp_integration.py tests/test_baseline_benchmark_evaluation.py tests/test_generation_telemetry.py` | 83 passed |
| `pytest -q tests/test_latency_experiments.py` | 13 passed |
| `pytest -q` | 366 passed in 66.38 seconds |
| `pytest -q tests/test_latency_experiments.py tests/test_inference_backends.py tests/test_repository_hardening.py` | 25 passed after follow-up changes |
| `python -m compileall -q app scripts evaluation benchmarks tools` | Passed |
| `git diff --check` | Passed; only Windows line-ending notices |
| `npm --prefix frontend test` | Node tests passed; 11 Vitest files/75 tests passed |
| `npm run frontend:lint` | Passed |
| `npm run frontend:build` | Passed |

The HTTP fixture tests execute an actual localhost SSE matrix at C1/C2, run the
legacy benchmark CLI, verify comparison outputs/quality artifacts and test Drive
paths/SHA checks. They stub engine dependencies and require no GPU/model download.
An initial full-suite failure in the dependency-location test was fixed to follow
the requirements include; the full suite was rerun and passed.

## Actual Modal/API verification and measurements

Transformers isolated L4: `/health`, `/ready`, `/api/v1/baseline/metadata?mode=hybrid`
all passed. Metadata reports Transformers and **NVIDIA L4**. Real production
`/api/v1/chat/stream` returned **508 incremental answer_delta events**, nonempty
grounded answer, valid source/citation structures and completion metrics:

| Single smoke observation | Value |
|---|---:|
| Input/output tokens | 2226 / 1007 |
| max_new_tokens | 1536 |
| finish_reason / hit limit / truncated | stop / false / false |
| Model TTFT | 1655.508 ms |
| Client answer TTFT | 5793.746 ms |
| Client E2E | 65176.632 ms |
| Server E2E | 63664.751 ms |

Question: “Trình bày nguyên nhân, diễn biến chính và ý nghĩa của chiến thắng
Bạch Đằng năm 938.” This is **one integration observation, not a representative
100-question benchmark or a Transformers-vLLM comparison**. Raw evidence is in
`reports/transformers_smoke_l4/`.

vLLM image/deployment succeeded. Native-context startup failed as documented above.
The 32K redeployment currently reports **waiting for a GPU_L4 worker/capacity**;
successful vLLM health/readiness/SSE has not yet been observed. Raw deployment and
server logs are retained under reports/. Exact manual verification command:

```bash
python -m scripts.inference_http_smoke --base-url https://tungduong156gli--vn-history-bench-vllm-hybrid-smoke-fastapi-app.modal.run --expected-backend vllm --output reports/vllm-smoke-next --timeout 900
```

The same command with `--expected-backend transformers` verifies the reference.
No full GPU benchmark, prefix-cache experiment or Central GPU run was completed.
Browser/UI test was waived by the user's later instruction; frontend configuration
was pointed at vLLM, but no browser chat result is claimed.

## Next experiment and limitations

First complete the 32K L4 vLLM production-API smoke when Modal capacity is available.
Then run the frozen Hybrid matrix at all four loads before choosing an engine or
prefix-cache/scheduler configuration. Native context does not fit L4; request
capacity limits must remain explicit. Central L4 memory/tool parity, vLLM LoRA
support and semantic output review remain independent checks. CPU retrieval/tool
time and engine queue time may dominate user latency even if decode accelerates.
There is currently **no measured evidence supporting a claimed vLLM speedup**.
