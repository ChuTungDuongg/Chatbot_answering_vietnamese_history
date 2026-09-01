# Modal Performance Audit

Date: 2026-08-28

Deployment tested: `qwen3-3f5f81301b0ca4ac`

## Runtime Placement

- GPU: NVIDIA L4
- Total VRAM: 22.03 GiB
- Torch: `2.13.0+cu130`
- CUDA: `13.0`
- `hf_device_map` summary: `{"0": 1}`
- Qwen input embedding device: `cuda:0`
- Embedder device: `cuda`
- Reranker device: `cuda:0`
- CPU offload: NO
- Disk offload: NO
- Peak allocated VRAM observed: 7.95 GiB
- Reserved VRAM observed: 8.40 GiB

## Cold Start

Measured by a former temporary L4 diagnostic function using the same Volume and shared Qwen runtime (historical result; production now targets A100).

Request-only run:

- Cold start total before warm request: 54,518.59 ms
- Artifact path validation: 3.71 ms
- Config load: 1.67 ms
- Artifact lock validation: 5,931.36 ms
- Corpus load: 7,010.42 ms
- FAISS load: 116.37 ms
- BM25 load: 105.37 ms
- Embedder load: 4,899.98 ms
- Reranker load: 3,438.11 ms
- Approximate Qwen base + adapter load: 33,011.62 ms

Benchmark run cold start total: 70,711.62 ms.

## L4 Microbenchmark

One warm-up generation, then three measured generations per adapter. Same shared base, same adapters, same production token budgets.

| Adapter | Input tokens | Output tokens | Median generation ms | Tokens/sec | Peak VRAM GiB |
|---|---:|---:|---:|---:|---:|
| research | 52 | 361 | 56,590.71 | 6.38 | 6.34 |
| evidence | 1,820 | 24 | 4,340.22 | 5.53 | 6.79 |
| history | 682 | 75 | 11,853.18 | 6.33 | 6.49 |

Interpretation: a single generation is not CPU-offloaded, but L4 throughput is low, roughly 5-6 output tokens/sec in these runs. Long outputs dominate wall time.

## Bạch Đằng Regression

Question:

`Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?`

Result:

- Status: failed
- Failure stage: evidence
- Failure code: `cross_id_claim`
- Cold start included in request metric: false
- Warm request total: 210,536.46 ms
- Telemetry total: 211,695.70 ms
- Actual `model.generate` calls: 5
- Total input tokens: 11,202
- Total output tokens: 1,172
- Average generation tokens/sec: 5.35

Research:

- Attempts: 1
- Steps: 4
- Actual LLM calls: 4
- JSON repairs: 0
- Research time: 96,065.73 ms
- Tool calls: 3
- Retrieval time inside tools: 1,756.78 ms

Per-call Research generation:

| Call | Input tokens | Output tokens | Generation ms | Tokens/sec |
|---:|---:|---:|---:|---:|
| 1 | 729 | 48 | 11,631.33 | 4.13 |
| 2 | 1,247 | 231 | 38,707.79 | 5.97 |
| 3 | 1,475 | 231 | 38,720.92 | 5.97 |
| 4 | 1,703 | 26 | 5,059.06 | 5.14 |

Retrieval:

- Search-history tool time included in request summary: 1,756.78 ms
- Ranking behavior was not changed.

Evidence:

- Attempts: 1
- Actual LLM calls: 1
- Input tokens: 6,048
- Output tokens: 636
- Generation latency: 114,366.63 ms
- Tokens/sec: 5.56
- Validation issue: `cross_id_claim`
- Deterministic recovery used: NO
- Targeted Evidence repair used: NO
- Reason recovery did not run: `cross_id_claim` is a hard issue by current contract

History:

- Called: NO
- Generation calls: 0
- History ms: 0

## Diagnosis

FINAL_DIAGNOSIS

ARTIFACTS_CANONICAL=YES

REMOTE_MATCHES_LOCAL=YES

REMOTE_WEIGHT_MATCH:
research=YES
evidence=YES
history=YES

EVIDENCE_RUNTIME_PASS=NO

L4_PRIMARY_CAUSE=PARTIAL

CPU_OFFLOAD=NO

CALL_AMPLIFICATION=MEDIUM

RETRAINING_JUSTIFIED_NOW=NO

ROOT_CAUSE_RANKING:

1. Evidence adapter produced a hard contract violation, `cross_id_claim`, after a 114.37s Evidence generation.
2. L4 per-generation throughput is low at roughly 5-6 output tokens/sec, so 1,172 output tokens across 5 calls naturally reaches multi-minute latency.
3. Research used 4 real generations and emitted 536 output tokens before Evidence, costing about 94.12s in model generation plus tool/retrieval time.
4. Evidence prompt/input was large at 6,048 tokens and output was long at 636 tokens, creating the single largest measured generation.
5. Artifact contamination is no longer the active cause: remote weights match canonical local hashes and stale managed files were removed.

NEXT_RECOMMENDED_ACTIONS:

1. Use the new telemetry on several production questions to see whether `cross_id_claim` repeats across Evidence outputs or is isolated to this case.
2. Add a targeted Evidence contract evaluation set for cross-ID attribution before considering retraining.
3. Run an A100 comparison only after collecting a few warm L4 traces; the current evidence says L4 is a latency contributor, not the sole failure cause.
