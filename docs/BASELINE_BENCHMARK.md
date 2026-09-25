# Baseline HTTP SSE benchmark

The benchmark calls the real `POST /api/v1/chat/stream` endpoint. It creates a fresh conversation before each timed request, sends `X-Client-ID`, consumes actual SSE frames, and uses `time.perf_counter_ns()` for client observations. Conversation creation is outside the timed interval. The client saves **one raw JSONL row for every cold, warmup, and warm request**, including failures. It never calls a Python generation function.

## Run it

Start the application with the intended model and retrieval artifacts already available. The first command measures one cold request before warming the process, then two warmups, then five measured repeats of **each** dataset question:

```bash
python -m benchmarks.latency.runner --mode hybrid --dataset evaluation/datasets/fixtures/questions.jsonl --output reports/baseline/hybrid-c1 --cold-start --warmup 2 --runs 5 --concurrency 1
```

Measure Central separately:

```bash
python -m benchmarks.latency.runner --mode central --dataset evaluation/datasets/fixtures/questions.jsonl --output reports/baseline/central-c1 --cold-start --warmup 2 --runs 5 --concurrency 1
```

Repeat warm experiments at `--concurrency 1`, `2`, and `4`, each with a new `--output` directory. A single process has at most one first cold request; the client requires concurrency 1 when `--cold-start` is set. Restart the server between **independent cold-start trials** and do not claim a cold trial if weights or indexes were already loaded. `--base-url` defaults to `http://127.0.0.1:8000`; `--timeout` defaults to 300 seconds.

Before requests, the client fetches `GET /api/v1/baseline/metadata?mode=hybrid` (or `mode=central`) to capture server Git SHA, model/config IDs, artifact hashes, hardware, and software versions where the server can observe them. The optional `--metadata-file path.json` fills any server-null facts from independently verified records. A conflicting non-null value fails the run. For example:

```json
{
  "server_hardware": {"gpu_model": null, "gpu_memory_bytes": null, "cuda_version": null, "cpu": null, "ram_bytes": null, "os": null},
  "corpus_hash": null,
  "retrieval_index_hash": null,
  "model_id": "Qwen/Qwen3-4B-Instruct-2507",
  "model_revision": null,
  "generation_settings": {"do_sample": false, "enable_thinking": false},
  "retrieval_settings": {}
}
```

The stream's `done` event supplies model ID/revision/settings and server metrics when available. The runner checks these against the pre-run metadata. Unknown values remain JSON `null`; it does not treat client hardware as server hardware. `run_metadata.json` also records UTC timestamp, **server** Git commit, separate client Git commit, dataset hash, client OS/CPU/RAM/Python and installed Torch/Transformers versions, mode, concurrency, and run parameters. Use a corpus audit manifest or deployment record to fill hashes the server cannot calculate. Never put credentials in the metadata file.

Outputs in the chosen directory are `run_metadata.json`, `latency_records.jsonl`, `latency_summary.json`, and `latency_summary.md`. The summary groups **cold**, **warm**, and **warmup** independently, reporting request count, success/error rates, and observed count, mean, population standard deviation, p50, p95, p99 for each timing or throughput metric. Failed requests remain in the error rate; incomplete metrics do not enter numeric aggregates.

## Timing definitions

All durations are milliseconds. The client measures from a monotonic `request_start` immediately before it opens the streamed chat HTTP request. Server spans use a server monotonic clock and are sent in `done.metrics`.

| Field | Definition and observer |
|---|---|
| `ttfb_ms` | Client request start to receipt of the HTTP response headers from the streamed endpoint. This marks response start, before parsing the first SSE event. |
| `first_status_event_ms` | Client request start to first `status` event. It is not a token latency. |
| `retrieval_ms` | Server retrieval start to retrieval finish. Null when no retrieval span was recorded. |
| `prompt_build_ms` | Server final context ready to generation request ready. |
| `generation_start_ms` | Server-relative point at which final visible generation starts. It is a timestamp offset, not a duration. |
| `model_ttft_ms` | Server generation start to first model-produced final-answer token, only when observed in the model stream. |
| `answer_ttft_ms` | Client request start to first nonempty `answer_delta` SSE event. Central includes planning and tool time before its final answer. |
| `generation_ms` | Server generation start to final generated token. |
| `e2e_ms` | Client request start to `done` event receipt. |
| `input_tokens`, `output_tokens` | Server tokenizer counts; never estimated from SSE chunks. |
| `tokens_per_second` | Server output tokens divided by full generation seconds, including prefill/TTFT, if supplied by the server. |
| `decode_tokens_per_second` | Prefer `(output_tokens - 1) / (generation_ms - model_ttft_ms)` when server has the relevant token boundary. |
| `tpot_ms` | Server post-first-token decode duration divided by `output_tokens - 1`, if at least two tokens were produced. |
| `inter_token_latency_ms` | Client gaps between consecutive nonempty `answer_delta` **chunks**. SSE chunks need not equal tokenizer tokens. |
| `itl_p50_ms`, `itl_p95_ms`, `itl_p99_ms` | Percentiles of those per-request chunk gaps. Null with fewer than two deltas. |

The raw row includes the assembled answer, sources, server request ID when supplied, HTTP status, phase, run index, failure detail, all client timings and all available server metrics. HTTP response start and answer timing reflect network and transport buffering. Server model TTFT is a separate measurement. No benchmark code estimates unreported retrieval, prompt, token, or model timing.

Where the server also reports first-status, E2E, or ITL values, the raw row stores them as `server_first_status_event_ms`, `server_e2e_ms`, and `server_itl_pXX_ms`, leaving the unprefixed fields as client observations.

Central rows can additionally carry `model_calls`, `tool_calls`, `tool_call_types`, `tool_execution_ms`, `tool_parse_failures`, `action_rounds`, `time_until_final_generation_ms`, and `final_answer_ttft_ms` from server telemetry. Missing Central observations remain null.

## Comparison discipline

Keep dataset hash, corpus/index hashes, exact model IDs and revisions, generation settings (`do_sample=false`, `enable_thinking=false` for the baseline), retrieval settings, hardware, server restart procedure, and load level fixed when comparing changes. Report cold and warm separately. Compare like-for-like concurrency; never mix concurrency 1 with concurrency 4 in a single headline. The included question fixture has no quality labels and is only a pipeline smoke test. Do not report its timing as representative of all Vietnamese history questions.
