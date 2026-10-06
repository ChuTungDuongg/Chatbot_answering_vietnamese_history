# Runtime planning and generation telemetry

This runtime change starts from remote `main` commit
`0500caec90cfa3a7cdd61c207d797e696a471d51`. It does not modify retrieval,
training, models, adapters, sampling, prompt grounding or citation contracts.

## Central planning

The developer trace includes `planning.rounds`, containing `round`, `started_ms`,
`finished_ms`, `wall_ms`, `model_ms`, `ttft_ms`, `preparation_and_queue_ms`,
`input_tokens`, `output_tokens`, `finish_reason`, `tools_requested`,
`parse_failures`, `completion_declared`, and `early_exit`. No prompt, planner output or hidden reasoning
is included. Model timing/tokens come from the existing planner `ModelDone`;
missing backend metadata remains null rather than estimated.

`planning.total_model_ms`, `total_ttft_ms`, `total_wall_ms`, `input_tokens` and
`output_tokens` aggregate rounds. `performance` also includes flat
`planning_model_ms`, `planning_model_ttft_ms`, `planning_input_tokens`,
`planning_output_tokens` and `planning_wall_ms`.

`performance.pre_final_breakdown_ms` partitions time before the final model
starts: request preparation, planning (including load/tokenization/queue), tools,
prompt construction, orchestration, prepare-to-model-request and final model
preparation/queue. These buckets sum to `generation_start_ms`. Planning's per-round
model/preparation split distinguishes actual decoding from warmup and queue time.
History retrieval and dense/BM25/fusion/reranker timings keep their existing meaning.

The planner schemas add the optional boolean argument
`__central_final_after_success`. It must be true on every call in the round to
declare that successful results will cover the whole question without another
tool dependency. It is stripped before builtin/MCP validation and execution;
registry/capability schemas and request permissions remain unchanged. Complex
remote schemas or schemas already owning that name are left untouched. Optional
MCP controls are omitted if they would exceed the existing schema budget. A bounded
companion `<plan_status>{"final_after_tools":true,"pending_tools":[]}</plan_status>`
is also supported; only decoded visible planner content can provide this status.
The runtime additionally requires all calls to succeed
with nonempty source IDs/text, no truncated evidence, no truncated/malformed plan,
no skipped calls, and no unresolved reader dependency. Search snippets keep the
next planner round when a corresponding reader is allowed. Mixed-domain queries
retain planning if either evidence domain is missing; attachments retain planning.
Absent/malformed completion status preserves the original loop limit. Tool output
cannot supply this status. No new model call measures or judges sufficiency.

Before: planner → tool → planner → final, even for a complete one-tool plan.
After, when completion is explicit and checks succeed: planner → tool → final.
Other plans retain multi-call/multi-round execution. There is no fixed latency
saving: it depends on the skipped planner round and model warmup.

The instruction names the scope of `search_history` explicitly. Narrow explicit
world/outside-Vietnam cues route planner schemas to allowed external tools;
request permission/effective-tool views remain unchanged. Hallucinated local
calls cannot substitute for external evidence in these requests. Vietnamese and
ambiguous queries retain the local fallback; mixed queries may use both. Wikipedia
is one option: permitted web and MCP tools remain available. Tools outside the
request's immutable view cannot execute. When no adequate evidence is obtained,
the unchanged final grounding prompt requires the model to state the limitation.
Fetched Wikipedia/page dictionaries now enter context collection, and a fetched
page replaces the same-ID search snippet without changing source/citation IDs.

## Completion and token budget

Hybrid's only budget configuration is `settings.hybrid_max_new_tokens`:
default 768 → **1536**, shared by vanilla and SFT. Environment overrides such as
`HYBRID_MAX_NEW_TOKENS=1024` or `2048` work locally and on Modal. Modal forwards an
explicit override and otherwise leaves the default to container `Settings`.
Central remains **1536** final tokens and **256** action tokens by default.

Shared `ModelDone` reports `max_new_tokens`, `finish_reason`,
`hit_max_new_tokens`, and `truncated`. Qwen inspects actual generated sequences
after generation without buffering streamed deltas or re-tokenizing text. An
observed EOS yields `stop/false/false`, including EOS at the ceiling. Non-EOS
termination at the requested ceiling yields `length/true/true` when no competing
time/string stop is configured. Unknown/incomplete backend information stays null;
cancellation is recorded as `cancelled` with unknown length/truncation flags.
Natural EOS can occur at any shorter length; the prompt does not request longer answers.

The API persists the answer and processes sources, marks `request_finished`, then
computes metrics once. The same final debug trace is saved by an owner/conversation
scoped update and emitted over SSE/REST. `e2e_ms` includes answer persistence;
it ends at `request_finished`, excluding the subsequent trace-only database write
and network delivery. Reloading the conversation no longer restores an earlier
trace with null `e2e_ms`. Existing first status, TTFT, generation, ITL, citation and
SSE delta behavior are retained. The frontend adds a Planning trace section.

## Validation and integration

Unit regressions live in `tests/test_central_planning.py` and
`tests/test_generation_telemetry.py`. They include the exact world-history query,
Wikipedia search→fetch, external MCP without Wikipedia, local preference,
mixed/multi-tool planning, conservative exit checks, actual EOS/ceiling/unknown
termination, vanilla/SFT budgets, overrides and identical saved/emitted traces.

The optional isolated Modal smoke uses a temporary database and existing V1
Volumes/Secrets, without deployment. It measures alternating FAISS/Qdrant warm
retrieval and dense p50/p95 with unmodified retrieval parameters, and checks real
streaming/trace metadata. Qdrant unavailable is an explicit benchmark skip.

```powershell
$env:RETRIEVAL_AVAILABLE_BACKENDS = 'faiss,qdrant'
$env:MODAL_QDRANT_SECRET_NAME = 'vn-history-qdrant'
modal run --write-result reports/runtime_telemetry/vanilla.json scripts/modal_runtime_telemetry_smoke.py::runtime_telemetry_smoke --variant vanilla --repeats 10
modal run --write-result reports/runtime_telemetry/sft.json scripts/modal_runtime_telemetry_smoke.py::runtime_telemetry_smoke --variant sft --repeats 0
```

See the task's final report for commands actually executed and measurements.
