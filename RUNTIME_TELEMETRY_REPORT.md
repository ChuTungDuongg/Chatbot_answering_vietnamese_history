# Central runtime and Hybrid budget implementation report

Remote `main` was fetched and inspected at
`0500caec90cfa3a7cdd61c207d797e696a471d51`, matching the initially clean local checkout.
Changes are on local branch `codex/central-runtime-telemetry`. No commit, push or
production deployment was performed. Modal checks used ephemeral runs and a
temporary conversation database.

## Central

### Files

Modified shared/runtime files:

- `app/central/runtime.py`: domain-aware planner schemas, conservative completion,
  round metrics, dictionary evidence and fetched-page replacement.
- `app/api/routes.py`: planner metrics, pre-final latency breakdown, actual generation
  budget/termination, and completed-trace ordering.
- `app/chat/store.py`: owner/conversation-scoped final debug trace persistence.
- `app/models/base.py`, `app/models/qwen.py`: reusable actual-sequence termination metadata.
- `app/rag/schemas.py`: optional planner metrics in `PreparedAnswer`.
- `frontend/src/components/DeveloperTrace.jsx`: Planning section.
- `frontend/tests/trace.test.js`: planning/termination sanitizer regression.

Added:

- `app/central/planning.py`: routing hints, completion protocol and planner-only schemas.
- `tests/test_central_planning.py`: routing, timing, multi-tool, early exit and MCP regressions.
- `tests/test_generation_telemetry.py`: budget, metadata, sequence termination and final trace tests.
- `frontend/tests/ui/DeveloperTrace.test.jsx`: display and hidden-reasoning filtering.
- `scripts/modal_runtime_telemetry_smoke.py`: real-model checks and paired retrieval measurements.
- `docs/RUNTIME_TELEMETRY.md`: schema, semantics and smoke instructions.
- `RUNTIME_TELEMETRY_REPORT.md`: this report.

### Planner telemetry

`planning.rounds` records each existing planner generation's:

```json
{
  "round": 1,
  "started_ms": 0,
  "finished_ms": 0,
  "wall_ms": 0,
  "model_ms": 0,
  "ttft_ms": 0,
  "preparation_and_queue_ms": 0,
  "input_tokens": 0,
  "output_tokens": 0,
  "finish_reason": "stop",
  "tools_requested": ["search_history"],
  "parse_failures": 0,
  "completion_declared": true,
  "early_exit": true
}
```

The values above illustrate the shape; actual round timing and tokens come from
`ModelDone`, and unavailable backend metadata stays null. Aggregates include
`total_model_ms`, `total_wall_ms`, `total_ttft_ms`, input/output tokens, domain and
exit reason. Flat performance fields expose the requested `planning_model_ms`,
`planning_model_ttft_ms`, `planning_input_tokens`, `planning_output_tokens`.

`pre_final_breakdown_ms` partitions request preparation, planner wall time, tools,
prompt building, orchestration, prepare-to-model-request and final model preparation/
queue. Its buckets sum to `generation_start_ms`; planner decoding is separated from
load/tokenization/queue inside each round. No measuring/judging LLM call was added.

In the final real local-history request, round 1 had 2,152.95 ms model generation
and 22,934.11 ms preparation/queue, explaining the previously opaque cold-start
time. The entire pre-final breakdown totals 26,305.98 ms. This is a cold request,
so its total cannot establish a latency improvement over an earlier cold run.

### Conservative early exit and routing

Planner-only tool argument `__central_final_after_success=true` declares full
question coverage and no pending dependency. It must be true on every requested
call. The runtime strips it before builtin/MCP validation and execution. The
capability/permission schemas remain immutable; complex or colliding remote schemas
are untouched, and MCP controls are omitted when their added size exceeds the
existing schema budget. A bounded visible `<plan_status>` companion is also supported.

Early exit additionally requires successful nonempty identified evidence, no
truncated/malformed plan, no truncated tool result, and no skipped calls. A search
snippet retains another planner round if its reader is allowed. Mixed-domain gaps
and attachments retain planning. Missing/false completion keeps the existing loop
limit, and all tools in a round still execute. A confirmed complete external plan
does not trigger a redundant automatic local fallback.

Before the optimization, a complete one-tool answer could take planner → tool →
planner → final. It can now take planner → tool → final. The final real Bạch Đằng
request demonstrated `action_rounds=1`, `tool_calls=1`, `model_calls=2`, with a valid
938 answer and an existing chunk citation. The action-round limit remains 2; real
and unit multi-round behavior still works.

The instruction explicitly scopes `search_history` to Vietnamese history and
prefers permitted Wikipedia/web/MCP evidence outside that corpus. Narrow explicit
world-history cues remove local-history schemas from that request's planner view,
without globally disabling the tool or changing `effective_tools`/permissions.
Hallucinated local calls cannot substitute for external evidence in this case.
Vietnamese/ambiguous questions keep local preference/fallback; mixed questions can
use both. Fetched-page dictionaries now enter the source context and replace their
same-ID snippets without changing citation IDs.

The regression uses exactly:
`tóm tắt lịch sử thế giới từ thế kỷ X tới hiện nay`.
Unit tests verify Wikipedia search→fetch and request-allowed MCP without Wikipedia.
The real final run used two `search_wikipedia` calls and no `search_history` call.
It did not fetch a full page; available snippets were insufficient, so the answer
explicitly stated that limitation instead of inventing a chronology. Routing and
grounding pass; this smoke does not claim a complete world-history answer or an
overall quality improvement.

### Completion/termination

The answer is persisted, `request_finished` is marked, and final metrics are
computed once. The same trace is then saved by a scoped update and returned over
SSE/REST. Successful saved, emitted and reloaded traces have the same non-null
`e2e_ms`. It includes answer persistence and excludes the later trace-only write
and network delivery. First status, final TTFT, generation, ITL and streamed text
retain their meanings.

Shared Qwen instrumentation reads actual returned generated IDs, not decoded text
or estimated token counts. EOS yields `stop/false/false`; non-EOS termination at
the configured ceiling without competing time/string stops yields
`length/true/true`. EOS at the ceiling remains `stop`. Unknown backend metadata or
another stop cause remains null. Cancellation is distinguished without asserting
length/truncation. Unit coverage exercises both model IDs; all real smoke requests
stopped naturally.

**Central final ceiling remains 1536**, planner ceiling remains 256. Sampling,
thinking, dtype, quantization and model identity were not changed.

## Hybrid

Modified `app/config.py`, `.env.example`, `modal_app.py`, `README.md` and
`MODAL_QUICKSTART.md`. Default output ceiling is **768 → 1536**.
Both vanilla and SFT stream through the unchanged request path selecting
`settings.hybrid_max_new_tokens`; there is no variant-specific ceiling or runtime
hardcode. `app/rag/hybrid_runtime.py` prepares evidence/prompt and does not choose
the final generation budget. Existing `app/services/metadata.py` already reads the
same setting; new tests verify its actual budget for both variants.

`HYBRID_MAX_NEW_TOKENS=1024` and `2048` remain valid overrides. Modal forwards an
explicit host override and otherwise leaves the default to container Settings.
Trace `generation.settings.max_new_tokens`, done metrics/settings and baseline
metadata report the actual resolved value. Shared stop/length/unknown telemetry
applies equally to Hybrid.

Real vanilla: budget 1536, natural EOS at 72 tokens, chunk citations retained.
Real SFT/PEFT: override 2048, adapter attached with unchanged fingerprint
`742c5ffdfb6f0bc0ff8be65558b1836ad7c008033ad3828473064ebb67fe3ece`, natural EOS at
13 tokens. The SFT smoke returned the correct year but no inline citation; this
task does not alter or claim to improve adapter answer/citation quality.
The ceiling does not force longer answers; the final grounding prompt is unchanged.

## Warm retrieval benchmark

Modal A100, V1 624,288 chunks, shared retrieval resources. Two excluded warmup
rounds, then 10 samples per query/backend; backend order alternated. Timings are
from the retrieval endpoint, with `final_k=3`. Dense timing aggregates the
unchanged query variants. Parameters, HNSW, weights and algorithms were untouched.

| Query | Backend | Dense p50 ms | Dense p95 ms | Retrieval p50 ms | Retrieval p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Exact world-history query above | FAISS | 116.30 | 122.62 | 479.91 | 486.24 |
| Exact world-history query above | Qdrant | 345.14 | 349.65 | 709.57 | 714.85 |
| Ngô Quyền giành chiến thắng trên sông Bạch Đằng vào năm nào? | FAISS | 232.12 | 243.83 | 617.76 | 632.62 |
| Same Bạch Đằng query | Qdrant | 532.65 | 537.21 | 916.33 | 921.20 |

This is a small latency sanity sample, not a quality evaluation or evidence that
either backend is generally better. The world-history retrieval benchmark times
the local pipeline only; it does not establish relevant world-history evidence.

Raw local ignored artifacts:

- `reports/runtime_telemetry/vanilla.json`: paired warm samples and initial live checks.
- `reports/runtime_telemetry/final-vanilla.json`: final planner completion protocol and real routing.
- `reports/runtime_telemetry/sft-2048.json`: real PEFT and Modal budget override.

Modal runs:
[benchmark](https://modal.com/apps/tungduong156gli/main/ap-3HZnBjWbnF34IJVQpifeZZ),
[final Central](https://modal.com/apps/tungduong156gli/main/ap-NYEJ45rsF3lS9fZPDKyiKc),
[SFT override](https://modal.com/apps/tungduong156gli/main/ap-8YCo6LWOLIoti4T0OnVUec).

## Validation actually executed

From the repository root (PowerShell), using the existing `.conda` runtime:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONCASEOK='1'
& ./.conda/python.exe -m pytest -q
```

**343 passed in 54.51s.** Plain full `pytest -q` initially failed collection because
the Windows checkout spells the directory `Training` while imports use `training`.
`PYTHONCASEOK=1` resolves that interpreter casing issue without editing training.
An intermediate new metadata-test fixture failed because it omitted the live
`app.state.hybrid_runtime` field; the fixture was corrected, and the final full
suite passes. The full suite includes FAISS/Qdrant, request isolation, MCP real SDK
transports, cancellation, conversations, attachments, citations and runtime tests.

Relevant final runtime test command, before the final schema-budget regression:

```powershell
& ./.conda/python.exe -m pytest -q tests/test_central_planning.py tests/test_generation_telemetry.py tests/test_mcp_integration.py tests/test_baseline_runtime.py tests/test_model_variants.py tests/test_qdrant_lanes.py tests/test_dynamic_retrieval.py
```

**147 passed in 31.41s.** The additional schema-budget regression is included in
the final 343-test full run.

In `frontend/`:

```powershell
npm test -- --run
npm run lint
npx vitest run tests/ui/DeveloperTrace.test.jsx
npm run build
```

**67 Node tests + 75 UI tests passed; lint passed; targeted Planning UI test passed;
production build passed.** The first lint run found an unused React import in the
new test; it was removed and lint/test/build passed afterward.

Remote checks, after setting UTF-8 and:

```powershell
$env:MODAL_SERVER_URL='https://api.modal2.com'
$env:RETRIEVAL_AVAILABLE_BACKENDS='faiss,qdrant'
$env:MODAL_QDRANT_SECRET_NAME='vn-history-qdrant'
& ./.conda/Scripts/modal.exe run --write-result reports/runtime_telemetry/vanilla.json scripts/modal_runtime_telemetry_smoke.py::runtime_telemetry_smoke --variant vanilla --repeats 10
& ./.conda/Scripts/modal.exe run --write-result reports/runtime_telemetry/final-vanilla.json scripts/modal_runtime_telemetry_smoke.py::runtime_telemetry_smoke --variant vanilla --repeats 0
$env:HYBRID_MAX_NEW_TOKENS='2048'
& ./.conda/Scripts/modal.exe run --write-result reports/runtime_telemetry/sft-2048.json scripts/modal_runtime_telemetry_smoke.py::runtime_telemetry_smoke --variant sft --repeats 0
```

**All three runs passed.** They verify real streaming, saved/emitted trace equality,
non-null E2E, actual budget, actual EOS, shared resource reuse and real external
world-history routing. The final smoke differs from the final source only by a
subsequent budget guard for optional MCP planner-control schemas; builtins in the
smoke are unaffected, and that guard passed the final full unit suite.

Additional checks:

```powershell
& ./.conda/python.exe -m compileall -q app scripts/modal_runtime_telemetry_smoke.py tests/test_central_planning.py tests/test_generation_telemetry.py
git -c core.safecrlf=false diff --check
git diff --exit-code origin/main -- app/rag/retrieval.py app/services/rag_service.py app/tools/policy.py app/mcp app/rag/prompting.py training artifacts evaluation configs
```

All returned exit code 0. Retrieval, request-local backend selection, MCP manager/
policy, frozen evaluation/training files, adapters, corpus and grounded prompt
bytes are unchanged.
