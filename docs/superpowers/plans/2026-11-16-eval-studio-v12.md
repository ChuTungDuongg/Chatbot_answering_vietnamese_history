# Eval Studio v1.2 implementation plan

**Goal:** Execute reproducible owner-started benchmarks with captured agent diagnostics while preserving all v1.0/v1.1 data and manual evaluation.

**Architecture:** Add durable SQLite benchmark jobs/items and append-only run measurements. Execute sequentially through the existing chat `_execute` boundary, existing provider quota guard and request tool policy using ephemeral conversation storage. Freeze resolved prompt resources, tool descriptions/schemas and runtime identity per job; apply variants only through benchmark-scoped wrappers, never global prompt or retrieval mutation. No paid calls in development/tests; explicit owner confirmation is required for every start/resume, never automatic startup recovery.

**Tech stack:** existing FastAPI/Pydantic/SQLite, React/Vite/Lucide, pytest/Vitest/Playwright.

## Audit and scope decisions

- v1.1 already stores immutable snapshots, experiment metadata, responses, human reviews, paired human comparisons, fixed blind assignments, evidence, atomic imports and safe offline backups.
- Existing chat paths enforce model allowlists, provider quota/concurrency, tool steering, cancellation, citations and hidden-markup filtering. Reuse these paths rather than a second RAG implementation.
- Existing offline metrics measure lexical answer overlap, citation IDs and evidence phrase/year checks, explicitly not historical correctness.
- Native agent replay may contain encrypted reasoning; wrappers must retain it only in memory for the request and store only allowlisted diagnostics, never raw messages/replay.
- Work is prepared on `codex/eval-studio-v1.2` in the current checkout; unrelated GPU/test/doc changes remain untouched. The user's latest instruction authorizes a normal direct push to `main` after QA; no force push or production deployment.
- Optional judge support in this milestone is versioned rubric/profile, structured result import/export contract and human calibration; automatic judge execution is deliberately not enabled.
- Prompt variants replace a closed set of runtime prompt resources per benchmark request. Tool-definition variants change descriptions only; executable argument schemas and tool policies stay enforced.
- Recovering a crashed in-flight request requires owner resume after lease expiry. An uncertain prior provider request may already have incurred cost; record the interrupted attempt and never automatically replay it.

## Milestones and interfaces

- [x] **M1 — Deterministic agent evaluation:** `behavior.py` defines `BehaviorSpec` and `evaluate_behavior(question, run, evidence, diagnostics)`; `judge.py` supplies versioned structured judge results, prompt contract and calibration. Tests use real stored records/mocked observations, not inference.
- [x] **M2 — Execution boundary:** `execution.py` exposes `freeze_configuration(app, experiment, options)`, `verify_configuration(app, frozen)` and `ChatBenchmarkExecutor(app).execute(question_text, frozen, cancel)`. Result is a validated run-input dictionary plus safe `diagnostics`; inputs never include reference answers. Capture actual full selected evidence plus retrieved chunks when available, planning exit status, argument validity/fingerprints, usage and provider errors. Test real Central/Hybrid paths with fake models/tools.
- [x] **M3 — Durable runner:** `benchmark.py` defines strict create options, additive schema, job/item status and leases, idempotent attempt persistence with runs+measurements in one transaction, retry/backoff/cancel/resume and comparison aggregation. APIs are owner/flag guarded. Tests cover last-write rollback, cancellation, stale leases, no auto-execution, snapshot integrity and partial/missing metrics.
- [x] **M4 — UI:** `BenchmarkWorkspace.jsx` owns create/preview/start/resume/cancel/progress/config identity, per-question evidence/diagnostics; integrate optional behavior JSON editor and separate automated comparison into existing workspaces. Scoped hooks/request ownership, keyboard and responsive states. Component and real browser fixtures never invoke a paid model.
- [x] **M5 — Compatibility and QA:** include new tables in backup/restore with legacy schema compatibility, preserve v1 snapshots/records, lock recorded configurations, document workflow/limits. Run all backend tests, frontend tests/lint/build, mocked execution E2E and legacy Eval E2E on desktop/tablet/mobile.

## Review focus

- Start/resume requests race: one durable lease admits one worker; no automatic paid execution on app startup.
- Cancel or process death after provider completion: writes are idempotent and atomic; uncertain in-flight attempts are visible and only explicitly retried.
- Working question changes or deletion after snapshot: model input and grading use snapshot; v1 FK limitation must be surfaced before execution.
- Missing traces/usage/reviews: unknown remains null, never a fabricated success, zero score, or historical truth claim.
- Config drift across resume: prompt/schema/runtime fingerprints are pinned or rejected, with no mutation of production defaults.
