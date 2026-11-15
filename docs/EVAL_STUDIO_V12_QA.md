# Eval Studio v1.2 — implementation and local QA

Date: 2026-11-16. Base: `6d6b58b` on `main`. No production deployment or paid
model requests were performed. The existing Studio data is never reset.

## Completed scope

- Additive benchmark jobs/items, append-only attempts and deterministic measurements.
- Existing Hybrid/Central execution paths, quotas and tool policies; per-request prompt
  and tool-description variants, pinned configuration/runtime fingerprints.
- Owner-confirmed start/resume, sequential rate control, bounded retries, cancellation,
  durable leases and explicit recovery after restart. No startup inference.
- Snapshot question input only; references/expected behavior stay outside inference.
- Actual evidence snapshots, final context/citations, safe tool diagnostics and known
  usage/latency. Missing usage/cost remains null.
- Expected behavior metadata and existing lexical/citation/grounding metrics, clearly
  separate from historical correctness and human scores.
- Automated paired comparison, metric-specific W/T/L, B-minus-A regression labels,
  per-question evidence drill-down; legacy human comparison/blind review preserved.
- Versioned optional judge profiles, structured offline contract/import, calibration.
  No automatic judge execution or judge configuration UI in this milestone.
- Backup schema 1.2 with 1.1 restore compatibility; all five new tables included.
- Responsive Benchmarks workspace, safe preview/start controls and legacy integration.

## Reproduced findings fixed before release

| Severity | Root cause and fix | Regression |
| --- | --- | --- |
| P1 | API passed unrelated job fields to execution freeze; pass only supported variant settings. | Real runtime API test with offline OpenAI SDK fixture. |
| P1 | Cancellation arriving alongside a finished provider result discarded the answer; retain completed task result. | Finish-then-cancel runner test. |
| P1 | Experiment could change between freeze and job insert; compare full expected experiment inside the write transaction. | Mutate experiment between freeze/create. |
| P1 | Expired worker could reset a newer attempt for retry; require current lease and attempt under a transaction. | Two-worker stale retry test. |
| P1 | Timeout cancellation set the job-wide user cancel token; keep external timeout and user cancellation separate. | Native stream cleanup and runner timeout/retry tests. |
| P1 | Overlapping cancellation interrupted provider close; shield child cleanup and avoid double cancellation. | Controlled slow-closing provider stream. |
| P1 | Sanitizing free-form MCP errors changed failure into success; classify failure before sanitization and keep a safe generic code. | Actual ToolRegistry failing MCP fixture and zero success rate. |
| P1 | Internal run insert bypassed completed-answer/state validation; validate before atomic attempt persistence. | Empty completed answer/invalid state rejection. |
| P2 | Missing source kind became invented history provenance; preserve null. | Captured evidence with unknown kind. |
| P2 | Grid content escaped the mobile main viewport; constrain grid tracks and wrap labels/notes, keep tables locally scrollable. | Main scroll/client and panel-bound assertions on all viewports. |

Harness corrections: unique snapshot names permit E2E reruns on the same fixture;
role-based select locator matches its accessible name; Node `process` import fixes
ESLint in the CI browser config. Legacy-schema backup mutation fixture skips tables
that do not yet exist, retaining its schema-tampering checks.

## Verification

- Full working-tree backend suite: **816 passed**, 0 failed/skipped, 122.81 seconds.
  Three cases come from pre-existing uncommitted parameterization in
  `tests/test_latency_experiments.py`; that file is intentionally excluded from this
  change. Its committed baseline was separately tested: **14 passed**.
- Frontend: **67 Node tests + 119 Vitest tests passed** (13 Vitest files).
  Windows command: `npm --prefix frontend test -- --pool=threads --maxWorkers=1`.
- ESLint, production Vite build, Python compileall and formatting check passed.
- Real authenticated offline Studio E2E: **21 passed**, desktop/tablet/mobile.
  Additional focused reruns verify the new stronger responsive assertions.
- Existing chatbot browser suite: **51 passed**, desktop/tablet/mobile; no provider calls.
- Evidence and comparison screenshots inspected at desktop/mobile dimensions.
- Restore/migration compatibility, atomic rollback, idempotency, owner/flag protection,
  blind review and existing agent/runtime regressions are included in backend coverage.
- Staged path/credential-pattern checks exclude databases, backups, credentials, cache
  and browser artifacts. Existing unrelated files were compared byte-for-byte against
  preserved copies and remain outside the commit.

The CI workflow now builds same-origin assets and runs the complete authenticated
Studio browser suite using an in-process offline executor and installed Chromium.
Cloud CI status must be checked on the actual pushed SHA; local results do not assert
that a future workflow is green.

## Operational limits

Docker CLI exists locally, but the Docker Desktop daemon is unavailable; container
build/smoke verification is delegated to the existing CI container job. Vite reports
a nonfatal JavaScript chunk-size warning (~529 kB minified).

Actual paid-model answer quality, GPU inference, production Qdrant and Modal
persistence/eviction/recovery were not exercised. Follow the staging/runbook in
[EVAL_STUDIO.md](EVAL_STUDIO.md) before enabling `EVAL_STUDIO_ENABLED`; it remains
false by default. Keep one SQLite writer/container. A process can die after upstream
billing but before persistence; explicit resume warns about uncertain attempts.

No exactly-once billing guarantee, external durable queue, automatic judge execution,
semantic historical verification, or reviewer-role isolation is claimed. Prompt
variants are limited to supported resources; tool variants change descriptions while
keeping executable schemas/policies. Legacy snapshot questions still require their
working question ID because of the existing response foreign-key relationship.

## File groups

- Backend: `app/eval_studio/{benchmark,benchmark_api,execution,behavior,analytics,judge}.py`,
  existing schema/store API compatibility and backup integration, `app/main.py` shutdown/router.
- Frontend: Benchmarks, automated comparison, expected behavior editor, measurements;
  existing navigation/evidence/API/CSS integration.
- QA: seven new backend test files, updated backup-security fixture, offline browser
  server, v1.2 component/E2E tests, repeatable legacy E2E and CI configuration.
- Documentation: Studio runbook, implementation plan, this report and README entry.

Preserved unrelated work: `.env.example`, `modal_app.py`,
`tests/test_latency_experiments.py`, and untracked `docs/P2_DAY1_AUDIT.md`.
