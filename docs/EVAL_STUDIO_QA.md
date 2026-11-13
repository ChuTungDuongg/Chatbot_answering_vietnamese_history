# Eval Studio v1.0/v1.1 — final local production readiness audit

Audit date: 2026-11-13. Repository base: `6ad88b2` on `main`. This audit includes
the previously uncommitted Studio implementation and its integration. No production
deployment, paid model requests, automatic judge, or RAG algorithm changes were made.

## Result and boundaries

**PASS WITH WARNINGS for local QA.** No reproduced P0/P1 remains unresolved.
Modal persistence, real GPU inference, production Qdrant, and cloud recovery were not
executed. The runbook in [EVAL_STUDIO.md](EVAL_STUDIO.md) must be completed in staging
before enabling the Studio in production. Test results establish application behavior,
not historical answer accuracy or deployed infrastructure performance.

Reviewed: Pydantic schemas, SQLite store/migrations, API authorization and feature gate,
batch import, experiments/runs/comparison, blind assignment/reveal, evidence provenance,
backup/restore CLI, React editor/autosave/request ownership, downloads, navigation,
responsive themes, and the auth/FastAPI/Modal integration. Existing Hybrid/Central,
tool registry/MCP, provider, SSE, citations, conversations and static routing were
covered by the complete repository regression suite.

## Reproduced issues and fixes

27 product defects were fixed: **0 P0, 12 P1, 15 P2**. Related manifestations are grouped
under one root cause below. Test harness fixes (Origin/Secure-cookie behavior and
separating real-backend Eval tests from mocked chatbot tests) are not counted as product
defects. Every listed fix has regression coverage; tablet layout also has browser
overflow/ID wrapping assertions and before/after screenshots.

| ID | Severity | Root cause → fix |
| --- | --- | --- |
| B01 | P1 | Run import preview omitted experiment configuration → token binds the referenced immutable snapshot and full experiment metadata; changed configuration returns 409. |
| B02 | P1 | Credential patterns missed quoted JSON keys and escaped newlines → input detection and decoded backup-record scanning; validation errors omit rejected input. |
| B03 | P1 | Diagnosis filter ran before choosing a run, resurrecting old scores → choose latest reviewed run/review first, then filter. |
| B04 | P2 | Cached import success ignored deleted/modified records → transactional receipts verify record/evidence fingerprints before returning success. |
| B05 | P2 | Import external IDs treated as globally unique → scope to `(experiment_id, external_id)`. |
| B06 | P2 | BOM-preserving CSV callers could not import exported CSV → remove leading UTF-8 BOM before parsing. |
| B07 | P2 | Repeated human-review submission appended duplicates → serialize and reuse an identical latest review; genuinely revised conclusions remain append-only. |
| B08 | P2 | SQLite `executescript` committed partial DDL before later failure → explicit transaction includes migration/seed and rolls back injected schema failure. |
| B09 | P2 | Very large numeric imported IDs poisoned SQLite allocator → bound allocated-ID range, preserve external IDs, repair old invalid counters. |
| B10 | P2 | Restore checked only columns/FKs → validate normalized table constraints and unique indexes too. |
| B11 | P2 | Backup/restore-copy discarded explicit lookup indexes → retain indexes; accept older archives missing optional nonunique indexes. |
| B12 | P2 | Unescaped `#` in SQLite URI selected the wrong file → read-only paths use encoded absolute file URIs. |
| F01 | P1 | Studio inherited remote chatbot API URL → owner APIs and downloads always use current origin. |
| F02 | P1 | Comparison request completed after filters/selection changed → invalidate stale request ownership. |
| F03 | P1 | Import preview completed after a different file/context was selected → invalidate stale preview. |
| F04 | P1 | File reads completed out of order → generation guard keeps the latest selected file. |
| F05 | P1 | Evidence response from previous run remained visible → keyed inspector and guarded load lifecycle. |
| F06 | P1 | Blind reveal from previous assignment could reach another assignment → assignment-scoped state and request guards. |
| F07 | P1 | Run-edit fetch crossed experiment selection → edit generation guard and reset. |
| F08 | P1 | Rubric scores persisted when switching output → reset per-response review form. |
| F09 | P1 | Delayed output save cleared next question's form → scope completion to original review generation. |
| F10 | P2 | Oversized-file rejection retained prior import content → clear prior file/preview state. |
| F11 | P2 | Workspaces had no browser history state → URL workspace parameter with guarded Back/Forward/refresh. |
| F12 | P2 | Captured empty trace arrays appeared missing → distinguish captured-empty from unavailable fields. |
| F13 | P2 | Evidence drawer leaked keyboard focus → initial focus, Tab containment, Escape and focus return. |
| F14 | P2 | Pending experiment creation allowed repeat submission → synchronous mutation guard and busy form. |
| F15 | P2 | Tablet editor header squeezed stable ID/actions → nowrap ID and stacked tablet header. |

Additional low-risk hardening: indexed FK/lookups, one read transaction for comparison
and evidence, read experiment configuration inside run-write transaction, bounded legacy
import request/record count, and Python formatting at line length 110.

## Verification evidence

| Command / workflow | Actual result |
| --- | --- |
| `.conda/python.exe -m pytest -q --disable-warnings --maxfail=5` | **741 passed**, 0 failed/skipped, 144.34 s |
| `npm --prefix frontend test` | **67 Node + 100 Vitest passed**; includes 17 added frontend QA tests |
| `npm --prefix frontend run lint` | exit 0 |
| same-origin `npm --prefix frontend run build` (`VITE_API_BASE_URL=/`) | exit 0 |
| `black --check --line-length 110` on Eval backend/tests | 13 files unchanged, exit 0 |
| `npx playwright test --config playwright.config.js` | **51 passed**, desktop/tablet/mobile, 40.9 s |
| `npx playwright test --config eval.playwright.config.js` | **18 passed**, desktop/tablet/mobile, 51.8 s |
| `git diff --check` | exit 0 |
| Staged-index copy: original latency tests plus all Eval backend tests | **73 passed**, 10.27 s |
| Staged-index copy: owner auth and Modal lifecycle/resources/readiness/import tests | **61 passed**, 19.75 s |

The first baseline was 713 passed/2 skipped because OpenAI SDK was absent. Installing
the project's pinned `openai==3.48.0` enabled the two offline official-SDK tests; the final
full run has no skips. An overlapping audit run observed a transient SDK import failure
while pip replaced SDK files; both affected tests and the complete final suite passed
after installation completed. No SDK tests called a paid API.

The staged-index copy excludes the user's unstaged GPU/test changes. One supplemental
combined auth/Modal run stopped making progress during the local readiness fixtures and
was interrupted without a completed result. A fresh readiness-only run passed 27/27;
the subsequent combined run with faulthandler diagnostics passed 61/61. No product
failure was reproduced by these reruns. Node fixtures also emit an existing Vite HMR
port-in-use warning while concurrent test servers initialize; tests finish successfully.

Playwright uses real Chrome. Studio tests run against actual FastAPI, owner auth and
temporary SQLite, rather than mocked Studio APIs. Covered: delayed autosave/edit/drain,
reload persistence, two-tab 409 conflict with retained draft/navigation protection,
question review, experiment creation, blind scoring/reveal/reload, same-snapshot paired
comparison, category drill-down, missing evidence, malformed import preview, keyboard
drawer behavior, Back/Forward, all seven workspaces in both themes, overflow checks,
and authenticated ZIP backup download. A successful two-run JSONL import also verifies
captured source text, Selected/Cited flags, final context IDs, citations, retrieval
backend and tool trace in the Inspector. Screenshots are local ignored `reports/` artifacts;
visual review corrected the tablet header and checked overview/editor/review/comparison/
data/backup states. The mocked chatbot E2E suite separately checks existing streaming,
citations, layout, clipboard/uploads and login behavior.

Backend regressions also exercise a failure on the final insert after prior records,
concurrent autosave/review/blind submissions, approval/quotas/duplicate detection,
40 outputs with only 20 reviewed questions, multiple reviews counted once, cross-snapshot
rejection, snapshot context after working-question edits, run-scoped repeated chunk IDs,
v1 migration preservation, corrupt/incompatible backups, existing-target refusal,
empty-target restore, chat-preserving restore-copy, and restart persistence.

## Local scale check

On temporary local SQLite, one atomic batch imported **200 runs** across two experiments,
then stored 200 reviews. Comparison counted **40 questions per experiment and 40 pairs**,
not 200 question scores. One observed sample: preview+import 150.53 ms, comparison
7.85 ms, backup+restore 165.85 ms. These are development-machine measurements, not a
production latency guarantee. Missing telemetry remains null; no metrics are fabricated.

## Security, data and operational conclusions

- All Studio routes use the owner-session dependency and default-off feature flag.
  Existing Origin/CSRF checks and Secure/HttpOnly/SameSite cookie policy are unchanged.
- Migrations are additive/idempotent. Existing question IDs, responses/reviews and
  immutable snapshots remain intact. No drop/recreate migration was introduced.
- Model output/review remain separate from dataset ground truth. Scores use fixed
  30/20/20/20/10 weights; missing reviews are not zero. Latest reviewed run and latest
  review are selected deterministically by SQLite insertion order; telemetry separately
  averages captured values across attempts.
- Blind mappings remain server-side, stable and immutable. Submission/reveal policy is
  transaction-protected. The shared owner role can intentionally inspect normal APIs;
  this is not adversarial multi-user reviewer isolation. Text may contain unknown
  self-identification beyond scrubbed known names.
- Evidence is stored per run and uses snapshot reference content. Inspector never
  queries Qdrant to reconstruct past evidence. Unavailable capture is explicitly shown.
- Backup contains only evaluation tables; chat/auth credentials are excluded. Restore
  creates a new file/copy, validates checksums/schema/relationships and never swaps a
  live database. The checksum is not a signature of backup authorship.
- Modal `/data/chat.sqlite3`, `vn-history-chat-data` mount and mutation Volume commit
  wiring were inspected, **not verified on Modal**. Keep one container/writer and follow
  staged restart/recovery verification before enabling `EVAL_STUDIO_ENABLED=true`.
- Forced tab termination before a successful server save cannot guarantee draft
  durability; beforeunload warns, navigation drains saves, and conflicts preserve the
  local draft. There is no durable offline draft store.
- Snapshot-only questions deleted from the working dataset must be restored before
  adding a completed run because the v1 response FK still references working questions.

## Changed-file scope and preservation

- Backend: new `app/eval_studio/` package; minimal router/flag wiring in
  `app/main.py`, `app/config.py`, `modal_app.py`, `.env.example`.
- Frontend: `frontend/src/eval/`, entry route in `frontend/src/main.jsx`, isolated
  Playwright config/specs, QA component tests, default Playwright exclusion.
- Tests: `tests/test_eval_studio.py`, `test_eval_v11.py`, `test_eval_final_qa.py`,
  `test_eval_security_qa.py`, and local browser fixture.
- Docs: this report, `EVAL_STUDIO.md`, README/doc index links, and the two existing
  Studio implementation plans.

Unrelated pre-existing work is intentionally excluded from commit: the GPU-default
hunks in `.env.example`/`modal_app.py`, `tests/test_latency_experiments.py`, and untracked
`docs/P2_DAY1_AUDIT.md`. Their original contents were saved in `.git/eval-qa-preserved/`
and checked for preservation. No databases, archives, `.env` credentials, dependencies,
screenshots or temporary test outputs belong in the commit.
