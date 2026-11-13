# Eval Studio v1.1 — implementation plan

**Goal:** Preserve v1.0 while adding traceable experiments, honest comparisons, transactional import, offline backup and blind review.
**Architecture:** Add idempotent SQLite tables without changing legacy records; every run pins a snapshot and links a legacy-compatible response/review. Keep confidential blind assignments server-side. Use a validated preview token and one transaction for imports, and an evaluation-only SQLite backup with offline restore to a new file. Build dedicated React workspaces on existing tokens and auth.
**Tech stack:** FastAPI, Pydantic, SQLite, React/Vite, Playwright.

- [x] Test and implement additive migration, experiment/run persistence and review association.
- [x] Test snapshot-safe aggregation and blind assignment/reveal invariants.
- [x] Test atomic idempotent bulk import, preview and revision conflicts.
- [x] Test consistent evaluation-only backup and safe offline restore.
- [x] Build experiments, comparison, blind review, evidence and import/backup workspaces.
- [x] Run legacy and new backend tests, lint/build and browser workflows, document gaps and Modal verification limits.
