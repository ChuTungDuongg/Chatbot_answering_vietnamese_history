# Qdrant Cloud to Local Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Copy an existing Cloud collection to a new local Docker collection, retaining IDs, all vector types, payloads and compatible indexes, with safe restart and auditable verification.

**Architecture:** A standalone REST migration uses qdrant-client for point operations and raw REST for collection/index configuration so unknown configuration is not lost through model deserialization. An atomic checkpoint and a matching server-side collection metadata token authorize resume. Bounded batches and a rolling source digest permit complete streaming verification after transfer.

**Tech Stack:** Python, existing qdrant-client==1.19.1 dependency, httpx, portalocker, pytest and httpx.MockTransport.

**Spec:** The user's 15 migration requirements in this chat; operator documentation/design is docs/qdrant_cloud_to_local.md.

## Global Constraints

- Keep production code, compose.day03.yaml, APP_MODE=api-only and volumes unchanged.
- Never delete Cloud/local collections, points, volumes or snapshots; never push or merge.
- Never write to vinai_day03_lab or an unowned existing target.
- Credentials come only from QDRANT_CLOUD_URL / QDRANT_CLOUD_API_KEY and secure getpass; no credential values in output or state.
- Source clients enforce read-only operations; destination URLs must be loopback.
- Stop if schema fields cannot be reproduced. Explicitly approved topology adjustments require --allow-config-adjustments and appear in the report.
- Dry-run performs reads only; copying and full verification use bounded batches.
- Copy source vectors directly; do not import application startup or rebuild embeddings.

## Review Focus

- Ambiguous create responses and crashes after upsert: resume requires checkpoint token plus matching target metadata, then replays the last unacknowledged batch.
- Same-count source changes: compare the transfer's rolling digest against a fresh complete source scan.
- New/unknown collection and payload-index settings: fail or report acknowledged topology changes, never silently omit options.
- Concurrent runs and lost/replaced collections: OS checkpoint lock and ownership recheck before writes; do not recreate a partially copied target.
- API/server errors containing credentials: redact known secrets and never serialize raw exception messages, responses, point contents or request headers.

## Files

- Create scripts/migrate_qdrant_cloud_to_local.py: CLI, REST clients, checkpoint, transfer and report.
- Create tests/test_qdrant_cloud_to_local.py: isolated mocked REST integration and failure/recovery tests.
- Create docs/qdrant_cloud_to_local.md: safe PowerShell setup, dry-run, copy, verify and resume.
- Do not modify dependency files: reuse the pinned qdrant-client already declared in requirements-runtime.txt.

## Task 1: Preflight and safety

- [x] Write and run failing tests for dry-run schema/count/estimates, configuration discovery, loopback/protected/existing target guards, secret-safe errors and unsupported configuration.
- [x] Implement Endpoint, Options, source selection and collection configuration planning.
- [x] Run focused tests, confirming reads do not mutate either database or create checkpoints.

## Task 2: Transfer, recovery and validation

- [x] Write and run failing tests for dense/named/sparse/multivectors, integer/UUID IDs, full payload preservation and payload indexes.
- [x] Add atomic state, server metadata ownership, OS locking, retried wait=True upserts and progress.
- [x] Test interrupted writes, ambiguous collection creation, foreign/missing checkpoints and collection replacement.
- [x] Implement counts/schema/index/sample/similarity checks plus full streaming comparison and source stability digest.
- [x] Verify corruption produces FAIL; verify-only performs no database writes.

## Task 3: Documentation and operational checks

- [x] Document exact PowerShell commands including secure key prompt, --dry-run, --resume and --verify-only.
- [x] Run migration tests, CLI help and repository test command; report unrelated environment failures precisely.
- [x] Check local connectivity and cloud credential presence without exposing credentials. Real dry-run/migration blocked by missing Cloud URL/key; exact operator commands documented.
- [x] Obtain independent code review and address material findings with RED-to-GREEN regression tests.
- [x] Inspect final diff and unchanged Compose hash; tracked production files have no changes.

## Execution record

- Existing linked worktree: training/day03-compose; user-created compose.day03.yaml is untracked and must stay unchanged.
- Local Qdrant v1.17.1 is reachable; collection list is empty.
- Cloud environment variables are absent; real migration awaits operator credentials.
- Test dependencies are isolated under ignored .venv/qdrant-migration-deps; package installation did not modify global Python.
- Focused verification: `python -B -m pytest tests/test_qdrant_cloud_to_local.py -q -p no:cacheprovider` -> 42 passed.
- Additional safety tests confirm strict-mode read/write/query limits, accurate effective batch estimates, and rejection of getpass's echoing fallback.
- Independent review found two Important issues, both fixed with failing-then-passing tests: ownership checks on each index retry; meaningful absent source schema families (extra sparse vectors, collection/per-vector quantization, enabled strict mode) must not be silently accepted.
- Other regression fixes cover report/checkpoint collision, exact dense vector names and ownership checks on every upsert retry. No deferred review findings.
- Whole repository `python -B -m pytest -q -p no:cacheprovider` cannot collect 15 modules because this host lacks `fastapi`, `uvicorn`, and `bm25s`. No production dependencies/settings were changed to mask this environment limitation.
- Collection errors: test_annotation_workflow, test_baseline_runtime, test_central_planning, test_conversation_sources, test_dynamic_retrieval, test_frontend_gateway, test_gateway_regressions, test_generation_telemetry, test_mcp_integration, test_observability_regressions, test_qdrant_lanes, test_repository_hardening, test_serving_observability, test_silver_annotation, test_training_six_way_drive.
- CLI help passes; real `--dry-run` without QDRANT_CLOUD_URL stops with sanitized setup error (exit 2) before any writes. Live local server remains v1.17.1 with no collections.
- Compose SHA256 baseline: E68236E9B81D4D444AD112419C384CE80FA74889D7DCF1A275CA39A92281C2EF.
