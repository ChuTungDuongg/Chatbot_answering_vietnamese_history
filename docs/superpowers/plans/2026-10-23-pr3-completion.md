# PR #3 audit and completion plan

Goal: Complete the existing three features on PR #3 without altering retrieval, models, SSE contracts or frozen offline prompts.

Spec: User request in this chat, 2026-10-23, five phases. Execution: inline, implementation and PR update already authorized.

Architecture: retain the GPU ASGI app and CPU gateway; repair process-local metrics and use vLLM 0.23.0 scheduler statistics with fixed numeric fields.

## Tasks

- [x] Read all 18 original changed files and relevant dependencies; compare with main.
- [x] Run baseline Python and frontend checks.
- [x] Add regression tests then repair metrics, cancellation, cache stats and admin authentication.
- [x] Add real HTTP streaming/upload/error/disconnect checks then repair gateway ownership and build isolation.
- [x] Strengthen live grounding/style instructions; preserve offline prompts and test Waterloo evidence boundaries.
- [x] Review CI, run full verification and check Docker/Modal availability.
- [ ] Write Vietnamese report with actual results, review final diff, commit, push and update existing PR.

## Review focus and ledger

Preserve dirty main. Bound metric labels, accept only finite nonnegative numbers, count all requests in the rolling window, report unavailable hardware data explicitly. Cancellation must close generation and upstream resources. Verify streamed first bytes before completion. Keep bearer data out of cache/logs. Do not infer casualties from troop strength.

Initial findings: max_samples truncates 5-minute counts; cancellation has no finish mark; nonexistent prefix-cache runtime attribute; output_tokens/cache not sanitized; non-ASCII bearer raises TypeError; proxy cancellation/body failures can leak resources; compressed uploads lose Content-Encoding; local Vite config may enter production build; no frontend Docker CI.

Ruling: keep accepted chat execution counters and HTTP admission/status counters separate. Use pure ASGI send observation for the three fixed serving routes so validation failures count without buffering SSE. User explicitly authorizes implementation so no separate design approval round. Native worktree is outside writable roots, therefore use reviewed shell writes to that exact attached checkout.

Phase 6: User authorizes ready + squash merge only after all checks and deployment acceptance pass. Never bypass protection.
Observed: 399 Python, 67 Node, 78 Vitest and 45 Playwright pass; Docker built and Linux gateway 8 pass. Modal CPU dotenv startup failure fixed and real PDF upload/ready verified. Initial real Waterloo samples show factual weaknesses despite empty lexical flags; prompt tightened and final live checks pending.
