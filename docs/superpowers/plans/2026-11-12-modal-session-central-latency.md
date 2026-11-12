# Modal terminal session and Central latency plan

> Execute inline using the executing-plans and test-driven-development workflows.

**Goal:** Let the owner stop the deployed app with Ctrl+C and reduce avoidable Central planning work while retaining the selected final reasoning effort.

**Architecture:** A local Python CLI owns a deploy/attach session, follows logs and stops the exact Modal app ID on interruption. Central uses a separately configured planning reasoning effort (default low, inherit available); final reasoning, provider, evidence checks and budgets remain unchanged.

**Tech stack:** Python subprocess/signals, existing Modal CLI/SDK, Pydantic settings, OpenAI Responses, pytest.

## Evidence and constraints

- The owner requested foreground Ctrl+C control and supplied the Napoleon/Waterloo/Hitler comparison with Luna/Very High.
- A recent Central Luna/xhigh trace measured 99.52s E2E: 54.69s planning, 22.99s tools, 21.77s final generation, four planning rounds. Logs do not contain question text, so this is not a confirmed question-level benchmark.
- Keep the stable deployment URL, auth, secrets and named Volumes. Do not use a second public dev URL or delete data.
- Do not deploy/start/stop the owner's current app during tests. Use fake CLI boundaries for lifecycle tests.
- Keep Very High for final generation; expose actual planning/final settings in diagnostics. No silent provider fallback or smaller final evidence budget.
- Frozen prompt resources remain unchanged. Extend the live agentic prompt only.

## Review focus

- Ctrl+C during deploy as well as while following logs must clean up the owned app.
- A failed deploy must not falsely report success or stop an unrelated deployment.
- Child CLI processes must exit and stop failures must return a useful nonzero result.
- Per-request planner options must not mutate final options or concurrent requests.
- Query batching and early finish must retain dependency, tool permission and citation checks.

## Tasks

- [ ] RED/GREEN: terminal session lifecycle tests, including attach, deploy, interrupt, failure and environment scope; implement scripts/modal_session.py.
- [ ] RED/GREEN: stage-specific reasoning and settings validation; wire Central startup and request clones, add telemetry.
- [ ] Extend live planner instructions for batching independent subquestions and avoiding unnecessary completion-only rounds; preserve frozen contracts.
- [ ] Run targeted and full feasible suites, regenerate the source-hashed offline report; document measured baseline and limits of verification.
- [ ] Review changes, commit, push, verify CI and deliver the PowerShell command. Keep the running deployment untouched until the owner opts to run the new command.
