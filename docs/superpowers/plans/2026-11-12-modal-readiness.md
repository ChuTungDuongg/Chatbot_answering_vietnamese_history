# Modal readiness before opening the browser

> **For agentic workers:** Use superpowers:executing-plans for this small change.

**Goal:** A terminal-owned deployment opens the website only after its ASGI startup completes.

**Design:** Keep the existing deployment and exact-App-ID cleanup. Discover the public `fastapi_app` URL from Modal metadata, start following logs, and warm `/health` with bounded, unauthenticated HTTP probes. Only HTTP 200 with JSON `status: ok` counts as ready. Then print the ready URL and open the browser. Cold-start time remains visible in the terminal.

**Approaches:** Keeping a GPU permanently warm would add idle cost; changing backend startup would expand this fix into retrieval lifecycle work. A local readiness gate addresses the observed deploy-versus-startup race without either change.

**Tech stack:** Python asyncio, existing HTTPX/Modal SDK dependencies and pytest.

**Spec:** The design above implements the owner's request to open the site successfully after starting the deployment.

## Constraints and review focus

- Preserve authentication, inference, resources, scale-to-zero and named Volumes.
- Never send passwords, cookies, API keys or service tokens in readiness probes.
- Bound startup waiting to 960 seconds by default; permit a positive finite `--ready-timeout` override.
- Ctrl+C during probing must stop the exact owned app without opening a browser.
- HTTP errors, redirects and malformed success bodies must not produce a false ready message.
- A failed startup must exit nonzero and clean up the app; a browser-launch failure must leave the ready URL usable.
- Resolve URL and environment from Modal rather than hardcoding an owner's URL.
- Support `--no-browser` for headless terminals and preserve attach behavior.

## Task: Readiness gate and regression coverage

- [x] Extend lifecycle tests for readiness ordering, timeout, cancellation, browser failure and attach.
- [x] Add real local HTTP-server tests for slow startup, invalid responses, timeout and cancellation; observe RED.
- [x] Implement once-per-session URL discovery and a bounded, cancellable health probe in `scripts/modal_session.py`; observe GREEN.
- [x] Update README deployment flow and `docs/DEPLOYMENT.md` with the ready message, options and cold-start limits.
- [x] Run targeted tests, full Python suite and compile checks; record the unavailable live check without restarting the stopped app.
- [ ] Review the full change, commit, run CI, integrate as previously authorized, and update the owner's checkout without disturbing its local edits.

## Execution record

- Baseline: 9 lifecycle tests passed on `5aae558`.
- Ruling: Reuse the attached worktree on `codex/modal-ready-before-open`; the owner's main checkout has unrelated local edits.
- RED: 14 missing-feature failures in the initial HTTP/metadata tests. Review reproduced four further failures for slow-stream deadline/cancellation and pending metadata/log startup.
- GREEN: 45 targeted lifecycle/readiness tests passed, including the review regressions and CLI option wiring.
- Initial full run inherited the owner's API-only `.env`: 16 local-provider tests failed (`Local inference is unavailable`), 633 passed. With test-only environment `LOCAL_INFERENCE_ENABLED=true OPENAI_ENABLED=false CENTRAL_MAX_ACTION_ROUNDS=4`, all 104 tests in the affected files passed and the intermediate full suite passed (649 tests). No real local models are loaded by these fakes; deployment settings were not edited.
- Review ruling: Use a cancellable async startup task covering metadata and complete HTTP reads, also monitoring log-process exit. A per-probe deadline closes slow streams; cancellation awaits resource cleanup before exact-ID cloud stop.
- Live check could not run: Modal reports `vn-history-rag-api` stopped at 22:16:22 +07:00 on 2026-11-12, with zero containers. No deployment was restarted for this local-controller fix.
- Resume, 2026-11-13: reproduced pending-DNS deadline/cancellation failures before fixing them. Own the startup event loop and close it without joining the OS resolver; HTTP tasks and async generators are cleaned up before exact-ID cloud stop. Interpreter exit may still wait for DNS after cloud cleanup (documented).
- Final verification: 659 Python tests passed (one existing Starlette/AnyIO deprecation warning), compileall passed, all five README Mermaid diagrams parsed using the existing local parser and DOM environment. The reviewer independently ran all 47 lifecycle/readiness tests and confirmed cloud-stop begins while DNS is still pending.
