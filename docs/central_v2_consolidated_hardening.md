# Central V2 consolidated retrieval and answer hardening

This pass works from the current repository state and preserves PREPARE → INITIAL_GROUNDING → optional ACTION / TOOL_EXECUTION → SYNTHESIS → optional QUALITY_REPAIR → FINAL. Public mode `central`, the default `Qwen/Qwen3-8B` base, optional V2 adapter, Hermes structured function calls, host-side history grounding, and the future domain-policy hook remain unchanged. No role-agent runtime is added.

1. **Raw citation root cause.** The previous host formatter deliberately expanded validated `[S1]` aliases into `[real_chunk_id]`. The source drawer also rendered `chunkId` in a `<code>` element. Those two paths exposed internal IDs even when generation and validation were correct.

2. **Citation display.** One request-local map now connects each alias to its original source ID, display index, title, kind, and comparison target. Only validated aliases are formatted as `[1]`, `[2]`, etc. Selected exact legacy IDs can still be canonicalized to aliases during validation; arbitrary numeric brackets are never looked up, even if a real source ID happens to be numeric. API source/context schemas, streaming sources, and stored conversation sources preserve `display_index` and provenance. Source cards show `[index] Title`; Agent Trace retains IDs. Citation order cannot renumber sources.

3. **Analytical parsing.** Vietnamese deterministic patterns now extract event, dynasty subject, actors, outcome, and facets alongside the existing question type and comparison targets. NFC/folding supports accented input and common unaccented forms. Accent-sensitive checks distinguish the surname Trần from trận. Comparison target names are removed before facet detection so Điện Biên does not become a request for diễn biến.

4. **Structured examples.**

   | Question | Type | Subject/event | Actors | Outcome |
   | --- | --- | --- | --- | --- |
   | Nguyễn Cao Kỳ là ai… | biography | subject: Nguyễn Cao Kỳ | — | — |
   | Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần? | cause | subject: Nhà Trần | — | suy yếu |
   | Vì sao Mỹ và VNCH lại thua chiến tranh Việt Nam? | cause | event: Chiến tranh Việt Nam | Mỹ; Việt Nam Cộng hòa | thất bại |
   | Vì sao Cách mạng Tháng Tám thành công? | cause | event: Cách mạng Tháng Tám | — | thành công |
   | So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ. | comparison | targets: both original names, preserved independently | — | — |

5. **Entity/event consistency.** The reusable layer checks normalized titles, URL/page titles, parenthetical scope, text, and source metadata. It preserves biography exact-title behavior, rejects the Chinese Nhà Trần scope, and filters calendar, street, square, city, tourism and commemoration collisions for event queries. Chiến thắng/Chiến dịch/Trận variants can refer to the same battle. This is a small set of grammatical and category rules, not an entity database.

6. **Query planning.** Cause plans use the target, applicable cause/outcome wording and actors, plus the original question. Comparison plans are separate for each target: the target itself plus useful/requested dimensions. Defaults are two variants per target/plan; the configurable bound is one to three. Independent history calls execute concurrently. Biography keeps its single history query.

7. **Overview anchors.** Exact canonical event/dynasty pages, including battle-title variants, receive an overview reservation. Therefore a relevant Chiến tranh Việt Nam overview survives stronger local campaign scores. Each comparison side can independently select its own canonical anchor.

8. **Coverage-aware selection.** Candidates are annotated with dimensions actually present in their text. Greedy selection rewards new requested dimensions and military, strategy, political, economic/logistical, domestic, international/diplomatic and opponent dimensions. Raw reranker scores are used within a retrieval batch only. Central requests ranked candidates before the shared retriever's final diversity selection through an opt-in `candidate_pool` argument; other callers keep the old default. Retrieval defaults to ten candidates per query, while synthesis defaults to four chunks (configurable three to six). Source admission is independent of the action-observation character budget, removing the earlier first-target budget starvation.

9. **Temporal handling.** For decline/collapse questions, late-dynasty causal cues are preferred over explicit aftermath/restoration cues. Hậu-dynasty wording is derived from the parsed dynasty name. Aftermath-heavy chunks are downranked and cannot count as strong causal support. No chronology database or uncertain historical dates are invented.

10. **Target-provenance bug.** Previously `grounding_targets` was consulted after normalization only to update a coverage counter/tool trace. Candidate and source objects never received that identity. Each target-specific result is now tagged before filtering and carries its target through ranking, normalization, merging, selection and citation metadata. When the same real source appears for both targets, a single display entry retains both verified origins in `comparison_targets`, with a primary `comparison_target`. Merging or truncating text rechecks those associations. A repeated host fetch can reuse its cached result for the second target without another network call.

11. **Comparison balance.** Each target has its own filtered/ranked queue. Round-robin reservations combine those queues within the final context limit, normally two chunks per side at the default four-chunk budget. Spare slots can go to a target with more evidence. Debug exposes candidate, selected and strong counts, dimensions and adequacy for every target.

12. **Sufficiency gate.** Synthesis is gated on the actual selected, truncated packet. A comparison needs both targets, the configured minimum of distinct strong texts per target, and at least two supported dimensions on each side. Defaults require one source of at least 100 characters and 12 words with relevant dimensions; the source minimum can be raised to two. Duplicate text does not inflate strong-source counts. Cause questions with an extracted target need target consistency, causal relevance and at least two analytical dimensions. Biography needs evidence for the subject. External result counts or nonempty packets cannot override the comparison gate.

13. **Wikipedia fallback.** Missing comparison sides use host-side search → consistency/overview ranking → fetch of one selected page → normalization into citable evidence for that target. Search snippets remain navigation results and cannot establish sufficiency. If the bounded plan fails, Central returns insufficient evidence without generating a one-sided comparison. Shared pages are fetched once and can retain both independently verified target origins.

14. **Duplicate-action handling.** The normal comparison path performs no ACTION generations. For other requests, a completed Wikipedia search receives its deterministic fetch before any sufficiency decision; an insufficient completed search/fetch does not trigger a second ACTION generation to repeat the search. Executed signatures and retained observations are supplied to ACTION, duplicate-only rounds stop, and rejected repeats report the prior observation plus instructions to fetch, change query or finish. Tool errors remain bounded by the existing action-round limit.

15. **Target-aware citations.** Deterministic validation associates explicit target mentions and headings with their source groups, and handles named comparison table columns. A Điện Biên Phủ paragraph/column citing only A sources fails; a paragraph asserting facts about both sides needs support from both. Validation runs on synthesis and repair. This is section/paragraph association, not a second model or a full entailment checker.

16. **Viewpoints.** `viewpoint_sensitive` is telemetry derived from specific `viewpoint_annotations`. Neutral cited facts pass even when every source contains annotations. Only copied sensitive speech, closely reproduced evaluative claims, or reported opinions presented as consensus require attribution. Repair can attribute the claim or replace it with supplied neutral facts. See [the span-level correction and UI report](central_viewpoints_and_frontend_redesign.md). The deterministic rules do not certify ideological neutrality or exhaustive paraphrase detection.

17. **Repair.** At most one tool-free repair remains. It must remove unsupported claims, add no new names/dates/events, retain both comparison sides, cite each factual paragraph with the corresponding target's aliases, and attribute viewpoints. Persistent grounding, citation, missing-side, missing-similarity/difference or viewpoint failures return insufficient evidence. Valid first syntheses skip repair.

18. **Nguyễn Cao Kỳ regression.** Six candidates reduce to the two correct chunks; irrelevant Nguyễn Cao and Trần Cao Vân sources do not reach synthesis. One fake synthesis call, no repair, visible `[1]` and `[2]`; real Unicode source IDs remain in metadata.

19. **Nhà Trần regression.** The subject is Nhà Trần, the Chinese dynasty page is rejected, late-Trần causal evidence ranks ahead of Hậu-Trần aftermath, cause variants are present, and aftermath alone fails sufficiency. Valid synthesis uses one fake call without repair.

20. **US/VNCH regression.** Analysis identifies the Vietnam War, both actors and defeat. The fixture includes the 1972 campaign, intervention, Lam Sơn 719, a war overview and a limited-war page. Under a three-chunk limit the overview survives and the selection covers multiple analytical dimensions. Valid synthesis uses one fake call without repair.

21. **CMT8 versus Điện Biên Phủ regression.** Candidate origins survive, all five lexical/place collisions are filtered, both target queues contribute strong sources, the packet is grouped A/B, and balance telemetry is populated. Missing or too-short B evidence never enters synthesis. The fallback fixture chooses the battle page, fetches it once, tags it B and reaches sufficiency with one synthesis generation. The low-observation-budget regression confirms that A cannot starve B.

22. **Performance implications.** Normal supported biographies, causes and comparisons use one synthesis generation; a genuine validation failure may add one repair. Comparison fallback adds bounded retrieval I/O, not ACTION generations. More candidates are inspected by deterministic code while the model receives a small packet. This removes the two repeated comparison ACTION generations in the reported failure pattern. No real Qwen latency, GPU throughput or production first-pass success rate was measured.

23. **Files changed.**

    - `app/agents/central_analytical.py`: consistency, dimensions, temporal/viewpoint annotations, coverage selection and sufficiency reporting.
    - `app/agents/central_question.py`: deterministic analytical fields, query planning and target-name handling.
    - `app/agents/central_evidence.py`: independent target queues, bounded packets and provenance.
    - `app/agents/central_agent.py`, `central_state.py`, `central_policy.py`: orchestration, result reuse, selected-packet gates, fallback and telemetry.
    - `app/agents/central_citations.py`, `central_prompt.py`: display map, target-aware checks and synthesis/repair contracts.
    - `app/agents/config.py`, `app/config.py`, `app/main.py`, `.env.example`: validated bounds and application wiring.
    - `app/tools/local_search.py`: opt-in candidate-pool return before final diversity selection.
    - `app/schemas.py`, `app/api/routes.py`: display/provenance fields in API, SSE, persisted sources and debug.
    - `frontend/src/components/RetrievedChunks.jsx`: readable numbered source labels and removal of visible raw IDs.
    - `tests/test_central_consolidated.py`, `tests/test_central_agent.py`, `tests/test_central_quality_hardening.py`: new regressions and updates to old raw-ID/snippet assumptions.
    - `frontend/tests/sourceCitations.test.js`: renders real React components to verify numbering, source counts, numeric-year preservation and debug-only IDs.
    - This report and the historical `docs/central_v2_quality_hardening.md` pointer.

24. **Focused verification.** Final results are recorded below after running the local CPU/fake tests. Coverage includes the four cases, structured parsing, query planning, disambiguation, dimension/overview selection, temporal relevance, target provenance/balance/sufficiency, search/fetch fallback, shared-page reuse, citation/display/numeric safety, viewpoint attribution, bounded repair, frontend rendering and Central isolation.

25. **Full verification.** All Python runs use `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and an empty `CUDA_VISIBLE_DEVICES`. Existing dataset/training tests use their local fixtures and fakes; production datasets, weights, adapters and volumes are not changed. Final counts, compile checks and frontend checks are recorded below.

## Final verification

- Focused Central regressions and isolation suite: **120 passed** (17.11 seconds).
- Full repository Python suite: **677 passed, 1 skipped** (40.85 seconds). The existing skip is `tests/trajectory_dataset/test_notebook_agent_flan_integration.py:48`, because the legacy V1 notebook is absent from this checkout.
- Frontend tests: **10 passed**, including real React rendering of the answer, source drawer and Agent Trace.
- Frontend production build: passed.
- Frontend ESLint: passed.
- `python -m compileall -q app tests scripts training`: passed.
- `git diff --check`: passed under the repository's normal Windows line-ending configuration.

The full run includes **32 added CPU/fake regression cases** in `test_central_consolidated.py`. No live model, retrieval network request or GPU inference was used to establish these results. The frontend production build is a local build only.

## Preserved constraints

- CENTRAL V2 ARCHITECTURE PRESERVED
- CENTRAL V1 ADAPTER NOT REINTRODUCED
- RAW CHUNK IDS NO LONGER USER-VISIBLE
- REAL SOURCE IDS REMAIN IN METADATA AND AGENT TRACE
- COMPARISON CANDIDATES RETAIN TARGET PROVENANCE
- COMPARISON REQUIRES EVIDENCE FOR BOTH TARGETS
- TARGET-AWARE CITATION VALIDATION IS ACTIVE
- WIKIPEDIA FALLBACK USES SEARCH THEN FETCH
- DUPLICATE SEARCH ACTION LOOP IS PREVENTED
- ANALYTICAL RETRIEVAL IS COVERAGE-AWARE
- NO MODEL TRAINING EXECUTED
- NO MODAL COMMAND EXECUTED
- NO GPU INFERENCE EXECUTED

No Qwen weights were downloaded; no real Qwen inference, dataset training, deployment or Modal Volume mutation was performed.
