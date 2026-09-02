# Central V2 runtime generalization

This change prepares the existing host policy for a future Central V2 adapter. It adds no benchmark, evaluation CLI, dataset, aggregate metrics, model comparison, training or inference job.

## Findings and changes

- Shallow synthesis had no explicit depth contract or answer-coverage check. Its existing 1,536-token ceiling was adequate; raising it would not fix a natural early stop. Depth now follows canonical question type, scope and selected evidence. Simple facts remain concise; developed broad analysis targets several distinct causal mechanisms, with flexible length and explicit limitations when evidence is sparse.
- Equivalent questions previously retained raw wording in fallback queries and excerpt scoring. Cause markers and outcome grammar now capture arbitrary actor names, normalize existing abbreviation metadata, and generate consistent semantic queries. `CentralQuestionAnalysis` carries depth and viewpoint intent alongside existing entity, facet, relation, time and granularity fields. Runtime and history compaction reuse that analysis.
- Selection uses target consistency, excerpt strength, actor/facet coverage, factual overview usefulness, source role, sensitive spans and retrieval rank. A neutral excerpt covering the same dimension can replace a sensitive one even when the sensitive span occupies less than 25% of the excerpt or only one dimension overlaps. Unique viewpoint evidence remains eligible. Historiography supplements factual evidence instead of displacing it by default.
- Broad-analysis candidates receive their synthesis-sized excerpts before selection costs are computed. `strong_evidence_dimensions` requires causal propositions in those excerpts; headers, stale metadata and discarded page material cannot establish coverage. The same detector builds the compact answer plan and checks expressed dimensions. Defaults flag fewer than three expressed dimensions when at least four are strongly supported; both thresholds are configurable through Central settings.
- Missing lexical target matches are retained as unconfirmed candidates and downranked. They cannot establish sufficiency or enter synthesis without verified target association. Explicit wrong entities/types, incompatible administrative levels and invalid citations remain rejected. A likely collapse permits one additional canonical `search_history` query with all normal checks intact; it never promotes rejected material.
- Sufficiency remains capability-specific: factual evidence, biography identity, related-entity coverage, causal breadth, separate requested facets, balanced comparisons, and administrative freshness/granularity. Existing partial facet and unresolved relationship behavior remains intact. Multi-actor plans also identify unsupported actor scopes explicitly.
- Direct copying now requires quotation overlap or a distinctive contiguous copy with at least 80% overlap. The existing exact short first-person-quotation guard remains. Reformulated sensitive propositions use `viewpoint_paraphrase`, including unknown-speaker quotations. Close factual support in a cited neutral sentence can clear attribution requirements; quotes, loaded language and opposing negation cannot use that exemption.
- Every viewpoint repair receives its claim, sensitive span, issue type, speaker hint and concrete action. A missing hint selects `remove_or_neutralize`. One isolated optional sentence may be removed when several supported causes remain; the full answer is revalidated and unsuccessful removal is reverted. Otherwise one full rewrite remains the limit. Shallow answers receive the existing full repair ceiling and supported plan. Issue fingerprints and counts report whether repair made progress; unchanged failures remain failures without another rewrite.
- Synthesis composes grounding, depth, comparison, biography/relation, facet and viewpoint contracts as needed. No historical conclusions were added to system policy. Per-request validation diagnostics are not an evaluation or metrics framework.

## Compatibility and files

The Central state machine, deterministic grounding, compaction, citation aliases/display numbering and current frontend remain. No OCR, clipboard, attachment implementation or attachment tests were edited. Qwen/Qwen3-8B still defaults to no adapter; the existing settings/runtime accept a future `/artifacts/adapters/central-v2` adapter with base-model validation. Central V1 was not reintroduced.

Production changes:

- `app/agents/central/depth.py`, `central_repair.py`: depth, evidence plan, coverage validation and safe repair helpers.
- `app/agents/central/question.py`, `central_compaction.py`, `central_facets.py`: canonical semantics, actor grammar and reused intent.
- `app/agents/central/analytical.py`, `central_evidence.py`: selection, excerpt roles, uncertainty and sufficiency.
- `app/agents/central/viewpoints.py`, `central_citations.py`: quote/paraphrase checks, neutral support and repair actions.
- `app/agents/central/prompt.py`, `central_agent.py`: composed instructions, recovery and validation integration.
- `app/agents/config.py`, `app/config.py`, `app/main.py`: configurable coverage thresholds.

Regression changes: `tests/test_central_answer_depth.py`; two existing viewpoint fixtures now retain developed supported causes after repair, and the packet-field assertion includes `source_role`. No benchmark artifacts were added.

## Validation and limits

Local CPU/fake tests only, with model downloads disabled and CUDA hidden. Focused regression run: **40 passed**. Broader Central and related regression run: **395 passed** (including the then-current 39 focused cases; the additional actor-scope case passed in the final focused run). The suite covers the named production regressions plus unrelated/synthetic cause subjects and actor pairs. `python -m compileall -q app tests` and `git diff --check` passed. Frontend schemas and frontend files were unchanged, so frontend tests were not rerun.

Final production-diff review found no added historical-question branches. The only newly displayed proper names in added production lines are the unchanged pre-existing actor abbreviation entries, moved into shared metadata. OCR/attachment implementation and test paths have no diff.

These are conservative surface heuristics, not semantic entailment. Unusual paraphrases, implied actors, source-role ambiguity and complex sentence structure may still require explicit attribution, qualified answers or the single full repair. A cited neutral sentence is checked for close factual correspondence; source independence is not established bibliographically. No live Qwen output, production latency, adapter quality or word-count guarantee was measured. Those checks remain deferred until authorized after adapter training.
