# TASK 1 — Genève vs Paris fix

## Root cause and target resolution

The comparison parser split the two conjuncts and cleaned punctuation but did not resolve an omitted entity head. It returned `Hiệp định Genève` and `Paris`. Retrieval then searched the bare second target, and Wikipedia selection considered the city title an exact match. Generic economic/international vocabulary could make that wrong entity look adequate. The defect began with the target semantics, not Qwen generation.

`central_targets.py` now separates raw/display, normalized, canonical, and expected entity type. A small historical-head grammar proposes shared heads; a cached title index over the already loaded local corpus confirms canonical identities and optional year suffixes. Dates are never supplied from a handwritten event table. Conflicting dated titles remain unresolved.

Ambiguous person/event ellipsis, such as `chiến dịch Điện Biên Phủ và Hồ Chí Minh`, requires a confirmed corpus title before inheriting the campaign head. Short agreement/dynasty ellipses can form a typed search hypothesis when the catalog is unavailable; telemetry marks it as requiring evidence confirmation. That hypothesis cannot become strong evidence without matching the target and type. Queries without a shared head, including Nguyễn Huệ/Gia Long and Paris/London, stay unchanged.

For the exact regression fixture:

| User/display target | Normalized target | Corpus canonical target | Type |
| --- | --- | --- | --- |
| Hiệp định Genève | Hiệp định Genève | Hiệp định Genève 1954 | agreement |
| Paris | Hiệp định Paris | Hiệp định Paris 1973 | agreement |

Resolution runs before initial tool calls. Telemetry includes `comparison_targets_raw`, `comparison_targets_normalized`, `comparison_canonical_targets`, `comparison_target_entity_types`, and `target_resolution_events`, including confirmation status and inherited head.

## Retrieval, sufficiency, and synthesis

Each target retains an independent adaptive query plan. Its canonical query runs first; the expanded query is skipped only when selected excerpts supply adequate target-specific coverage and diversity. Both targets must be retrieved. The title index is cached against the loaded corpus snapshot, without invoking model loaders.

Wikipedia hits are filtered for canonical target and entity type, then ranked by canonical overview, relevant time/causal evidence where applicable, and retrieval rank. Search snippets guide fetching and cannot establish sufficiency. Paris city, PSG, Paris By Night, and Commune pages cannot stand in for the agreement. The exact fallback fixture fetches the agreement result and never page 2584.

Comparison sufficiency re-annotates each selected excerpt for each claimed target before counting strong sources or dimensions. Incorrect types or canonical identities count as no strong evidence. Shared historical overview sources may still support both targets when both event passages survive compaction; a title naming just one event does not erase the other passage. City/artifact exclusions remain in force.

The synthesis contract explicitly calls for context/objectives, main content, results, historical significance, and recognizable “Điểm giống nhau” / “Điểm khác nhau” sections. Its suggested 350–650 words is conditional on available evidence. Exhausting retrieval with one target missing produces a deterministic qualified limitation naming the missing target, without inventing the second side.

## Repair and performance

The previous repair prompt already joined all detected issues. However, `repair_reason` exposed only the first issue, and the final failure list omitted persistent missing-significance and missing-explanation issues. The patch records `repair_reasons` as the complete list and passes that list to one bounded quality repair. Those structural issues now remain blocking after repair. Citation-only normalization/alignment recovery stays intact.

Citation, target coverage, similarity, difference, significance, named-claim, year, viewpoint, and administrative-premise checks recompute from the current answer. Historical diagnostic entries remain separate from current issue fields.

The exact local fixture performs **2 search_history calls + 1 synthesis**, skips both expanded queries, uses no Wikipedia call or repair, and finishes with no validation failure. The provided failed production trace had 6 tool calls and 2 model calls. These are architecture/call-count results from fake tests, not a measured production latency claim.

## RECENT / ADMINISTRATIVE-LEVEL GROUNDING

The additional query `Vì sao Việt Nam lại bắt đầu sáp nhập tỉnh năm 2025?` previously had no parsed subject/event and could fall through to generic usable-evidence sufficiency. Commune/district arrangements and implementation lists then substituted for provincial causes; the missing event also allowed unrelated recent conversation turns into context.

The parser now records a structured administrative level, time scope, reform event, Việt Nam subject when explicit, freshness requirement/reason, and a premise-validation flag. Compound unit names are consumed before individual words so “thành phố thuộc tỉnh” stays district-level and “thị xã” does not become commune-level. “Xã hội”, “tình hình”, and “Đảng” do not spuriously create unit/currentness matches.

Freshness requires reform/policy context plus an explicit date in the current-year-minus-two through current-year-plus-one window, or contemporary wording. Ordinary 938/1945 historical questions keep their existing path. Tests pin the policy clock to September 2026.

Core sufficiency requires a citable, sufficiently long selected excerpt about the requested level and time. Cause questions additionally require a policy objective/cause tied to that level and phase; an immediately following objective can inherit explicit preceding context. Earlier objectives elsewhere in the page and lists of mergers cannot establish current causes. Local content needs publication/update/revision metadata at least as recent as the requested year. Fetched live content must still establish the explicit requested time and level; fetching an old lower-level article is insufficient. Undated contemporary questions need date metadata rather than assuming a fetch alone proves currentness.

Local retrieval remains first and skips its secondary query once this core is sufficient. Otherwise bounded deterministic Wikipedia search/fetch runs, followed by web search/fetch only when configured as usable. `local` and `local-only` providers expose no web-search tool. Search-result titles may route a fetch but never supply missing granularity in the fetched excerpt. Exhausted sources produce a qualified limitation, with no action or synthesis generation from insufficient evidence.

The exact fresh local fixture keeps **1 local search + 1 synthesis, no repair**, and excludes unrelated history. Lower-level-only fixtures remain insufficient. The prompt requests an evidence-qualified distinction between an earlier general program and a specific phase. A fresh answer validator rejects unsupported “first/began” assertions, while allowing source-supported earlier district and later provincial phases to be described separately.

Telemetry includes administrative match/mismatch counts, observed levels, cause/core counts, freshness and time fields, premise status, and current-source fallback usage/reason. No policy conclusion is hardcoded.

## Backend files

- Added `app/agents/central/targets.py` and `app/agents/central/administration.py`.
- Updated `app/agents/central/question.py`: structured semantics and bounded query planning.
- Updated `app/tools/local_search.py`: cached loaded-corpus title resolution.
- Updated `app/agents/central/analytical.py`: target/type/level/time/cause sufficiency.
- Updated `app/agents/central/compaction.py`: prefer relevant administrative excerpts.
- Updated `app/agents/central/evidence.py`: revalidate comparison provenance after compaction.
- Updated `app/agents/central/agent.py`: resolution, deterministic fallback, all-issue repair telemetry, premise checks, qualified limitations.
- Updated `app/agents/central/prompt.py`: comparison synthesis contract.
- Added `tests/test_central_comparison_resolution.py` and `tests/test_central_administration.py`.

# TASK 2 — Cherry blossom branding

The old white temple/star inside a rounded square lived in `LogoMark.jsx`. It was used by the sidebar identity, collapsed/mobile header, empty state, and assistant avatar, including loading states. A separate old temple mark lived in `public/favicon.svg`.

`BrandMark` replaces it with five rounded, subtly notched petals at 72-degree intervals and a small center, using a 64×64 SVG view box. It has no raster dependency, emoji, gradient, or background square. `size` and `className` are supported; an optional `label` makes standalone usage an accessible image. Next to the visible product name it is decorative (`aria-hidden`). The product name remains Sử Việt AI.

Dark mode uses peach petals `#dda08c` and a cream/gold center `#f0d9a8`. Light mode uses deeper coral `#b65e4f` and muted gold `#ac7b3d`. Both come from CSS variables on the same component. The favicon repeats the same geometry and adapts via `prefers-color-scheme`. No raster build tooling was added.

## Frontend files

- Added `frontend/src/components/BrandMark.jsx`; deleted `LogoMark.jsx`.
- Replaced imports/usages in `frontend/src/App.jsx`, `components/ChatSidebar.jsx`, `components/ChatMessage.jsx`, and `components/EmptyState.jsx`.
- Updated `frontend/src/index.css` with blossom colors/fills; removed obsolete logo-only sizing rules from `styles/sidebar.css`, `styles/conversation.css`, and `styles/shell.css`. Existing layout/overflow rules remain intact.
- Replaced `frontend/public/favicon.svg`.
- Added `frontend/tests/brandMark.test.js` and `frontend/e2e/brandMark.spec.js`.

## Verification

- Full CPU/fake Python suite: **826 passed, 1 skipped**. The skip is a pre-existing integration test for an absent legacy V1 notebook.
- The full suite includes all 46 new comparison/administrative cases and existing Central biography, relationship, battle, dynasty-decline, war-cause, CMT8, comparison, viewpoint, citation, and history-gating regressions.
- Frontend Node tests: **16 passed**.
- Playwright: **21 passed**, across desktop, tablet, and mobile, with the API mocked locally and browser GPU disabled.
- Blossom inspected at 16, 20, 24, 32, and 40 px in both themes. Browser checks verify dimensions, five petals, center, no SVG clipping, and over 3:1 petal/center contrast against the app background. Screenshots also confirm sidebar and assistant-avatar integration.
- Existing third-turn loading/completion, five-turn conversation, long response, resize, scroll-follow/scrollback, citation, source drawer, and Agent Trace browser cases pass.
- `npm run build`, `npm run lint`, `python -m compileall -q app tests`, and `git diff --check`: passed.

The code keeps Qwen/Qwen3-8B as the base default, optional future adapter support, structured tool calling, deterministic grounding, adaptive retrieval, evidence compaction, comparison balancing/provenance, span-based viewpoints, relationship handling, and the current UI shell. No API schema was changed.

"HIỆP ĐỊNH GENÈVE VÀ PARIS" RESOLVES TO TWO AGREEMENTS
BARE PARIS CITY IS NOT ACCEPTED AS PARIS-AGREEMENT EVIDENCE
COMPARISON SUFFICIENCY REQUIRES CANONICAL TARGET CONSISTENCY
REPAIR RECEIVES ALL OUTSTANDING QUALITY ISSUES
PROVINCE-LEVEL QUESTIONS NO LONGER PASS ON COMMUNE-LEVEL EVIDENCE
RECENT POLICY QUESTIONS REQUIRE FRESHNESS-AWARE GROUNDING
STYLIZED CHERRY-BLOSSOM BRAND MARK REPLACES THE OLD LOGO
CENTRAL V2 ARCHITECTURE PRESERVED
CENTRAL V1 ADAPTER NOT REINTRODUCED
NO MODAL COMMAND EXECUTED
NO GPU INFERENCE EXECUTED

No model training, model-weight download, deployment, or live model inference was executed.
