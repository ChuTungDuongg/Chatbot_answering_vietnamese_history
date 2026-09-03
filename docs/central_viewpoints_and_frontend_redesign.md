# TASK 1 — Central viewpoint fix

1. **Exact root cause.** The previous detector treated any sufficiently long quotation, unscoped first-person wording, or selected loaded language as sensitive. `check_citations` applied that detector to whole answer paragraphs, with a broad paragraph-level attribution exemption. Evidence rendering and the synthesis prompt also imposed a blanket attribution instruction for a sensitive source. The validator did not directly reject a paragraph simply by reading the source boolean; the combination of the overbroad answer detector and source-wide prompt caused the regression. Quoted policy names could fail despite being ordinary factual synthesis.

2. **Old semantics.** A quote in evidence flagged the entire source; a quote in an answer could trigger `unattributed_viewpoint` without establishing that the answer reused a sensitive claim. Citation support and attribution were therefore insufficiently distinguished.

3. **New annotations.** `central_viewpoints.py` records `type`, original-text `start`/`end`, bounded `text`, `reason`, optional `attribution_hint`, and `requires_attribution`. Types cover direct quotes, first-person speech, high-confidence evaluative language, and explicitly reported opinions. A quoted policy label is annotated but does not require attribution. First person is restricted to speech/quotation context. Generic words such as thất bại, chiến thắng, quân đội, chính quyền and đối phương are not loaded markers. The evidence packet recomputes annotations from the exact visible text, including after truncation. `viewpoint_sensitive = bool(viewpoint_annotations)` is telemetry only.

4. **Neutral-fact rule.** Normal supported factual paraphrases need valid citations, even when every cited source contains sensitive annotations. The validator does not use the sensitivity boolean. Existing grounding, dates, citation aliases and comparison-target checks remain separate and unchanged.

5. **Attribution rule.** The validator matches actual answer sentences against attribution-required spans in their cited sources, including close wording and VNCH/full-name normalization. Copied speech, loaded claims and reported opinions presented as consensus require attribution. High-confidence loaded wording introduced by the answer is also checked. An attribution in an unrelated preceding sentence does not exempt a later claim. Matching is deliberately deterministic and conservative; it does not claim exhaustive semantic paraphrase detection or ideological classification.

6. **Repair.** A genuine failure still receives at most one tool-free repair. The prompt receives the offending claim, source excerpt and speaker hint. It can attribute the claim, remove it, or replace it with neutral facts already in the packet. It must not add unsupported facts. Unresolved viewpoint, grounding, year, citation and comparison errors retain fail-closed behavior.

7. **US/VNCH regression.** The fake-runtime production-like test asserts the correct cause/event/actors/outcome analysis, 20 retrieval candidates, 4 selected sources, all 4 sensitive, `evidence_sufficient=true`, and `causal_target_and_dimension_coverage`. Neutral synthesis passes on the first generation: `repair_used=false`, no viewpoint issues, numeric citations, no insufficient-evidence fallback.

8. **Genuine failures.** Unattributed loaded wording and the Chomsky opinion are detected. Explicit attribution passes. Both a neutral supported rewrite and an attributed repair recover in one attempt; repeating the invalid answer still fails closed.

9. **Python verification.** The focused viewpoint/consolidated/quality run passed 122 tests. The complete local suite passed **698 tests, with 1 skip**, in 131.34 seconds. The skip concerns an absent legacy V1 Colab notebook. Tests ran with Hugging Face/Transformers offline and CUDA visibility disabled. Existing Nguyễn Cao Kỳ, Nhà Trần, Bạch Đằng 938, US/VNCH, comparison, retrieval and architecture tests remain passing. Compilation and `git diff --check` also passed.

Changed backend files: `app/agents/central/viewpoints.py`, `central_analytical.py`, `central_evidence.py`, `central_citations.py`, `central_prompt.py`, `central_agent.py`; tests in `tests/test_central_viewpoints.py` and `tests/test_central_quality_hardening.py`.

CENTRAL V2 ARCHITECTURE PRESERVED  
VIEWPOINT SAFETY PRESERVED BUT NO LONGER SOURCE-WIDE  
US/VNCH NEUTRAL SYNTHESIS NO LONGER FAILS CLOSED  
CENTRAL V1 ADAPTER NOT REINTRODUCED

Public mode remains `central`, using `Qwen/Qwen3-8B` by default with an optional adapter. No role-agent delegation, routing redesign, retrieval rollback or model-runtime change was introduced. The production-like result above is verified with fake inference, not a live model run.

# TASK 2 — Frontend redesign

10. **Design direction.** A warm, restrained research conversation with graphite surfaces, terracotta accents, quiet navigation and generous reading space. The existing Sử Việt AI identity is retained. No Claude logos, branding or proprietary assets were used.

11. **Typography.** Self-hosted Inter Variable from `@fontsource-variable/inter`, licensed OFL-1.1, includes Vietnamese glyphs and system fallbacks. Answer text is 16px/1.8 on desktop and 15.5px/1.8 on mobile. Headings, lists, tables, block quotations and citations have consistent spacing.

12. **Tokens.** `src/index.css` owns font stacks, content widths, radii, surfaces, borders, text, accent, semantic colors and shadows. Dark mode uses warm graphite `#242321`, off-white `#eeeae4` and terracotta `#dda080`. The existing theme switch/persistence remains, with a coherent cream/light palette.

13. **Layout.** The desktop reading column is 800px with generous outer padding. User messages have compact right-aligned surfaces; assistant answers remain open document-style prose. A new `EmptyState` integrates the composer and four history prompts. Suggestions populate and focus the existing input. The old monolithic CSS was replaced with shell, sidebar, conversation, composer and sources modules.

14. **Composer.** A large rounded autogrowing textarea, bottom action toolbar, compact unchanged mode choices, attachment control and integrated Send/Stop button. It sits within the empty state and at the conversation bottom after submission. Enter, Shift+Enter and IME behavior are preserved. Browser testing exposed an existing idle reset during first-conversation creation; one UI status assignment now restores processing before the request, keeping Stop visible. API formatting and SSE handling are unchanged.

15. **Citations/sources.** Known display indices become subtle clickable citation controls. Clicking opens and expands the relevant source in the drawer. Known legacy bracketed source IDs are converted for display only; stored messages and source metadata are untouched. Years and existing code/links are preserved. Source count, title, kind, text and original URL remain available; retrieval scores/raw identifiers are absent from normal source cards. The drawer supports focus entry, Tab containment, Escape and return to the citation.

16. **Agent trace/actions/states.** Agent trace is a quiet disclosure, collapsed by default, with wrapping monospace data when expanded. The existing debug gate and payload are preserved. Raw identifiers remain available there. Copy gives temporary “Đã sao chép” feedback. Loading uses restrained dots. Legitimate insufficient-evidence responses use a muted informational panel; system errors retain distinct error styling.

17. **Responsive/accessibility.** Browser tests cover 1440×1000 desktop, 834×1112 tablet and 390×844 mobile. Small screens use an off-canvas sidebar, safe composer padding, wrapping Vietnamese prose, contained tables/trace text, one-column suggestions and a viewport-fitting drawer. Focus-visible styling, button labels and reduced-motion behavior are included. Screenshots of dark/light answers, empty states, sources and insufficient evidence were generated; representative desktop, tablet and mobile images were visually inspected. Chromium emulation does not substitute for testing every physical mobile device.

18. **Frontend files.** Updated `src/App.jsx`, `App.css`, `index.css`, `main.jsx`; components `ChatInput`, `ChatMessage`, `ChatSidebar`, `RetrievedChunks`, `DeveloperTrace`, `StatusIndicator`, `AttachmentTray`; added `EmptyState`, `SourcesDrawer`, `services/citations.js` and five `styles/*.css` modules. Dependency manifests add Inter and Playwright. Tests include `citationFormatting.test.js`, updated `chatModes.test.js`/`sourceCitations.test.js`, `e2e/conversation.spec.js` and `playwright.config.js`. `.gitignore` excludes generated browser results. `services/api.js`, mode values, persistence format, debug payload and backend routes are unchanged.

19. **Frontend verification.** **13 Node tests passed; 9 Playwright tests passed** across all three viewports. Coverage includes numeric/legacy citations, years, source count/titles, debug-only raw-ID visibility, copy, source focus, long source text, empty suggestions, mobile sidebar, textarea growth, IME/Shift+Enter, keyboard mode selection, exact streaming request shape, reload persistence, loading and insufficient evidence. Browser APIs are served by local mocks; no production backend is contacted. Run `npm run test:ui` with locally installed Chrome. Screenshots are generated beneath `frontend/test-results/`.

20. **Build/lint.** `npm run build` and `npm run lint` passed. The production build includes the Vietnamese font subset. All work was local CPU/frontend work; the installed browser was launched with `--disable-gpu`.

RAW CHUNK IDS REMAIN DEBUG-ONLY  
FRONTEND VISUALLY REDESIGNED  
NO MODEL TRAINING EXECUTED  
NO MODAL COMMAND EXECUTED  
NO GPU INFERENCE EXECUTED

No model weights were downloaded, datasets normalized, Modal resources changed or deployment performed.
