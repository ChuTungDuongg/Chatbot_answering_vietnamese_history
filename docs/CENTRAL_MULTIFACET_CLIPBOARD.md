# Central multi-facet grounding and clipboard images

## Task 1 — Central multi-facet / viewpoint fix

The event extractor previously stopped at outcome/verb clauses but did not recognize a trailing coordinated analytical clause. It therefore made `Chiến tranh Việt Nam và hệ quả lâu dài của nó` the event, while the facet detector classified `hệ quả` as `result`. The entity filter then compared otherwise relevant sources against that polluted event. The parser now separates the event and requested facets before any retrieval.

For `Vì sao Mỹ lại thua chiến tranh Việt Nam và hệ quả lâu dài của nó?`:

| Field | Value |
| --- | --- |
| question_type | cause |
| canonical_event | Chiến tranh Việt Nam |
| actors | Mỹ |
| outcome | thất bại |
| requested_facets | cause, consequence |
| raw_event_clause | chiến tranh Việt Nam và hệ quả lâu dài của nó |

Clause boundaries also handle cause + significance and context + method + result. Explicit event years remain attached to the event. An undated overview title does not admit evidence explicitly describing a different battle year.

Each facet gets one primary canonical-event query and a bounded expansion only if its selected excerpts are insufficient. `retrieval_facet` / `retrieval_facets` record why a source was found; `evidence_facets` are independently recomputed from the excerpt actually sent to synthesis. Query labels never confer evidence coverage. Selection rewards uncovered requested facets without imposing rigid source quotas.

All requested facets covered permits full synthesis. Some covered permits a grounded partial answer with an explicit limitation for each missing facet. None covered returns insufficient evidence without a speculative synthesis. The validator detects omitted supported sections and attempted unsupported missing-facet sections. One repair receives all current issues, and validation is recomputed from the repaired answer.

Missing facets alone receive deterministic Wikipedia search using the canonical event plus facet. Results must pass the existing target checks before fetching; snippets do not become citable evidence. Existing good evidence is retained. `suspected_filter_collapse` exposes an all/mostly-identical entity rejection when candidates existed; the bounded canonical facet fallback is the available recovery, with no filter bypass or extra retry loop.

For general analytical questions, equally relevant neutral excerpts receive preference over quote-dense excerpts. Compaction scores the actual sentence/quote windows and avoids unnecessarily adjoining a sensitive quote to a neutral window. Explicit requests for viewpoints retain their viewpoint evidence. There is no extra model or embedding call.

Genuine direct quotations remain span-validated. Existing conservative matching thresholds remain: quoted overlap of at least five normalized words, at least 90% of the answer quote and three distinctive words; or eight contiguous copied words with four distinctive words. The existing complete short first-person quote exception remains. Neutral synthesis does not inherit a source-level viewpoint violation.

When a matched annotation names a speaker, attribution must name that speaker and include a reporting cue. Generic `một số học giả` or `theo nguồn` cannot replace Noam Chomsky. Synthesis receives compact annotated spans and speaker hints; repair receives the affected claim, exact matched span, required hint, and all outstanding issues. It can name the speaker or replace the unnecessary quote with supported neutral prose. `repair_viewpoint_action` records `attribute` or `neutralize` after successful revalidation.

The CPU/fake regressions demonstrate:

- Exact multi-facet production question: two primary history calls, two expansions skipped, one synthesis, no repair, both facets covered.
- A 25-row valid event/related-page fixture is retained; unrelated evidence is rejected.
- Missing consequence: one consequence expansion, one canonical Wikipedia search/fetch attempt, supported partial answer if still missing.
- Canonical Wikipedia recovery can supply only the missing consequence facet with one synthesis and no action-model generation.
- Simple US-loss question with strong neutral evidence: one history call, one synthesis, no repair.
- Both supplied Chomsky quotes are detected without specific attribution; exact attribution and neutral replacement each pass after one bounded repair.

These are call-count assertions, not production latency promises.

## Task 2 — Clipboard image / OCR

The existing architecture is multipart upload → image validation / existing Tesseract OCR (or PDF text extraction with OCR fallback) → temporary conversation chunks and existing retrieval embeddings → `search_uploaded_documents` → Central's evidence packet. It already stores conversation/owner-scoped extracted text. This patch reuses that path and its existing 20 MB limit. The existing five-file cap now also bounds total conversation attachments, including a transaction-protected backend check.

The composer reads image file items from `clipboardData`, generates MIME-derived `clipboard-image-N` names, and calls the same `handleFilesSelected` used by picker and drop. PNG, JPEG and WebP are allowed; SVG is rejected. Text-only paste is untouched. Mixed image/text paste retains native text insertion, including when uploading creates the first conversation.

Previews use short-lived object URLs, with compact thumbnails, filename, upload state, and removal. Removing an in-flight upload removes its eventual server record before sending can resume. Queued removed files are skipped. URLs are released on removal, conversation changes, deletion, and unmount. Failed OCR remains visible and removable, and failed IDs cannot be sent as ready attachments. The tray scrolls horizontally inside a bounded height; the existing message scroller and composer docking remain intact.

Chat JSON contains attachment UUID references and optional user text, never image bytes/base64. Empty text is accepted only with valid ready attachments belonging to the current owner/conversation. The backend uses `Phân tích nội dung ảnh đính kèm.` internally when needed, while storing/displaying an empty user text plus the actual attachment references. Existing text-only clients can omit the new optional field.

The optional OCR provider interface defaults to the existing Tesseract implementation. Tests inject a fake provider; there is no OCR installation/download or extra LLM. Metadata preserves attachment ID, sanitized filename, MIME, upload origin, OCR success/error, provider, optional confidence, extracted character count, and upload OCR duration. Central records attachment counts/types, clipboard count, document tool calls and bounded per-attachment OCR metadata. OCR duration is the cached upload processing time, not a second OCR run during chat. The evidence packet explicitly labels extracted text as attachment evidence that may contain OCR errors, not user instructions. IDs remain host-scoped and absent from model-controlled tool arguments.

Tesseract was not found on this machine's PATH. Real screenshot OCR accuracy was therefore **not verified**. Empty/unreadable images and unavailable-engine errors are tested with fake providers and produce controlled failures. Previews are intentionally session-local; after reload, existing records show filenames/icons because the existing backend does not serve original image bytes.

### Local manual check

1. For the verified browser/fake workflow, run `npm run test:ui -- e2e/clipboard.spec.js` from `frontend`. Its Vite server uses `127.0.0.1:4173`, mocks every API route, and starts no backend/model runtime.
2. For a later real OCR check, use an already available, authorized **local** backend that has working OCR and does not start cloud/GPU services. This verification did not launch such a backend. Do not use the repository's remote frontend endpoint for this check.
3. In a PowerShell terminal in `frontend`, set `$env:VITE_API_BASE_URL='http://127.0.0.1:8000'` to that local backend's actual address, then run `npm run dev -- --host 127.0.0.1`. This environment value overrides the saved frontend configuration.
4. Capture readable text with Win+Shift+S, focus the composer and press Ctrl+V. Confirm thumbnail, filename, processing status and removal.
5. Select **Central Agent**, type `Đọc nội dung trong ảnh này.` and send. Inspect the attachment source and OCR metadata under Agent Trace / Performance. Then repeat with no text and confirm the user message shows the attachment rather than the internal default instruction.
6. Repeat after two completed turns, with a long response, after resizing, and on mobile. Scroll older messages and verify the composer stays docked. If OCR is unavailable, expect the controlled failure; do not infer that the text was read.

### Changed files

- Central parsing, selection, validation and telemetry: `app/agents/central/question.py`, `central_facets.py`, `central_analytical.py`, `central_compaction.py`, `central_evidence.py`, `central_viewpoints.py`, `central_prompt.py`, `central_agent.py`.
- Shared attachment/API path: `app/chat/attachments.py`, `app/chat/store.py`, `app/api/conversations.py`, `app/api/routes.py`, `app/schemas.py`, `app/tools/attachment_search.py`, `app/tools/registry.py`.
- Composer/upload UI: `frontend/src/App.jsx`, `components/ChatInput.jsx`, `components/AttachmentTray.jsx`, `components/ChatMessage.jsx`, `services/attachments.js`, `services/api.js`, `styles/composer.css`.
- Regressions: `tests/test_central_multifacet.py`, `tests/test_clipboard_ocr.py`, three existing Central viewpoint/evidence test files, `frontend/tests/attachments.test.js`, `frontend/e2e/clipboard.spec.js`.

No changes to Central mode, base-model selection, optional adapter support, role-agent routing, citation display, comparison resolution, BrandMark, theme or viewport-height ownership. No training, model download, Modal command, GPU inference, OCR-model download or dataset normalization was executed.

### Verification results

| Check | Result |
| --- | --- |
| New multi-facet and fake OCR cases | 26 passed |
| Focused multi-facet, OCR and viewpoint regressions | 71 passed |
| Full Python suite | 853 passed, 1 pre-existing skip (absent legacy V1 notebook) |
| Frontend unit tests | 19 passed |
| Playwright, installed Chrome with GPU disabled | 39 passed across desktop, tablet and mobile |
| Frontend build / lint | Passed |
| `python -m compileall -q app tests` | Passed |
| `git diff --check` | Passed |

The native desktop mixed text/image Ctrl+V test passed. Browser tests use local API mocks; backend multipart tests invoke the real ASGI router with fake OCR and embeddings. No production latency or real OCR-quality measurement is claimed.
