# Central V2 quality and performance hardening

This document records the earlier biography-focused pass. Current behavior and verification are in [Central V2 consolidated hardening](central_v2_consolidated_hardening.md). In particular, the former expansion to visible real chunk IDs described below has been replaced by numeric display citations.

Central retains PREPARE → INITIAL_GROUNDING → optional ACTION / TOOL_EXECUTION → SYNTHESIS → optional QUALITY_REPAIR → FINAL. The public mode is `central`; the Qwen3-8B base, optional future Central V2 adapter, Hermes function calling, host-side `search_history`, and future request-policy hook remain intact. This work changes deterministic selection and answer validation around that state machine.

## Confirmed causes

- Question analysis had no biography category or subject extraction.
- Central normalized every returned row within its result/character budget, without an entity consistency gate or score-noise filter. Retrieval diversity could therefore bring other people's articles into synthesis.
- The citation regex accepted ASCII IDs only. Even an accurately copied Unicode chunk ID could be treated as missing, and the prompt asked the model to copy those long IDs.
- Quality repair used the full configured 1,024-token maximum regardless of original answer size. A minimum word-count check could also trigger another generation for a compact supported answer.
- Citation syntax checks did not check paragraph coverage or unsupported high-risk factual tokens. Repair validation was incomplete for non-analytical requests.

The existing tool normalizer already omitted dense/BM25/RRF scores from observations. The new packet further removes tool-call envelopes, source-ID copying, and rejected evidence from the synthesis input; it does not claim those scores were previously dumped into that input.

## Biography and evidence selection

`biography` is canonical and sets `analytical=true`. Reusable Vietnamese/ASCII-folded patterns recognize identity, biography, life/career, activities, roles, and offices. NFC-normalized capture spans preserve the original accented subject in `analysis.subject`, Central debug, request debug, and request telemetry. Accent-sensitive exclusions avoid confusing Trần/Trận, Hồ/Họ, and Đặng/Đảng.

Selection runs before observation truncation, separately per retrieval batch, then enforces the final biography bound across accumulated evidence. Exact normalized title/page-title/URL-title matches rank first. Once at least two retained exact-title rows exist, unrelated entity collisions are excluded. Other titles can remain when their text names the full subject or they share the exact subject's page identity. Without matching evidence, normal retrieval order remains the fallback. Other question types do not receive the biography title gate.

The production retriever stores `CrossEncoder.predict()` output directly as `reranker_score`; its min-max/fused retrieval score is separate. Absolute probability calibration is therefore not assumed. Default noise filtering requires at least two higher-scoring anchors and a downward adjacent gap covering at least 75% of that batch's score range. It is invariant to positive rescaling and shifting of raw scores, works with negative logits, and preserves unscored candidates. It never compares scores across separate queries or tools. Flat batches and a lone high outlier do not trigger this rule. An optional absolute floor applies only in explicitly configured probability mode, with a strong top result and scores in [0, 1]; the top result is retained.

For the Nguyễn Cao Kỳ fixture, scores 0.999 and 0.86 survive; Trần Cao Vân at 0.0033 and Nguyễn Cao at 0.055 do not. The synthesis packet contains two sources, with no attempt to fill spare slots. Biography context defaults to at most four sources.

## Evidence and citation contract

Each host-side packet item has `alias`, `real_source_id`, `title`, `source_kind`, and bounded `text`. The model sees only `[S1]`, title, source kind, and text. Aliases follow selected evidence order, deduplicate real source IDs, and exist only within the request. Original retrieval metadata stays outside the model input.

Synthesis instructions require evidence-supported historical paragraphs to carry citations, including a citation on a compact factual answer. They prohibit introducing unsupported people, dates, events, states, titles, or offices and require acknowledging unestablished requested information rather than filling gaps from model memory. Biography guidance suggests identity, life dates, offices, career stages, and later life only where supported; it does not require an “Ý nghĩa” section.

- `[S1]` and `[S2]` are valid only when supplied.
- `[S99]`, `[source_1]`, `[source]`, and `[1]` are invalid.
- `[938]` and `[1945]` remain prose, never aliases.
- Backward compatibility accepts an exact selected real chunk ID, including Unicode, and canonicalizes it to the corresponding alias before validation.
- Unambiguous case/spacing variants, doubled brackets around a supplied alias, and comma/semicolon lists consisting entirely of supplied aliases normalize deterministically. Missing or unknown citations are never guessed.
- Expansion uses one substitution pass to restore `[real_chunk_id]` for the frontend. Source lists deduplicate repeated references while preserving citation order.

## Grounding and repair

The lightweight risk guard compares years and conservative person/state/dynasty/event/office patterns with the current question and model-visible selected evidence. IDs, filtered evidence, and conversation history cannot establish support. It ignores generic functional words and recognizable year durations. The Bạch Đằng fixture catches unsupported “Lê Đại Hành” and “Đại Cồ Việt” as `unsupported_evidence_claim` and accepts the corrected cited Ngô Quyền answer.

This is a token-presence risk signal, not sentence entailment or a comprehensive fact checker. It can miss names outside its patterns, incorrect relationships between supported entities, or a sentence citing the wrong one of several available chunks. A passing result must not be interpreted as proof of historical correctness.

Valid first answers finish immediately. Recoverable citation formatting is normalized without another model call. Missing citations, uncited factual paragraphs, invalid aliases, or grounding risks may receive one tool-free rewrite of the existing answer against the same evidence. Repair does not select or call tools. Its budget is:

```text
min(configured maximum, max(configured minimum, original output tokens + margin))
```

Defaults are maximum 1,024, minimum 192, margin 96. If a backend omits token counts, a bounded character-based estimate substitutes for the original token count. ACTION remains 256 tokens. Length alone no longer triggers quality repair; existing comparison-content checks remain.

The repaired answer passes the same citation/grounding checks. Persisting citation or grounding failures return the existing insufficient-evidence response, with quality failures retained in debug, and never cause a third generation. `repair_used` records an executed repair generation even when its output fails validation.

Fake-runtime results demonstrate one Central call and no repair for a valid grounded biography or compact significance answer; missing citations permit at most two total calls. This removes the second generation in those passing paths (two calls to one). Production latency and first-pass success rate have not been measured because real model execution was excluded.

## Configuration and telemetry

All new settings are wired through Settings and `app.main` into CentralAgentConfig; example environment values are documented in `.env.example`.

| Setting | Default |
| --- | --- |
| `CENTRAL_BIOGRAPHY_MAX_SOURCES` | 4 |
| `CENTRAL_BIOGRAPHY_MIN_EXACT_HITS` | 2 |
| `CENTRAL_RERANKER_TAIL_GAP_RATIO` | 0.75 |
| `CENTRAL_RERANKER_SCORE_MODE` | raw |
| `CENTRAL_RERANKER_SCORE_FLOOR` | unset |
| `CENTRAL_RERANKER_STRONG_SCORE` | 0.5, probability mode only |
| `CENTRAL_REPAIR_MIN_NEW_TOKENS` | 192 |
| `CENTRAL_REPAIR_TOKEN_MARGIN` | 96 |

Debug/provenance/performance include candidate counts before/after filtering, filtered count/reason counts, biography entity/exact-title hits, evidence characters/source count, alias map, unsupported names/years, repair reason/usage/avoidance reason/budget, and final quality issues. Evidence size is recorded before synthesis. Central debug also retains per-batch filter events and per-generation grounding-risk checks; request telemetry exposes these summary fields through `central_quality`.

## Files changed

- `app/agents/central/question.py`: biography analysis and subject extraction; remove length-only repair trigger.
- `app/agents/central/evidence.py`: new deterministic selection and synthesis packet.
- `app/agents/central/citations.py`: new alias checking, normalization, paragraph coverage, and expansion.
- `app/agents/central/grounding.py`: new conservative factual-token risk checks.
- `app/agents/central/agent.py`: integrate selection, packets, validation, bounded repair, and telemetry.
- `app/agents/central/state.py`: request-local filter/repair/risk debug state.
- `app/agents/central/prompt.py`: synthesis, biography, and dedicated repair contracts.
- `app/agents/config.py`, `app/config.py`, `app/main.py`, `.env.example`: validated configuration and wiring.
- `app/telemetry.py`, `app/api/routes.py`: quality telemetry and subject visibility.
- `tests/test_central_quality_hardening.py`: fake-runtime regressions and deterministic contracts.
- `tests/test_central_agent.py`: update packet/budget expectations, ground the numeric-year fixture, and allow enough setup time in the generation-timeout test.
- `docs/central_v2_quality_hardening.md`: this report.

## Local verification

- Focused Central suite: **88 passed**.
- Full repository suite: **645 passed, 1 skipped** (43.51 seconds). The existing skip is the absent legacy V1 Colab notebook in `test_notebook_agent_flan_integration.py`.
- Frontend tests: **9 passed**.
- `python -m compileall -q app tests scripts training`: passed.
- `git diff --check`: passed with the repository's normal Windows line-ending configuration.

Python tests used `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and an empty `CUDA_VISIBLE_DEVICES`. Model/runtime fixtures are local fakes; dataset tests use their existing fixture/temp paths. No production Central V2 dataset rebuild was performed.

CENTRAL V2 ARCHITECTURE PRESERVED  
CENTRAL V1 ADAPTER NOT REINTRODUCED  
NO MODEL TRAINING EXECUTED  
NO QWEN WEIGHTS DOWNLOADED  
NO MODAL COMMAND EXECUTED  
NO GPU INFERENCE EXECUTED
