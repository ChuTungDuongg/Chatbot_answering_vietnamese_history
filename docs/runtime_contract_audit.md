# Active Qwen3 runtime contract audit

## Root cause

The Bạch Đằng regression was a combination of two runtime-contract defects:

1. Retrieval ranked a highly relevant `Trận Bạch Đằng (938)` chunk first, but the
   Evidence runtime exposed only `text[:1800]`.  The relevant `Ý nghĩa` section is
   later in that same chunk, after the tactical discussion.  The Evidence adapter
   therefore saw the “mưu kế” portion but could not see the answer-bearing passage.
2. The selected contexts then entered `RAGGenerator.answer_from_retrieval`, which
   rendered the Qwen2.5-era four-section prompt and could invoke legacy guards,
   repair, or three additional structured-section generations.  This differs from
   the History Qwen3 SFT distribution.

The corpus is not missing the requested fact.  Local retrieval returned the
answer-bearing chunk as rank 1 with reranker score approximately `0.9994` and final
retrieval score approximately `1.0896`.

## Canonical History SFT contract

Audit source: `datasets/history_answerer/train.jsonl` and
`training/history_answerer/*`.

- 994 rows: 200 `grounded_qa`, 644 `noisy_context`, 100
  `insufficient_context`, and 50 `false_premise`.
- Every row has exactly two messages with roles `user`, `assistant`; there is no
  History system message and no conversation-history message.
- User structure is exactly `Câu hỏi:` followed by `Tài liệu tham khảo:` and one
  or more `[source_id] title\ntext` blocks.
- Assistant structure is exactly `Nguồn được dùng:` followed by `Trả lời:`.
- Runtime citations must be a subset of IDs visible in the current user message.
- A structural replay rebuilt all 994 user prompts byte-for-byte.

Observed full-dataset prompt distribution:

| Type | Rows | Evidence count p50/p95 | User chars p50/p95 |
| --- | ---: | ---: | ---: |
| `grounded_qa` | 200 | 1 / 1 | 3,069 / 3,195 |
| `noisy_context` | 644 | 3 / 3 | 9,027 / 9,350 |
| `insufficient_context` | 100 | 2 / 2 | 6,072 / 6,348 |
| `false_premise` | 50 | 3 / 3 | 9,011 / 9,331 |

Evidence SFT uses the same `EVIDENCE_AGENT_SYSTEM`, JSON request, and canonical
`EvidenceModelOutput` as runtime.  Its raw candidate text is typically about 2.9k
characters per item and about 13k characters at p95 per row.  Runtime now keeps an
approximately 14k-character total candidate budget, caps the visible pool at eight
ranked candidates, and chooses only extractive question-relevant windows when an
item must be shortened.

Research runtime uses the same `RESEARCH_AGENT_SYSTEM`, `ResearchPolicyState`,
tool definitions, sorted JSON serialization, and action schema as Research SFT.
It intentionally omits the optional training-only `trajectory_class` label because
that gold label is not causally available at inference time; injecting a guessed
`false_premise` or `no_tool` label would leak supervision and can suppress required
search.  The question, retrieval question, observations, limits, tools, and observed
evidence IDs retain the canonical field structure.

The active builder lives in `app/agents/history_contract.py`.  It deliberately
does not import or reuse `PromptBuilder`.

## Active path before repair

```text
Research LoRA
  -> tools / HybridRetriever
  -> Evidence LoRA
  -> HistoryAnswererAgent
  -> RAGGenerator.answer_from_retrieval
  -> Qwen2.5-era PromptBuilder
  -> History LoRA used as a generic generate backend
  -> AnswerGuards
  -> optional repair or three-section expansion
  -> response
```

## Active path after repair

```text
Research LoRA
  -> tools / HybridRetriever
  -> Evidence LoRA
  -> selected extractive textual evidence
  -> canonical History SFT message builder
  -> History LoRA (one generation)
  -> canonical citation validation
  -> response + provenance metadata
```

The History role receives no embeddings, retrieval API, evidence-store API,
Research policy state, legacy section instructions, or conversation history.

## Component classification

| Component | Classification | Active shared-Qwen3 behavior |
| --- | --- | --- |
| `agents/research_agent.py` | ACTIVE-REQUIRED | Tool-policy only; emits no final history answer. |
| `rag/research_runtime.py` | ACTIVE-UTILITY-ONLY | Conversation-aware retrieval query utilities only. |
| `rag/retrieval.py` and tools | ACTIVE-REQUIRED | Hybrid retrieval and evidence collection. |
| `agents/evidence_agent.py` | ACTIVE-REQUIRED | Canonical EvidenceModelOutput and extractive compression validation. |
| `agents/history_contract.py` | ACTIVE-REQUIRED | Canonical History messages and citation parsing. |
| `agents/history_answerer.py::HistoryAnswererAgent` | ACTIVE-REQUIRED | Direct `adapter="history"` generation exactly once. |
| `rag/prompting.py` | LEGACY-STATIC-RAG | Not imported or called by active shared-Qwen3 lifespan. |
| `rag/guards.py` | LEGACY-STATIC-RAG | Not used to rewrite/override active History output. |
| `rag/generation.py` generation/repair/section methods | LEGACY-QWEN2.5 | Lazy-imported only for `legacy-merged`. |
| `LegacyRAGHistoryAnswerer` | LEGACY/COMPAT | Explicit benchmark compatibility wrapper. |
| Qwen2.5 merge and Phase-1 utilities | LEGACY-QWEN2.5 | Training/benchmark-only; not mounted in active runtime. |

## Debug contract

For `debug=true`, the API returns nested `answer_provenance`, `research`,
`evidence`, and `history` metadata.  It includes tool queries, bounded evidence
previews and scores, Evidence status/selection/missing information, History input
and cited IDs, generation count, and guard provenance.  It does not expose a full
private prompt.

## Retrieval-only reproduction

```bash
python -m scripts.debug_retrieval \
  --question "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?" \
  --artifact-root artifacts/vn_history_deployment \
  --final-k 10 --candidate-k 10
```

This command loads the embedder, FAISS, BM25S, and reranker only.  It does not load
Qwen, adapters, or a generation backend.
