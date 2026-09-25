# Deterministic evaluation

Evaluation reads a versioned question JSONL and saved prediction JSONL from the HTTP SSE benchmark. It does not load Qwen, call a judge, or import application runtime. Run it after a benchmark:

```bash
python -m evaluation.runner --dataset evaluation/datasets/fixtures/questions.jsonl --predictions reports/baseline/hybrid-c1/latency_records.jsonl --output reports/evaluation/hybrid-c1 --phase warm
```

`--phase` can be `warm` (default), `cold`, `warmup`, or `all`. Keep phases separate for meaningful comparisons. `--source-ids path.txt` accepts one known corpus source ID per line, or a JSON string array, to enable an independent source-existence check. The output directory must be new. It receives `per_question.jsonl`, `metrics.json`, and `report.md`. Each per-question row retains available retrieval rank and score diagnostics for later ablations.

## Dataset labels

`evaluation/datasets/question.schema.json` defines schema version 1. Required fields are `id`, `question`, and `category`. Optional fields are `gold_answer`, `relevant_chunk_ids`, `relevant_source_ids`, `gold_citation_source_ids`, `required_facts`, `factual_paragraph_indices`, `answerable`, `in_domain`, and `notes`. An **omitted** label means unknown. The two included questions are unlabeled examples; their answer and retrieval scores are N/A. Curate and version real labels before claiming any historical quality result.

## Metric families

| Family | Scores | Eligibility and meaning |
|---|---|---|
| Retrieval, chunk | HitRate@K, Recall@K, Precision@K, MRR@K, nDCG@K for K=1,3,5,10 | Requires `relevant_chunk_ids`. Binary relevance. |
| Retrieval, source | Same metrics | Requires `relevant_source_ids`. Repeated chunks from one source collapse to its first rank. |
| Answer similarity | Normalized Exact Match, token precision/recall/F1, ROUGE-L F1 | Requires `gold_answer`. Unicode NFKC/casefold word tokenization. These are surface-form comparisons, **not factual accuracy**. |
| Grounding diagnostics | Required-fact phrase recall, unverified answer years and rate, insufficient-answer behavior, OOD refusal behavior | Each needs its matching label or retrieved source text. Exact phrase/year checks cannot prove semantic support. |
| Citations | Citation validity, cited-retrieved rate, citation precision/recall, source ID existence, answer citation coverage | Validity uses aliases available in retrieved sources. Precision/recall need `gold_citation_source_ids`; corpus existence needs `--source-ids`; paragraph coverage needs `factual_paragraph_indices`. |

For retrieval, HitRate@K is 1 when any gold ID appears in the first K. Recall@K is matched gold IDs divided by all gold IDs. Precision@K is matched IDs divided by K. MRR@K is reciprocal first relevant rank (zero on a miss). nDCG@K is binary DCG divided by ideal DCG. Explicitly empty gold relevance has no positive denominator: recall, MRR and nDCG are N/A, while hit and precision can be zero. No gold list means every retrieval score is N/A. Failed requests count in success rate and have N/A quality scores rather than false zeros.

Citation validity is recognized reference occurrences that resolve to retrieved IDs divided by recognized occurrences. `[S1]`, `[1]`, explicit chunk/source IDs and `display_index` aliases are supported. Bracketed historical years are excluded unless they are explicit aliases. Citation precision is unique cited gold sources divided by unique cited targets, including invalid IDs; recall is unique cited gold sources divided by gold citation sources. No citations or no gold denominator produces N/A rather than a perfect score. `answer_citation_coverage` checks only labeled factual paragraphs, using zero-based paragraph indices; it measures a valid reference's presence, not whether the citation truly entails the paragraph. Source ID existence compares references with a supplied corpus ID set. Retrieval diagnostics preserve any available dense, BM25, RRF, reranker, and final ranks/scores, but missing stages remain absent.

The deterministic grounding check compares 3–4 digit years in the answer against retrieved source text. The question itself is not treated as evidence. It reports **unverified years**, not a definitive hallucination verdict. If source text is unavailable, year support is N/A. `required_facts` measure normalized literal phrase presence only. `answerable=false` and `in_domain=false` enable conservative refusal phrase checks; they do not measure full answer quality. Human labels or a separately versioned optional judge can supplement these metrics later, but must never replace the deterministic baseline. Reports keep retrieval, answer, grounding and citation groups separate, with observed denominators and no aggregate “accuracy” score.
