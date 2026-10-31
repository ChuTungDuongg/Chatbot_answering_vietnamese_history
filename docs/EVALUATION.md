# Deterministic evaluation

Evaluation reads a versioned question JSONL and saved prediction JSONL from the HTTP SSE benchmark. `evaluation.runner` does not load Qwen, call a judge, or import application runtime. The separate headless retrieval runner below loads retrieval resources. Run it after a benchmark:

```bash
python -m evaluation.runner --dataset evaluation/datasets/fixtures/questions.jsonl --predictions reports/baseline/hybrid-c1/latency_records.jsonl --output reports/evaluation/hybrid-c1 --phase warm
```

`--phase` can be `warm` (default), `cold`, `warmup`, or `all`. Keep phases separate for meaningful comparisons. `--source-ids path.txt` accepts one known corpus source ID per line, or a JSON string array, to enable an independent source-existence check. The output directory must be new. It receives `per_question.jsonl`, `metrics.json`, and `report.md`. Each per-question row retains available retrieval rank and score diagnostics for later ablations.

## Dataset labels

The human review workflow for Corpus V1 is documented in
[`GOLD_ANNOTATION_GUIDE.md`](GOLD_ANNOTATION_GUIDE.md). It keeps draft questions,
retrieved evidence, and approved gold labels in separate workspace fields.
Only explicitly accepted records can be exported into this evaluation schema.
The separate automatic SILVER pipeline is documented in
[`SILVER_BENCHMARK.md`](SILVER_BENCHMARK.md). Its outputs carry explicit
automatic provenance and cannot enter GOLD export without human review.

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

## Headless retrieval quality

Sau khi có runtime metadata/index hợp lệ, chạy một pass mỗi câu hỏi mà không tải Qwen:

```powershell
$env:APP_MODE = 'retrieval-only'
$env:ARTIFACT_ROOT = './artifacts/corpus_v1'
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'
python -m evaluation.run_retrieval --dataset evaluation/datasets/fixtures/questions.jsonl --output reports/retrieval/v1-faiss
```

Với Qdrant, cấu hình URL/key/collection riêng, manifest đã finalize và đổi `RETRIEVAL_DENSE_BACKEND=qdrant`; dùng output directory khác. `--resume` kiểm tra dataset/corpus/config/index identity và tiếp tục IDs còn thiếu. Output gồm predictions, progress, run manifest và báo cáo metric. Giữ cùng câu hỏi và retrieval settings giữa các lane; fixture không có nhãn nên chỉ kiểm tra pipeline. Xem [Retrieval](RETRIEVAL.md) cho validation và dense diagnostic.

GOLD cần người kiểm duyệt answer/evidence trên đúng corpus revision. Chunk IDs V0/V1 khác nhau nên không chép relevance labels sang corpus khác. Giữ unknown labels null/omitted, adjudicate tranh chấp, đa dạng category và giữ final test riêng khi tuning. Quy trình chi tiết ở [GOLD annotation](GOLD_ANNOTATION_GUIDE.md). Frozen SILVER và six-system evaluation ở [Training](TRAINING_AND_EVALUATION.md).

## Known live-answer limitation

The [recorded acceptance report](https://github.com/ChuTungDuongg/Chatbot_answering_vietnamese_history/blob/5c18145/docs/OBSERVABILITY_FRONTEND_IMPLEMENTATION_REPORT.md) found semantic grounding errors on the Waterloo counterfactual question: a Central answer inverted Prussian/French roles and a Hybrid answer mixed coalition strength with battlefield troop counts. Valid citation IDs and successful streaming did not establish that cited evidence entailed those claims. Live style instructions alone did not resolve the recorded failures.

`python -m scripts.check_live_answer_regressions --base-url https://YOUR-CPU-FUNCTION.modal.run --mode central --retrieval-backend faiss --output reports/acceptance/waterloo-central.json` captures a new live result. Its heuristic flags aid review; empty flags do not prove historical correctness. Assess answer/evidence consistency on both modes and intended dense backends before treating a deployment as quality-accepted. This documentation cleanup does not change the prompt, retriever, model or resolve that recorded limitation.
