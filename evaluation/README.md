# Future Central BASE versus ADAPTED evaluation

This package is offline tooling. Production never imports it, and it does not import training. Only two synthetic schema fixtures are included; they are not a real benchmark and have no historical correctness labels. No real evaluation or judge inference was executed.

```text
evaluation/
  schema.py, io.py, recording.py
  configs/paired.json
  datasets/README.md, question.schema.json, fixtures/questions.jsonl
  metrics/specs.py, aggregate.py
  runners/run.py, compare.py
  logs/.gitkeep
  reports/.gitkeep
```

## Raw records and reproducibility

`Question`, `EvaluationRecord`, `Annotation` and `RunMetadata` in `schema.py` define versioned JSON-only contracts. Future runs write `evaluation/logs/<run-id>/metadata.json` and append/flush one `records.jsonl` row per question. Duplicate IDs, existing output directories and mismatched run/variant labels are rejected. Exceptions produce error rows so failures cannot silently disappear from paired comparisons.

Records retain question/category, parsed semantics, answer, source and selected evidence metadata, structured tool trace, validation/citation/repair information, calls/tokens/latency, failure reason and actual adapter state. They also retain the normalized graph name/version/topology fingerprint, nodes, routes and per-node timing/call deltas. `raw_result` preserves the existing application response, including visible evidence excerpts, so future metric revisions can be applied without inference. No tensors, checkpoints or LangGraph internal serialization are logged. `signals` holds conservative derived observations; absence stays unknown.

Metadata records timestamp, host commit, model/cache revision, adapter path/content hash, dataset version/content hash, retrieval artifact content hash, prompt/host-source hash, generation/retrieval/tool/context settings, graph topology fingerprint, seed, environment versions and hardware. Secrets are excluded from the settings snapshot. Future datasets/logs may contain user content; only curated non-sensitive evaluation questions should be used.

The runner invokes `app.main.lifespan` and the resulting **same production CentralAgent**, with no alternative host or legacy fallback. Each variant runs in a fresh process. Shared settings and question order are identical; the per-question seed is `seed + row_index`. Central stays lazy, so first-request model initialization is measured separately through the host cold-start flag. Existing local model/cache and retrieval artifacts are required. Mutable web/Wikipedia/uploaded-document tools are disabled equally in this initial paired profile. No Modal infrastructure is created by this package.

BASE has no adapter; ADAPTED has the configured fingerprinted V2 adapter. Comparison verifies equal host commit, question set and text, model revision, retrieval index/settings, tools, prompts, context budgets, generation settings, seed and software environment. Successful records must confirm actual configured/loaded adapter state; a silently unadapted successful run is rejected. Equal known hardware class is required for latency comparison; `--without-latency` explicitly removes latency deltas when hardware is unavailable/different. Cold versus warm means are also reported. These controls reduce confounding; finite hardware scheduling and kernel nondeterminism remain possible.

## Commands

Input validation is the default and does not import production, Torch or a model:

```bash
python -m evaluation.runners.run --variant base
```

After V2 training, curate a real question set and change `configs/paired.json` identically for both variants. Only then, on the intended host with all artifacts already installed, use the explicit execution commands:

```bash
python -m evaluation.runners.run --config evaluation/configs/paired.json --variant base --run-id base-run --execute
python -m evaluation.runners.run --config evaluation/configs/paired.json --variant adapted --run-id adapted-run --execute
```

Future offline rescoring/reporting, with no model imports:

```bash
python -m evaluation.runners.compare --base evaluation/logs/base-run --adapted evaluation/logs/adapted-run --output evaluation/reports/comparison-run
```

`calculate_metrics(records)` also scores one saved run. `write_reports` emits `comparison.json`, `comparison.md` and `per_question.csv`. Reports show per-metric BASE/ADAPTED values, absolute/relative deltas and eligible pair counts, plus per-question host pass/fail and optional reviewer win/tie/loss. Relative delta is null for a zero baseline. Deltas use the **same eligible questions** on both sides; complete per-variant observed metrics are retained separately. Conditional measures such as repair success compare the subset where both sides qualify, not an invented common score.

## Metric groups and limitations

`metrics/specs.py` is the metric catalog and records the unit, denominator, eligibility gate and preferred direction. `aggregate.py` implements the common mean/count-ratio/median/nearest-rank-p95 reducers. The seven groups are kept separate:

| Group | Measurements |
|---|---|
| reliability | Host success/failure, validation failure, insufficient/partial answers, repairs and one-generation success |
| grounding | Host risk checks, unsupported names/years, selected target/facet consistency, filter collapse, labelled sufficiency accuracy |
| citations | Valid/invalid reference fractions, uncited factual paragraphs, citation-target matches and repair outcomes |
| behavior | Tool usage/success/parse/malformed calls, local history use, counts, annotated unnecessary calls, legacy-role calls (required zero) |
| answer_quality | Supported dimension/facet/actor coverage, depth, comparison balance, relationship validity and optional review labels |
| viewpoint | Quote/paraphrase issues, labelled false positives and repair outcomes |
| efficiency | Mean/median/p95 latency, calls/tokens, repair generations and cold/warm latency |

Every metric exposes observed record count and denominator. Missing fields or zero eligible denominator produce `null`/N/A, never a made-up zero or perfect result. Trace heuristics measure host behavior, not absolute historical truth. Answer depth compliance measures the available host breadth check, not literary quality or every aspect of requested length. Facet/actor coverage measures selected evidence support, not semantic entailment of the final answer.

Current traces lack complete citation-occurrence and factual-paragraph denominators. Their unique-ID lists and uncited-only classifications must **not** be substituted for those totals. These rates stay N/A unless saved records provide complete `citation_stats` (`valid_citations`, `invalid_citations`, `total_citations`, `uncited_paragraphs`, `factual_paragraphs`, `target_matched_citations`, `target_checked_citations`). Raw excerpts and citation diagnostics remain available for a future reviewed extractor. Similarly, unknown latency/token counters on failed requests are not fabricated.

Optional per-record `annotation` accepts reviewer/method/rubric provenance, historical correctness, completeness, relevance, neutrality, coherence, sufficiency correctness, unnecessary-call counts, partial-answer correctness, viewpoint ground truth and paired preference. A human or later external judge may supply these labels; neither is mandatory. No judge is invoked by any metric. Preserve the original raw records when producing annotated copies.

Generated `logs/` and `reports/` contents are ignored; `.gitkeep`, schemas, code/configuration and tiny fixtures remain tracked. No evaluation UI, tracking platform or aggregate magic score is introduced.
