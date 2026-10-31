# Corpus V1 SILVER benchmark

**SILVER** means automatically generated and automatically adjudicated. It is
not a human-reviewed GOLD dataset. The existing 50-question pilot in
`evaluation/annotation/workspace/`, review UI, GOLD export, and
[`GOLD_ANNOTATION_GUIDE.md`](GOLD_ANNOTATION_GUIDE.md) remain authoritative for
GOLD. This pipeline uses a separate ignored SQLite workspace at
`evaluation/annotation/workspace_silver_v1/`. It never copies or changes
`artifacts/corpus_v1/chunks.jsonl`.

## Hai workflow tự động

`auto_annotate` bên dưới dùng semantic endpoint và workspace riêng. `codex_batch` nhập record/audit theo batch cố định; không tự gọi model. Frozen V1 đã chọn 3.000 câu hỏi từ 60 batches, xem [Training và evaluation](TRAINING_AND_EVALUATION.md). Tên workspace/master có `5000` phản ánh trần thiết kế, không phải số câu của benchmark V1 đã freeze.

## Prerequisites and cost check

The Corpus V1 FAISS, BM25, runtime manifest, and local E5/BGE weights must be
available as described in the retrieval runtime documentation. The automatic
semantic model is an OpenAI-compatible JSON chat endpoint. A local server can
serve Qwen or another model at `http://127.0.0.1:8001/v1`; this tool does not
start or download it. Set `SILVER_LLM_BASE_URL` to another loopback endpoint if
needed, and pass `--model` with the model ID actually served. `--provider
external` requires an HTTPS endpoint, `SILVER_LLM_API_KEY` in the environment,
and `--execute-paid`. Never put a key in a command, config, or report. The UI
does not require any paid service.

The safe first command is:

```bash
python -m evaluation.annotation.auto_annotate --target 500 --deep-target 150
```

It prints a model-call estimate and exits **before any corpus scan, model load,
network request, or workspace write**. With 12 candidate chunks per question
and six per semantic batch, the planning upper estimate for 500 usable items
is 500 generation + 1,000 relevance + 500 independent verification + up to
500 contradiction checks + 300 deep-answer/verification calls = **up to 2,800
calls**. Replacements can increase this; the configured 1,500-attempt ceiling
has a conservative bound of 7,800 calls. These are call counts, not a cost
quote or calibrated probability. Actual counts depend on rejection, early
domain gates, contradictions, and deep failures. Always inspect the estimate
before passing `--execute`.

## Run and resume

The real FAISS-only run with a local endpoint is:

```bash
python -m evaluation.annotation.auto_annotate --target 500 --deep-target 150 --phase retrieval --dense-mode faiss-only --provider local --model Qwen/Qwen3-4B-Instruct-2507 --execute
```

To use both dense backends when Qdrant has a **finalized** V1 manifest and the
runtime manifest has been refreshed, use `--dense-mode both`. It validates the
remote point count and vector/HNSW configuration using the production loader.
The default `--dense-mode auto` uses Qdrant only when its finalized manifest
and required environment variables are present. A `qdrant.partial/` directory
never enables it. FAISS-only development needs no Qdrant credentials. Both
backends supply **candidates**, never relevance labels.

For a small real integration run, add `--max-questions 3`; it keeps the run
incomplete and checkpointed. To continue, use the same arguments with
`--resume` and remove `--max-questions`. `--start-after N` can explicitly skip
earlier candidate sequences in an existing workspace. `--concurrency N` runs
semantic work in parallel; production retrieval is serialized to protect the
single loaded service. Use `--concurrency 1` for reproducible quality work.
An OS workspace lock prevents concurrent processes. Each generated question,
retrieval result, semantic pass, confidence decision, and deep result is
committed in SQLite. A crash does not require regenerating completed IDs or
rerunning completed semantic calls. Resume refuses a changed corpus, config,
model, backend availability, or retrieval manifest identity. Change those by
starting a **new** SILVER workspace; do not overwrite the human pilot.

`--target 500` is an **absolute target of usable high/medium confidence SILVER
retrieval records**, not 500 additional rows or 500 raw drafts. The generator
replaces duplicate, low-confidence, and rejected attempts until it meets the
target or the finite `--max-attempts` ceiling. It reports raw attempts,
rejections, and usable count; shortages remain explicit.

After retrieval, annotate the balanced deep subset:

```bash
python -m evaluation.annotation.auto_annotate --target 500 --deep-target 150 --phase deep --dense-mode faiss-only --provider local --model Qwen/Qwen3-4B-Instruct-2507 --execute --resume
```

Deep selection covers category, difficulty, question type, answerability,
retrieval misses, and backend disagreement. Failed deep verifications stay in
the workspace and replacements are selected from the remaining eligible
retrieval records. Automatic fields inside SQLite are named `silver_answer`,
`candidate_required_facts`, and `candidate_citation_source_ids`. They are
projected into the existing evaluation schema only in a SILVER export with
`annotation_origin=automatic` and `annotation_status=auto_reviewed`.

## Evidence and confidence

Generation starts with sampled Corpus V1 provenance, category quotas, source
and title caps, famous-subject limits, and near-duplicate rejection. The
retriever saves FAISS, optional Qdrant, BM25, RRF, and reranked candidates.
The origin chunk and, for multi-source questions, a second corpus source are
shown as candidate evidence. Neither origin nor any top-k rank becomes a label.

The first semantic pass judges each **actual chunk text**, source/title, and
nearby context when needed as `relevant`, `not_relevant`, or `uncertain`, with a
reason. The second pass independently verifies answerability, domain, and
evidence sufficiency. Medium/hard or multi-source questions receive a conflict
check. Material disagreement, missing answer evidence, unresolved
contradiction, or a domain conflict forces low confidence and
`needs_human_review`. Other transparent penalties include uncertain evidence,
source concentration, retrieval misses, absent BM25/reranker support, backend
disagreement, edge cases, and multi-hop complexity. The 0–100 heuristic is
**not a calibrated probability**. FAISS/Qdrant overlap is diagnostic and never
proves historical correctness. A retrieval miss does not imply out of domain.

Deep answers may use only the verified evidence. Required facts must be
atomic, each mapped to supporting chunk and source IDs. Citation sources must
support at least one verified fact. A separate pass checks fact and citation
support. Unsupported deep answers remain in the workspace for human audit.

## Validate, report, export, audit

```bash
python -m evaluation.annotation.silver_validate --tier silver-retrieval
python -m evaluation.annotation.silver_validate --tier silver-deep
python -m evaluation.annotation.silver_report
python -m evaluation.annotation.export --tier silver-retrieval --output evaluation/datasets/v1_silver/questions.jsonl
python -m evaluation.annotation.export --tier silver-deep --output evaluation/datasets/v1_silver/deep_questions.jsonl
```

The validator checks IDs, labels, chunk/source consistency, semantic-pass
agreement, duplicate questions, supported facts/citations, and contradictions.
It reports errors and warnings separately and never repairs labels. Default
SILVER exports include only automatic high/medium records. They omit human
review flags and keep explicit automatic provenance. Generated datasets and
reports are ignored by Git.

Prepare the approximately 35 risky + 15 random high/medium audit queue:

```bash
python -m evaluation.annotation.silver_audit --size 50 --risky 35
python -m evaluation.annotation.ui --workspace evaluation/annotation/workspace_silver_v1/audit_gold --lookup-workspace evaluation/annotation/workspace_silver_v1 --port 8765
```

Open `http://127.0.0.1:8765/`. The queue is **pending** in the existing human
UI. Its initial chunk/source decisions are empty, regardless of SILVER labels.
The reviewer must inspect and explicitly accept evidence and answerability.
After review:

```bash
python -m evaluation.annotation.export --tier audit-gold --output evaluation/datasets/v1_gold/audit_questions.jsonl
python -m evaluation.annotation.silver_report
```

Only human-accepted rows enter the audit GOLD file. The report shows exact
agreement rates with explicit human-reviewed denominators, including rates by
SILVER confidence group. It does not call those values historical accuracy.
If a reviewer edits a question, save it and rerun candidate retrieval evidence
before accepting it:

```bash
python -m evaluation.annotation.silver_audit --refresh-evidence
```

This refresh uses FAISS/BM25 retrieval and preserves human edits. The user must
still mark relevance explicitly.

## Interpretation and limits

The automatic pipeline target is configurable; its outputs are SILVER. Corpus sampling and automatic
question generation can favor common titles or source styles. Automatic labels
can inherit semantic-model and retriever bias. Exact FAISS and approximate
Qdrant agreement only shows retrieval overlap. The 50-question audit tests
automatic annotation quality against explicit human review, while the full
human GOLD workflow remains the reference. Keep retrieval quality, answer
quality, latency, and load tests as separate experiments. Do not tune on final
benchmark test labels.

## Codex batch workflow

### Batch contract

- The general batch tool has a 5,000-record ceiling (100 batches of 50). The frozen V1 experiment stops at 60 batches / 3,000 records. Batch `n` owns IDs
  `vn_hist_silver_{(n-1)*50+1:04d}` through `vn_hist_silver_{n*50:04d}`.
- Each completed batch has **18 easy, 22 medium, and 10 hard** questions.
  Difficulty reflects evidence and reasoning: an easy question asks one direct
  fact, medium combines facts or explanation, and hard requires genuine
  comparison, multiple sources, complex chronology, or premise correction.
- Inspect actual Corpus V1 chunks and source metadata. Retrieval rank is not a
  relevance label. Select only chunks that directly support the answer or
  correct a false premise; cite only sources supporting factual claims.
- For every record, write a first annotation and an independent verification
  note. The audit file records exact corpus quotes, relevance decisions, and
  which required facts each quote supports. Resolve contradictions or phrase
  the question as a source comparison; do not silently choose one account.
- A complete batch is immutable. If a policy change requires amendment,
  preserve the old batch, audit, and master files under the ignored workspace's
  `superseded/` directory before republishing the same IDs. Document the reason.

### Local storage and commands

The ignored workspace `evaluation/annotation/workspace_codex_5000/` contains a
SQLite draft store and `batches/batch_01.jsonl`, etc. Each accepted record is
saved in one SQLite transaction, so an interrupted batch resumes at the next
stable ID. The ignored cumulative master is rebuilt from validated batches in
deterministic order; never append to it manually.

```powershell
python -m evaluation.annotation.codex_batch status
python -m evaluation.annotation.codex_batch complete
python -m evaluation.annotation.codex_batch merge
```

The `add` and `revise` actions accept a JSON file with `record` and `audit`
objects. `complete` requires exactly 50 validated records, checks IDs and
duplicate questions across batches, writes the batch and audit atomically, and
rebuilds the master. `merge` revalidates completed batches and rebuilds the
master after an interrupted publication. Both validate the existing
`evaluation.schema.Question` format. This tool reads the existing corpus lookup
in read-only mode; it does not modify the corpus or the human GOLD workspace.

The automatic labels are provisional benchmark labels. Their quality should be
checked against a separately human-reviewed GOLD audit set before drawing
historical accuracy conclusions.
