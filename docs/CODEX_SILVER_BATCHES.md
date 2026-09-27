# Codex SILVER batches

The benchmark under `evaluation/datasets/v1_silver/questions_500.jsonl` is
automatically annotated **SILVER**, even when the schema names a field
`gold_answer`. It is separate from the human review workspace and GOLD exports.
Do not claim that these records were reviewed by a person.

## Batch contract

- Ten ordered batches of 50 new records yield 500 records. Batch `n` owns IDs
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

## Local storage and commands

The ignored workspace `evaluation/annotation/workspace_codex_500/` contains a
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
