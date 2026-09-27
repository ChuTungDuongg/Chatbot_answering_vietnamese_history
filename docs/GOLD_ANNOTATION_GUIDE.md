# Human review guide for the Vietnamese History benchmark

## Scope and tiers

This is an **evaluation benchmark**, not training data. The UVW source
`train`/`validation`/`test` partitions are provenance of the original corpus;
they are not benchmark splits. A reviewer may assign a separate benchmark
`dev` or `test` split. Keep final `test` labels away from tuning scripts.

Build the benchmark in stages: review a 50-question pilot, revise these rules,
expand to about 150, then to about 400 accepted retrieval questions. From the
accepted retrieval set, select 100–150 diverse questions for deeper answer and
citation review. The deep queue is another review task; selection does not
create deep gold automatically.

Tier A (retrieval) needs an accepted question, category, explicit answerability
and domain labels, and reviewed relevant chunk/source IDs. Tier B (deep QA)
adds a reviewed answer, atomic required facts, acceptable citation source IDs,
and optional factual paragraph indices. The generator's `draft_question`,
`draft_answer`, `candidate_required_facts`, origin chunk, and retrieved top hits
are **candidate material only**. They never become gold through ranking or
generation alone.

## Review decisions

1. Read the question and inspect the retrieved dense, BM25, RRF, and final
   reranked stages. Open the full chunk text and nearby chunks when needed.
2. Edit the question and category for a clear, bounded historical claim. Mark
   difficulty as `easy`, `medium`, or `hard` based on evidence demands, not
   question length. Easy usually needs one explicit fact; medium connects
   facts in one source or nearby chunks; hard genuinely needs multiple sources
   or hops. The generator's difficulty is only a draft.
3. Mark each examined **chunk** and **source** `relevant`, `not_relevant`, or
   `uncertain`. Leave unexamined evidence undecided. A neighboring chunk is
   never relevant merely because it is adjacent to a relevant chunk.
4. Set `answerable` and `in_domain` explicitly. Write notes explaining
   disputed or uncertain cases. Press **Accept retrieval** only after checking
   the supporting IDs. The page checks that a relevant chunk's source is also
   marked relevant. Use **Needs review** for unresolved evidence, **Reject**
   for unsuitable questions, or **Skip** to keep a pending item. Each action
   persists immediately in SQLite.
   Editing question wording clears old retrieval evidence and relevance
   decisions. Save the edit, rerun candidate generation with `--resume` to
   retrieve for the new wording, then review its evidence before acceptance.
5. For selected deep questions, write a new gold answer and fact/citation
   labels. Press **Accept deep QA** separately. Editing an accepted retrieval
   label reopens retrieval review; editing accepted deep labels reopens deep
   review. `reviewed_at` and `review_version` document each acceptance. Use
   `reviewer_id` only if your team wants a non-sensitive identifier.

`pending`, `accepted`, `rejected`, and `needs_review` are the retrieval states.
Deep selection has its own `pending`, `accepted`, and `needs_review` states.
`retrieval_reviewed`, `answer_reviewed`, and `citation_reviewed` are separate
gates. A candidate cannot enter final gold because it was generated or
retrieved. Unknown labels stay absent/null; an explicit empty relevance list
means a reviewer found no positive evidence.

## Label definitions

**Relevant chunk:** A specific Corpus V1 retrieval unit that directly supports
an answer or a correction to the question's premise. A topic mention alone is
not enough. Favor the smallest set of chunks that covers the needed facts.

**Relevant source:** An article/document containing relevant evidence. Multiple
relevant chunks may belong to one source. A source can be relevant when the
reviewer finds useful evidence in another chunk of that article; inspect it
before marking it.

**Gold citation source:** A source acceptable to cite for the final answer.
It may differ from the retrieval gold source list: a source with a clear
summary may be an acceptable citation while a different source supplied the
retrieval unit you labeled. Citation IDs must exist in Corpus V1 and must be
judged from the source content, not copied from top-k results.

**Gold answer:** A concise human-reviewed historical answer with appropriate
scope, time, names, and uncertainty. Do not paste a model draft without
checking it. For unanswerable or false-premise questions, the gold answer can
explain the limitation or correct the premise.

**Required fact:** One atomic fact essential to an adequate answer. Write
several small facts instead of one compound paragraph. The deterministic
evaluator tests normalized phrase presence, so wording should be stable and
specific; human review remains the authority.

**Factual paragraph indices:** Zero-based paragraphs in the gold answer that
need citations. Leave null when not reviewed. Use `[]` only when reviewed and
no paragraph needs a factual citation.

## Edge cases and variants

- **Ambiguous:** If a person, place, or event name has several possible
  referents, edit the question to identify one or mark `needs_review`.
  If ambiguity is the intended benchmark challenge, document the possible
  interpretations in notes and set `answerable=false` only when the corpus
  cannot resolve them.
- **False premise:** Preserve the premise only if the question tests whether
  an answer corrects it. Mark `answerable=false` when no factual answer to
  that premise exists; supporting correction chunks may still be relevant.
  Explain the correction in notes and, for deep review, the gold answer.
- **Out of domain:** Set `in_domain=false`, `answerable=false`, and normally
  use explicit empty retrieval gold lists. Do not force a historical citation.
- **Insufficient evidence:** Set `in_domain=true`, `answerable=false` when
  the corpus does not support a reliable answer. Explain what is missing.
- **Multiple correct answers:** Allow all well-supported alternatives in the
  gold answer and relevant sources. Avoid narrowing the question afterward
  merely to fit a single retrieved chunk.
- **Date/name variants:** Check historical period, calendar conventions,
  diacritics, transliterations, and aliases. Put accepted variants in notes
  or the gold answer; do not mark a different date correct just because a
  generated draft proposed it.
- **Duplicated evidence:** Mark the chunks that directly support the answer,
  then keep source IDs distinct. Multiple near-identical chunks from one
  article do not create multiple independent sources. Use the duplicate
  report to remove repeated questions and overrepresented titles.

## Worked examples

These examples illustrate decisions, not preapproved Corpus V1 IDs.

| Question | Retrieval review | Deep review |
|---|---|---|
| “Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?” | Mark a chunk that states the significance relevant. Mark its article source relevant. A chunk that only mentions the battle date is not enough for the significance claim. | Gold answer explains the outcome and significance. Required facts are separate claims; acceptable citation sources must substantiate them. |
| “Hiệp định Genève được ký năm 1884 phải không?” | This draft has a suspicious premise. Verify the agreement and date in source text. Mark a correcting chunk relevant, label the premise according to the verified evidence, and note the correction. | Gold answer explicitly corrects the date if the source supports the correction. Do not copy the generator's guessed year. |
| “Sự kiện này diễn ra khi nào?” | The referent is missing. Mark `needs_review` or edit the question to name the event. Do not label the origin chunk relevant merely because the generator used it. | Do not create a deep gold answer until the question is resolved. |
| “Thời tiết Hà Nội ngày mai thế nào?” | Mark `in_domain=false`, `answerable=false`; explicit empty chunk/source gold is appropriate. | If selected for deep review, a brief refusal is the gold answer and fact/citation lists may be empty. |

## Commands

Prepare the small Corpus V1 runtime metadata if it is not already present:

```powershell
python -m scripts.retrieval.prepare_v1_runtime
```

Create the **first 50 draft candidates** and retrieve evidence with the FAISS
lane. This can take substantial CPU time because E5 and the BGE reranker run
once for every candidate; it never loads Qwen. `--resume` continues incomplete
evidence after interruption. The generated JSONL and SQLite databases remain
local and ignored by Git.

```powershell
python -m evaluation.annotation.generate_candidates --limit 50 --dense-backend faiss --workspace evaluation/annotation/workspace
python -m evaluation.annotation.generate_candidates --limit 50 --dense-backend faiss --workspace evaluation/annotation/workspace --resume
```

Start the local review page at `http://127.0.0.1:8765`:

```powershell
python -m evaluation.annotation.ui --workspace evaluation/annotation/workspace --port 8765
```

Review coverage and validate accepted labels before export:

```powershell
python -m evaluation.annotation.report --workspace evaluation/annotation/workspace --output reports/annotation/coverage.json
python -m evaluation.annotation.validate_dataset --workspace evaluation/annotation/workspace --tier retrieval
```

Once the pilot is reviewed, expand the total draft queue to about 150, then
400. Change `configs/annotation_pilot.json` if the category plan needs tuning.
The command requires the current queue to be fully reviewed before expansion.

```powershell
python -m evaluation.annotation.generate_candidates --limit 150 --expand --workspace evaluation/annotation/workspace
python -m evaluation.annotation.generate_candidates --limit 400 --expand --workspace evaluation/annotation/workspace
```

Select a balanced deep review queue only after enough Tier A items are
accepted. Selection weighs category, difficulty, and observed retrieval
behavior, including misses; it does not export deep gold.

```powershell
python -m evaluation.annotation.select_deep_subset --workspace evaluation/annotation/workspace --limit 125
```

Export only accepted and validated labels. These paths are examples; keep
final `test` labels private while tuning. Exports refuse to overwrite existing
files and satisfy `evaluation/datasets/question.schema.json`.

```powershell
python -m evaluation.annotation.export --workspace evaluation/annotation/workspace --tier retrieval --output evaluation/datasets/v1_gold/questions.jsonl
python -m evaluation.annotation.export --workspace evaluation/annotation/workspace --tier deep --output evaluation/datasets/v1_gold/deep_questions.jsonl
```

Back up and restore the transactional workspace without copying full corpus chunks:

```powershell
python -m evaluation.annotation.workspace backup --workspace evaluation/annotation/workspace --file reports/annotation/workspace-backup.jsonl
python -m evaluation.annotation.workspace restore --workspace evaluation/annotation/new_workspace --file reports/annotation/workspace-backup.jsonl
python -m evaluation.annotation.corpus --workspace evaluation/annotation/new_workspace
```

The source index is a local SQLite file containing IDs and byte offsets, not
chunk text. Candidate drafts can contain short excerpts; the backup includes
those drafts and reviewer edits. Rebuild the source index from the same corpus
after restoring a workspace.
The UI displays retrieved evidence but
never converts rank or origin provenance into a gold label. Model assistance
is optional; the provided generator uses deterministic templates and no API
key, external LLM, or paid service.
