# Human-reviewed evaluation gold set

The two rows in `evaluation/datasets/fixtures/questions.jsonl` are demonstrations without labels. They cannot establish retrieval or answer accuracy. `evaluation/datasets/question.schema.json` defines the version 1 question contract; `id`, `question`, and `category` are required. The following labels should be supplied by human review, or left null/omitted when unknown:

| Field | Review meaning |
| --- | --- |
| `gold_answer` | Reviewed historical answer with scope and uncertainty. |
| `relevant_chunk_ids` | Chunks supporting the question in a specific corpus version. |
| `relevant_source_ids` | Source documents supporting the answer. |
| `gold_citation_source_ids` | Source IDs that a correct cited answer should identify. |
| `required_facts` | Atomic facts an answer must cover. |
| `answerable` | Whether the available corpus can answer the question. |
| `in_domain` | Whether the question belongs to the history task. |

Record the corpus version and artifact hashes for each labeling pass. Chunk IDs change between V0 and V1, so review relevance for each corpus rather than copying labels blindly. Adjudicate disputed answers and use multiple question categories, including world history, people, places, and heritage. Keep unknown labels null. Compare V0 and V1 on the same reviewed questions before replacing the runtime corpus.
