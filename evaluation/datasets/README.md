# Versioned questions

Each JSONL line follows `question.schema.json` (schema version 1). `id`, `question`, and `category` are required. All gold labels are optional. Omitted labels mean **unlabeled**, not a negative or zero score. An explicit empty relevance list means an annotator found no relevant IDs; recall, MRR, and nDCG then have no positive denominator and remain N/A.

`relevant_chunk_ids` and `relevant_source_ids` are separate rankings. `gold_citation_source_ids` is the set of expected cited source IDs. `required_facts` supports only exact normalized phrase coverage. `factual_paragraph_indices` is a zero-based list of answer paragraphs to check for citation coverage. `answerable` and `in_domain` enable coarse refusal checks. Labels should be curated and provenance recorded before claiming historical accuracy.

The included fixture has two **unlabeled** questions. It tests the pipeline, not retrieval or answer quality.
