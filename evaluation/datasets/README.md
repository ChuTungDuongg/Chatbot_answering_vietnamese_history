# Future question sets

Use one UTF-8 JSON object per line conforming to `question.schema.json` (generated from `evaluation.schema.Question`). Required: `id`, `question`, `category`. IDs must be unique. Expected question type, event, actors, facets, depth, tool behavior, source IDs and notes are optional annotations, not mandatory model targets.

`fixtures/questions.jsonl` contains two invented questions for schema/runner tests only. It is not a curated benchmark, cannot establish model quality, and has no fabricated historical ground truth. Curate the actual frozen set after Adapter V2 training; use identical ordered bytes for BASE and ADAPTED runs. Keep training examples out of the future evaluation set and document curation/versioning then.
