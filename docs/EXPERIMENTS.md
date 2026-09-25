# One-change-at-a-time experiments

Capture the baseline before changing retrieval, models, prompts, chunking, indexes, or serving parameters. Keep the historical corpus unchanged. Each experiment changes one named variable and uses the same dataset labels, exact model revisions, generation settings, hardware, load level, and warm/cold protocol unless the changed variable is one of those fields. Save raw JSONL and metadata with every run.

## Record template

```text
Experiment ID:
Git commit:
Hypothesis:
Single variable changed:
Baseline configuration:
Experimental configuration:
Dataset path, schema version, content hash:
Corpus and retrieval index hashes:
Model IDs and revisions:
Generation and retrieval settings:
Hardware and software versions:
Run command and output paths:
Runs, warmups, concurrency:
Cold/warm restart procedure:
Metrics before (observed n, mean, p50, p95, p99):
Metrics after (observed n, mean, p50, p95, p99):
Absolute delta:
Relative delta (N/A when baseline is zero):
Retrieval/answer/grounding/citation regression:
Interpretation and limitations:
Decision:
```

The first proposed optimization experiment is **Transformers serving versus vLLM serving** for one mode at a time. Keep the exact Qwen model and revision, prompt, generation settings, corpus, FAISS + BM25S retrieval, dataset, hardware, and concurrency fixed. Compare externally observed answer TTFT, E2E, throughput, and answer/citation outputs. Record any output differences and backend-specific settings. Do not implement this experiment as part of baseline V0.

A later, separate backend experiment can compare the preserved FAISS + BM25S baseline with Qdrant-backed dense retrieval. Record collection version, payload/filter policy, cosine configuration, import/index hash, health state, retrieved IDs, and latency. That experiment must not silently become the V0 default or change chunking, embedding model, BM25, reranker, and model settings at the same time.
