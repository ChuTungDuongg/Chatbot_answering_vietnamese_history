# Tests

Run Python checks from the repository root:

```powershell
python -m pytest -q
```

The baseline tests use fake model/token streams and temporary corpus fixtures. They check the two-mode router, vanilla model IDs, retrieval without Qwen loading, SSE event order and incremental deltas, cancellation, telemetry, evaluation math, and corpus preservation. They do not download Qwen weights or run GPU inference.

Frontend checks are separate:

```powershell
npm --prefix frontend test
npm run frontend:lint
npm run frontend:build
```

Live answer quality and latency require a configured full-mode server with the intended corpus, indexes, model revisions, and hardware. Use [`../docs/BASELINE_BENCHMARK.md`](../docs/BASELINE_BENCHMARK.md) and [`../docs/EVALUATION.md`](../docs/EVALUATION.md) for those measurements.
