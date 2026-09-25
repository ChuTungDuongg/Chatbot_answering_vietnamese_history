# Application runtime

FastAPI exposes two inference modes: `hybrid` and `central`. The mode router selects the requested runtime without changing the model or falling back to another mode.

| Area | Responsibility |
|---|---|
| `api/` | Chat, SSE, conversation, and attachment endpoints. |
| `rag/` | Retrieval interface, existing hybrid search implementation, prompt assembly, and Hybrid runtime. |
| `central/` | Central tool use, local history grounding, and final answer runtime. |
| `models/` | Vanilla Qwen model loading and incremental text generation. |
| `chat/` | SQLite conversation storage and uploaded-document handling. |
| `telemetry.py` | Request trace and timing fields. |

`app_mode` controls startup: `api-only` serves lightweight API features, `retrieval-only` loads corpus/index dependencies, and `full` enables model generation. Model loading is lazy by default. `HYBRID_MODEL_ID` is `Qwen/Qwen3-4B-Instruct-2507`; `CENTRAL_MODEL_ID` is `Qwen/Qwen3-8B`. The baseline uses no adapters and defaults to deterministic decoding with thinking disabled.

The retrieval implementation reads the preserved historical corpus and its FAISS and BM25S indexes. Startup does not build or modify them. Missing required artifacts or model files produce an explicit error.

For a streamed answer, `POST /api/v1/chat/stream` accepts `conversation_id`, `question`, and `mode`; it emits `status`, incremental `answer_delta`, `sources`, `done`, and `error` SSE events. The server persists the same visible answer assembled from deltas. Central tool calls may occur before visible final-answer generation.

See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the dependency diagram and [`../docs/BASELINE_BENCHMARK.md`](../docs/BASELINE_BENCHMARK.md) for timing definitions.
