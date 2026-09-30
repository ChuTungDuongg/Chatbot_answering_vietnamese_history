# Application runtime

FastAPI exposes two inference modes: `hybrid` and `central`. The mode router selects the requested runtime without changing the model or falling back to another mode.

| Area | Responsibility |
|---|---|
| `api/` | Chat, SSE, conversation, and attachment endpoints. |
| `rag/` | Retrieval interface, existing hybrid search implementation, prompt assembly, and Hybrid runtime. |
| `central/` | Central tool use, local history grounding, and final answer runtime. |
| `models/` | Qwen model loading, optional PEFT adapter, and incremental text generation. |
| `chat/` | SQLite conversation storage and uploaded-document handling. |
| `telemetry.py` | Request trace and timing fields. |

`app_mode` controls startup: `api-only` serves lightweight API features, `retrieval-only` loads corpus/index dependencies, and `full` enables model generation. In full mode the hybrid model loads at startup, so an invalid SFT adapter fails startup. `MODEL_VARIANT=vanilla` uses the unmodified `Qwen/Qwen3-4B-Instruct-2507`; `MODEL_VARIANT=sft` attaches the local adapter specified by `MODEL_ADAPTER_PATH` to the same base model. The tokenizer always comes from the base model. The Central Agent remains vanilla `Qwen/Qwen3-8B` and follows `RUNTIME_LOADING_STRATEGY`. Decoding defaults to deterministic with thinking disabled.

The SSE `done` event and baseline metadata expose `model_variant`, `adapter_attached`, and `adapter_fingerprint`. The fingerprint uses the same file-tree SHA-256 algorithm as six-way evaluation and is computed once when the model loads. `/ready` reports Hybrid and Central load state without forcing a lazy Central load.

The retrieval implementation reads the preserved historical corpus and its FAISS and BM25S indexes. Startup does not build or modify them. Missing required artifacts or model files produce an explicit error.

For a streamed answer, `POST /api/v1/chat/stream` accepts `conversation_id`, `question`, and `mode`; it emits `status`, incremental `answer_delta`, `sources`, `done`, and `error` SSE events. The server persists the same visible answer assembled from deltas. Central tool calls may occur before visible final-answer generation.

See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the dependency diagram and [`../docs/BASELINE_BENCHMARK.md`](../docs/BASELINE_BENCHMARK.md) for timing definitions.
