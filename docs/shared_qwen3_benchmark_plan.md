# Qwen3 shared-base benchmark plan

Benchmark the legacy static Hybrid RAG + Qwen2.5 History baseline against Agentic Hybrid RAG + one shared Qwen3 base + three role LoRAs on the same questions, retrieved evidence, and scoring code where applicable.

Quality metrics: answer correctness, Token F1, ROUGE-L, multi-hop success, conflict precision/recall/F1, insufficient-evidence accuracy, and false-premise handling.

Grounding metrics: citation precision/recall/F1, source-ID validity, hallucinated citation rate, per-claim support, selected-set answer-slot coverage, partial-evidence retention, and minimal sufficient subset rate.

Systems metrics: end-to-end latency p50/p95, LLM calls/request, generated tokens/request, GPU-seconds/request, peak VRAM, requests/second, and tokens/second. Record backend, dtype/quantization, context length, concurrency, adapter rank, warm-up procedure, and hardware.

Do not claim universal quality, latency, throughput, or VRAM improvement until this benchmark is run. Shared-base serving targets reduced duplicated residency and simpler maintenance; it does not reduce the per-token FLOPs of Qwen3-4B History generation relative to Qwen2.5-3B.

