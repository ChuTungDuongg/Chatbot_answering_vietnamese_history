# 📚 Hybrid RAG Runtime

[⬅️ Backend](../README.md) · [📦 Artifact contract](../../artifacts/README.md)

```text
question
  → intent/OOD analysis + query variants
  → multilingual-e5-base dense search (FAISS IP)
  + BM25S sparse search
  → weighted reciprocal-rank fusion
  → BAAI/bge-reranker-v2-m3
  → metadata boost + context diversity
  → text chunks cho Evidence/History agents
```

- `retrieval.py` reuse Phase 9 retrieval; `SearchHistoryTool` chỉ wrap nó, không duplicate implementation.
- `prompting.py` tạo grounded prompt và kiểm soát token budget/history.
- `generation.py` gọi merged Qwen2.5 History model, parse source IDs và cho tối đa một evidence-only repair.
- `guards.py` kiểm tra source IDs, year support, format và chất lượng answer.

Embedding chỉ phục vụ retrieval/similarity/rerank candidate selection. Vector không được đưa trực tiếp vào History Answerer.
