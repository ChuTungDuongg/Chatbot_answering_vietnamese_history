---
name: search_history
kind: retrieval
provider: shared_hybrid_retriever
requires_env: []
inputs:
- query
- top_k
outputs:
- list of corpus chunks with chunk_id, title, text, source_kind=history
side_effect: false
requires_confirmation: false
---

# search_history

Search the local Vietnamese history corpus for grounded source chunks. Prefer this for Vietnamese history, including local events and figures. Do not assume it covers unrelated world history; use permitted external sources for those questions.

Uses the injected retriever and the request-selected FAISS/Qdrant backend.

Arguments and defaults are declared in `../../definitions/tools.yaml`; the adjacent `tool.py` Pydantic model validates runtime input. Registration does not grant permission: existing request policies still apply.
