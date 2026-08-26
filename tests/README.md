# 🧪 Test Suite

[🏠 Project README](../README.md)

```bash
python -m pip install -r requirements.txt -r requirements-training.txt
python -m pytest -q
```

| Test | Contract |
|---|---|
| `test_tool_registry.py` | Typed tool call và call record. |
| `test_agent_loop.py` | Evidence selection path. |
| `test_evidence_schema.py` | Pydantic ID invariants. |
| `test_session_evidence_store.py` | Add/dedup/search session evidence. |
| `test_history_answerer.py` | Selected text contexts tới History wrapper. |
| `test_history_loss.py` | User masking và source-line weight 1.6 của Phase 6. |
| `test_training_cli.py` | `python -m training... --help` smoke. |
| `test_api_smoke.py` | FastAPI routes import/exposure. |
| `test_upload_modal_volume.py` | Bundle upload vào đúng root layout của Modal Volume. |

Tests không tải Qwen 3B/4B, embedding model, FAISS deployment corpus hoặc Hugging Face datasets. Modal GPU sanity scripts là integration checks riêng và có thể phát sinh chi phí.
