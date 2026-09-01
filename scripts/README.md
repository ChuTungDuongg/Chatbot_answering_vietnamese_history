# ☁️ Modal Upload Scripts

[🏠 Project README](../README.md) · [📦 Artifact contract](../artifacts/README.md)

## 🚚 Upload bundle đầy đủ

```bash
python scripts/upload_modal_volume.py \
  --volume vn-history-artifacts \
  --local-dir artifacts/vn_history_deployment \
  --remote-dir / \
  --dry-run
```

`--dry-run` vẫn validate path nhưng chỉ in các lệnh `modal volume put --force`. Bỏ flag để chạy upload thật.

## 🧩 Upload từng component

```bash
python scripts/upload_modal_volume.py \
  --volume vn-history-artifacts \
  --history-adapter outputs/history-answerer-full/adapter \
  --research-agent outputs/research_agent \
  --evidence-agent outputs/evidence_agent \
  --central-agent outputs/qwen3-8b-agent-v1/final_adapter \
  --retrieval-dir artifacts/retrieval \
  --corpus artifacts/corpus/vn_history_rag_chunks_enriched.jsonl \
  --config-dir artifacts/vn_history_deployment/config \
  --manifest artifacts/vn_history_deployment/manifest.json \
  --dry-run
```

Remote destinations gồm `/adapters/history`, `/adapters/research`, `/adapters/evidence`, `/adapters/central`, `/retrieval`, `/corpus`, `/config` và `/manifest.json`.

## ✅ Sau upload

```bash
modal volume ls vn-history-artifacts
modal run scripts/modal_artifact_sanity.py
modal serve modal_app.py
```

Script không đọc hoặc hard-code Modal token. Authentication do `modal setup` hoặc Modal environment quản lý.
