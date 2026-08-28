# 📦 Deployment Artifact Contract

[🏠 Project README](../README.md) · [☁️ Upload CLI](../scripts/README.md)

Git chỉ giữ tài liệu contract. Model weights, corpus và indexes thật bị `.gitignore` loại khỏi repository.

## 🗂️ Layout chuẩn

```text
artifacts/vn_history_deployment/       # local ARTIFACT_ROOT
├── adapters/
│   ├── research/                      # Qwen3 Research LoRA
│   ├── evidence/                      # Qwen3 Evidence LoRA
│   └── history/                       # fresh Qwen3 History LoRA
├── retrieval/
│   ├── faiss/chunks.index
│   ├── faiss/manifest.json
│   └── bm25s_index/
├── corpus/vn_history_rag_chunks_enriched.jsonl
├── config/inference_config.json
├── config/model_registry.json
├── legacy/qwen25_history/model/       # optional benchmark-only baseline
├── manifest.json
└── EXPORT_SUCCESS.txt
```

Trên Modal, Volume được mount trực tiếp ở `/artifacts`. Shared Qwen3 base tải/cache một lần ở `/hf-cache`; artifact không chứa ba bản sao base weights.

## 🧱 Artifact theo mode

| Mode | Bắt buộc |
|---|---|
| `api-only` | Không artifact. |
| `retrieval-only` | corpus, config, manifest, FAISS, BM25S. |
| `full` + `legacy-merged` | retrieval artifacts + legacy History model (baseline only). |
| `full` + `transformers` | retrieval artifacts + ba Qwen3 adapters. |
| `full` + `vllm` | retrieval artifacts + endpoint đã phục vụ ba role names. |

Cả Research, Evidence và History adapter phải khai báo cùng `Qwen/Qwen3-4B-Instruct-2507`; exporter/runtime fail sớm khi metadata lệch base.

## 🏗️ Tạo bundle

```bash
python -m training.scripts.export_artifacts \
  --research-agent outputs/research_agent \
  --evidence-agent outputs/evidence_agent \
  --history-agent outputs/history-answerer-full/adapter \
  --corpus artifacts/corpus/vn_history_rag_chunks_enriched.jsonl \
  --retrieval-dir artifacts/retrieval \
  --output-root artifacts/vn_history_deployment
```

## ☁️ Upload

```bash
python scripts/upload_modal_volume.py \
  --volume vn-history-artifacts \
  --local-dir artifacts/vn_history_deployment \
  --remote-dir / \
  --dry-run
```

Bỏ `--dry-run` khi danh sách lệnh đúng. CLI validate local path trước khi ghi Volume.

## ✅ Sanity

```bash
modal volume ls vn-history-artifacts
modal run modal_artifact_sanity.py
modal run modal_runtime_sanity.py
modal run full_modal_runtime_sanity.py  # legacy Qwen2.5 layout only
```

Artifact sanity kiểm tra layout, manifest/corpus count, index và ba model roles. Runtime sanity có thể tải model/GPU và phát sinh chi phí.

## 🔒 Quy tắc

- Corpus count, FAISS `ntotal` và BM25 manifest phải khớp.
- Không sửa manifest để né validation.
- Không force-add `.safetensors`, `.index`, corpus lớn hoặc SQLite vào Git.
- Upload config mới cần restart/redeploy container; runtime không hot-reload config.
- `modal_fix.py` và `full_modal_runtime_sanity.py` chỉ dành cho legacy merged-Qwen2.5 layout.
